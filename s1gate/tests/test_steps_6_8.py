"""Calibration methods, threshold modes and rules, the gate's escalation paths, the loop tools."""
import json
from pathlib import Path

import numpy as np
import pytest

from s1gate.backends.base import Backend
from s1gate.backends.frontier import parse_answer
from s1gate.backends.keyword import KeywordBackend
from s1gate.calibrate import apply_calibration, fit_all, fit_vector
from s1gate.cli import main
from s1gate.gate import Gate, record_correction
from s1gate.loop import build_train, loop_report, promote, read_log, review_queue
from s1gate.records import Record, decontaminate, dedupe, jaccard, read_jsonl, shingles
from s1gate.report import build_report
from s1gate.schema import DecisionSchema
from s1gate.thresholds import NEVER, parse_rules, pick_for_coverage, tradeoff_table

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "configs" / "schema.pos.yaml"
EVAL = ROOT / "data" / "eval" / "eval.sample.jsonl"


class Fixed(Backend):
    """Returns the same probabilities for every row (per question)."""
    name = "fixed"

    def __init__(self, probs):
        self.probs = probs

    def predict(self, text, schema):
        return {q.name: dict(self.probs[q.name]) if q.name in self.probs else
                {o: 1 / len(q.labels) for o in q.labels} for q in schema.questions}


def _skewed(n=600, k=4, seed=0, bias_option=2):
    """A model that is fine except it under-rates one option."""
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), gold] += 2.5
    logits[:, bias_option] -= 1.5
    P = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return P, gold


def test_vector_scaling_fixes_a_single_biased_option():
    P, gold = _skewed()
    labels = ["a", "b", "c", "d"]
    preds = [{"id": str(i), "probs": {"q": dict(zip(labels, p))}, "gold": {"q": labels[g]}} for i, (p, g) in enumerate(zip(P, gold))]
    s = DecisionSchema(name="t", questions=[{"name": "q", "prompt": "?", "options": labels}])
    temp = fit_all(s, preds, "temperature")
    vec = fit_all(s, preds, "vector")
    assert vec["detail"]["q"]["nll_after"] < temp["detail"]["q"]["nll_after"]
    assert vec["biases"]["q"]["c"] > 0.5  # pushes the under-rated option back up
    t, b = fit_vector(P, gold)
    out = apply_calibration(dict(zip(labels, P[0])), t, dict(zip(labels, b)))
    assert abs(sum(out.values()) - 1) < 1e-9


def test_threshold_coverage_mode_and_tradeoff():
    conf = np.linspace(0.5, 1.0, 100)
    correct = (conf > 0.7).astype(float)
    picked = pick_for_coverage(conf, correct, 0.5)
    assert picked["coverage"] >= 0.5 and picked["accuracy_at_tau"] == 1.0
    table = tradeoff_table(conf, correct, 10)
    assert [r["target"] for r in table][:2] == [0.80, 0.85]
    assert all(r["coverage"] >= table[-1]["coverage"] for r in table)
    assert parse_rules(["row_action=compliance_review"], ["row_action.normalize=0.8"]) == {
        "row_action": {"always_escalate": ["compliance_review"], "option_tau": {"normalize": 0.8}}}


def _gate(probs, **kw):
    return Gate(DecisionSchema.load(SCHEMA), Fixed(probs), **kw)


def test_gate_rules_option_tau_and_always_escalate():
    probs = {"row_action": {"normalize": 0.85, "commit": 0.15}}
    g = _gate(probs, thresholds={"row_action": 0.95})
    assert g.decide("x")["row_action"].decided_by == "fallback"
    g = _gate(probs, thresholds={"row_action": 0.95}, rules={"row_action": {"option_tau": {"normalize": 0.8}}})
    d = g.decide("x")["row_action"]
    assert (d.option, d.decided_by, d.tau) == ("normalize", "gate", 0.8)
    g = _gate({"row_action": {"compliance_review": 0.999, "commit": 0.001}},
              rules={"row_action": {"always_escalate": ["compliance_review"]}})
    d = g.decide("x")["row_action"]
    assert d.escalated and d.decided_by == "rule" and d.option == "ask_human" and d.model_option == "compliance_review"
    with pytest.raises(ValueError):
        _gate(probs, rules={"row_action": {"always_escalate": ["nope"]}})


def test_gate_escalates_to_frontier_then_falls_back():
    unsure = {"row_action": {"commit": 0.5, "normalize": 0.5}}
    frontier = Fixed({"row_action": {"normalize": 0.95, "commit": 0.05}})
    g = _gate(unsure, thresholds={"row_action": 0.9}, escalator=frontier, escalator_min_confidence=0.8)
    d = g.decide("x")["row_action"]
    assert d.escalated and d.decided_by == "frontier" and d.option == "normalize"
    timid = Fixed({"row_action": {"normalize": 0.6, "commit": 0.4}})
    d = _gate(unsure, thresholds={"row_action": 0.9}, escalator=timid).decide("x")["row_action"]
    assert d.decided_by == "fallback" and d.option == "ask_human"

    class Down(Backend):
        def predict(self, text, schema):
            raise ConnectionError("unreachable")
    d = _gate(unsure, thresholds={"row_action": 0.9}, escalator=Down()).decide("x")["row_action"]
    assert d.decided_by == "fallback"


def test_gate_typed_decision_model():
    g = _gate({"row_action": {"no_op": 0.9, "commit": 0.1}})
    typed = g.decision_model(g.decide("x"))
    assert typed.row_action.value == "no_op"


def test_frontier_answer_parsing_is_typed():
    s = DecisionSchema.load(SCHEMA)
    out = parse_answer('```json\n{"row_action": {"option": "duplicate", "confidence": 0.97}, '
                       '"needs_human": {"option": "no"}, "severity": {"option": "9"}}\n```', s)
    assert max(out["row_action"], key=out["row_action"].get) == "duplicate"
    assert abs(out["row_action"]["duplicate"] - 0.97) < 1e-9
    assert len(set(out["severity"].values())) == 1  # not an option -> uniform -> escalates
    assert parse_answer("not json", s)["needs_human"] == {"yes": 0.5, "no": 0.5}


def test_dedupe_is_label_aware_and_exact():
    a = Record(id="a", text="distributor=X; qty=1000; spq=250; ref=R1", labels={"row_action": "commit"})
    b = Record(id="b", text="distributor=X; qty=1100; spq=250; ref=R1", labels={"row_action": "pack_qty_issue"})
    c = Record(id="c", text="distributor=X; qty=1000; spq=250; ref=R1 ", labels={"row_action": "commit"})
    d = Record(id="d", text="distributor=X; qty=1000; spq=250; ref=R1", labels={"row_action": "duplicate"})
    kept, dropped = dedupe([a, b, c])
    assert [r.id for r in kept] == ["a", "b"]           # the hard pair survives, the copy does not
    kept, _ = dedupe([a, b], label_aware=False)
    assert jaccard(shingles(a.text), shingles(b.text)) < 0.9 or [r.id for r in kept] == ["a"]
    kept, dropped = dedupe([a, d])
    assert kept == [] and {r.id for r in dropped} == {"a", "d"}  # same text, conflicting labels


def test_prefix_dedupe_matches_brute_force():
    rng = np.random.default_rng(1)
    base = [r.text for r in read_jsonl(EVAL)]
    texts = []
    for _ in range(300):
        t = list(base[rng.integers(len(base))])
        for _ in range(rng.integers(0, 12)):
            t[rng.integers(len(t))] = chr(rng.integers(97, 123))
        texts.append("".join(t))
    recs = [Record(id=str(i), text=t) for i, t in enumerate(texts)]
    for thr in (0.8, 0.9, 0.95):
        kept, _ = dedupe(recs, thr, label_aware=False)
        ref, ref_sh = [], []
        for r in recs:
            sh = shingles(r.text)
            if any(jaccard(sh, o) >= thr for o in ref_sh):
                continue
            ref.append(r.id)
            ref_sh.append(sh)
        assert [r.id for r in kept] == ref
        held = recs[:20]
        k1, _ = decontaminate(recs[20:], held, thr)
        brute = [r.id for r in recs[20:] if not any(jaccard(shingles(r.text), shingles(h.text)) >= thr for h in held)]
        assert [r.id for r in k1] == brute


def _log(tmp_path):
    s = DecisionSchema.load(SCHEMA)
    log = tmp_path / "decisions.jsonl"
    gate = Gate(s, KeywordBackend(), thresholds={"row_action": 0.5}, log_path=log)
    rows = read_jsonl(EVAL)
    gate.decide_batch([r.text for r in rows], [r.id for r in rows])
    return s, log, rows


def test_queue_report_and_build_train(tmp_path):
    s, log, rows = _log(tmp_path)
    logged = read_log(log)
    queue = review_queue(logged)
    assert queue and all(not r.labels and "gate" in r.meta for r in queue)
    corr = tmp_path / "corr.jsonl"
    auto = [r for r in logged if r["decisions"]["row_action"]["decided_by"] == "gate"]
    for r in rows[:5]:
        record_correction(corr, r.id, r.text, r.labels, reviewer="t")
    rep = loop_report(logged, read_jsonl(corr))
    week = next(iter(rep["periods"].values()))
    assert week["row_action"]["rows"] == 26 and week["row_action"]["reviewed"] == 5
    assert 0 <= week["row_action"]["coverage"] <= 1 and len(auto) == week["row_action"]["by"].get("gate", 0)
    train, stats = build_train(read_jsonl(corr), [Record(id="s1", text="synthetic row one", labels={})], 3)
    assert stats["corrections"] == 5 and len(train) == 16 and train[0].source == "correction"


def test_promote_compares_nll_and_coverage():
    cur = {"questions": [{"question": "q", "nll": 0.5, "at_threshold": {"coverage": 0.4}}]}
    better = {"questions": [{"question": "q", "nll": 0.4, "at_threshold": {"coverage": 0.45}}]}
    worse = {"questions": [{"question": "q", "nll": 0.4, "at_threshold": {"coverage": 0.3}}]}
    assert promote(better, cur)[0] and not promote(worse, cur)[0] and promote(better, None)[0]


def test_report_html_and_full_cli_loop(tmp_path):
    preds, cal, thr = tmp_path / "p.jsonl", tmp_path / "c.json", tmp_path / "t.json"
    assert main(["eval", "--schema", str(SCHEMA), "--data", str(EVAL), "--backend", "keyword",
                 "--preds-out", str(preds), "--report-out", str(tmp_path / "r0.json")]) == 0
    assert main(["calibrate", "--schema", str(SCHEMA), "--preds", str(preds), "--out", str(cal),
                 "--method", "vector", "--frozen"]) == 0
    assert main(["thresholds", "--schema", str(SCHEMA), "--preds", str(preds), "--calibration", str(cal),
                 "--target", "0.9", "--min-support", "5", "--always-escalate", "row_action=compliance_review",
                 "--option-tau", "row_action.normalize=0.5", "--out", str(thr), "--frozen"]) == 0
    assert json.loads(thr.read_text())["rules"]["row_action"]["always_escalate"] == ["compliance_review"]
    assert main(["eval", "--schema", str(SCHEMA), "--data", str(EVAL), "--backend", "keyword",
                 "--calibration", str(cal), "--thresholds", str(thr), "--report-out", str(tmp_path / "r1.json")]) == 0
    r1 = json.loads((tmp_path / "r1.json").read_text())
    assert "at_threshold" in r1["questions"][0] and "coverage_at_in_sample" in r1["questions"][0]
    assert main(["compare", "--reports", str(tmp_path / "r0.json"), str(tmp_path / "r1.json")]) == 0
    out = tmp_path / "report.html"
    assert main(["report", "--schema", str(SCHEMA), "--preds", str(preds), "--calibration", str(cal),
                 "--thresholds", str(thr), "--out", str(out)]) == 0
    page = out.read_text()
    assert page.count("<svg") >= 7 and "row_action" in page and "prefers-color-scheme" in page
    s = DecisionSchema.load(SCHEMA)
    assert build_report(s, [], title="empty").startswith("<!doctype html>")
    bundle = tmp_path / "b.json"
    assert main(["bundle", "--schema", str(SCHEMA), "--name", "v1", "--backend", "keyword",
                 "--calibration", str(cal), "--thresholds", str(thr), "--out", str(bundle)]) == 0
    assert json.loads(bundle.read_text())["schema_fingerprint"] == s.fingerprint()
    assert main(["promote", "--candidate", str(tmp_path / "r1.json")]) == 0


def test_never_threshold_escalates_everything():
    g = _gate({"row_action": {"commit": 0.99, "normalize": 0.01}}, thresholds={"row_action": NEVER})
    assert g.decide("x")["row_action"].decided_by == "fallback"


def test_mine_confusions_boosts_weights_and_drafts_pairs(tmp_path):
    import yaml

    from s1gate.synth import Recipe, mine_confusions
    recipe = yaml.safe_load((ROOT / "configs" / "synth.pos.yaml").read_text())
    report = {"questions": [{"question": "row_action", "confusion": {
        "duplicate": {"duplicate": 10, "commit": 6}, "commit": {"commit": 30, "normalize": 1}}}]}
    new, found = mine_confusions(report, recipe, min_count=3)
    assert found == [{"question": "row_action", "gold": "duplicate", "pred": "commit", "count": 6, "rate": 0.375}]
    w = {s["id"]: s.get("weight", 1.0) for s in new["scenarios"]}
    old = {s["id"]: s.get("weight", 1.0) for s in recipe["scenarios"]}
    assert w["D2"] > old["D2"] and w["C1"] == old["C1"]
    assert "mined_pairs_todo" not in new  # duplicate->commit already has a hard pair (H04)
    out = tmp_path / "r2.yaml"
    out.write_text(yaml.safe_dump(new))
    Recipe.load(str(out))  # still a valid recipe
