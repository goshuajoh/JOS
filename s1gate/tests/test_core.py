import json
from pathlib import Path

import numpy as np
import pytest

from s1gate.backends.gliner import _parse
from s1gate.calibrate import apply_temperature, fit_temperature
from s1gate.cli import main
from s1gate.metrics import ece, evaluate_question
from s1gate.records import Record, decontaminate, dedupe, read_jsonl, validate_records
from s1gate.schema import DecisionSchema, Question
from s1gate.thresholds import NEVER, pick_threshold

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "configs" / "schema.pos.yaml"
EVAL = ROOT / "data" / "eval" / "eval.sample.jsonl"


def test_schema_loads_and_exports():
    s = DecisionSchema.load(SCHEMA)
    assert s.names == ["row_action", "needs_human", "severity"]
    assert s.question("needs_human").labels == ["yes", "no"]
    js = s.json_schema()
    assert "row_action" in js["properties"]
    assert len(s.fingerprint()) == 12


def test_schema_rejects_bad_noul_and_fallback():
    with pytest.raises(ValueError):
        Question(name="x", kind="noul", prompt="?", options=["y", "n"])
    with pytest.raises(ValueError):
        Question(name="x", prompt="?", options=["a", "b"], fallback="c")


def test_sample_eval_is_valid():
    s = DecisionSchema.load(SCHEMA)
    assert validate_records(read_jsonl(EVAL), s) == []


def test_metrics_perfect_and_uniform():
    labels = ["a", "b"]
    perfect = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]
    rep = evaluate_question("q", perfect, ["a", "b"], labels)
    assert rep.accuracy == 1.0 and rep.ece < 1e-9
    uniform = [{"a": 0.5, "b": 0.5}] * 4
    rep = evaluate_question("q", uniform, ["a", "b", "a", "b"], labels)
    assert abs(rep.nll - np.log(2)) < 1e-6


def test_ece_detects_overconfidence():
    conf = np.full(100, 0.99)
    correct = np.r_[np.ones(70), np.zeros(30)]
    assert abs(ece(conf, correct) - 0.29) < 1e-6


def test_temperature_softens_overconfident_model():
    rng = np.random.default_rng(0)
    n, k = 400, 4
    gold = rng.integers(0, k, n)
    # 70% right but always 97% sure
    pred = np.where(rng.random(n) < 0.7, gold, (gold + 1) % k)
    P = np.full((n, k), 0.01)
    P[np.arange(n), pred] = 0.97
    t = fit_temperature(P, gold)
    assert t > 1.5
    soft = apply_temperature({"a": 0.97, "b": 0.01, "c": 0.01, "d": 0.01}, t)
    assert 0.6 < soft["a"] < 0.85


def test_threshold_picks_lowest_tau_meeting_target():
    conf = np.array([0.99] * 50 + [0.8] * 50)
    correct = np.array([1.0] * 50 + [0.0] * 50)
    out = pick_threshold(conf, correct, target=0.95, min_support=20)
    assert out["tau"] == 0.99 and out["coverage"] == 0.5
    assert pick_threshold(conf, correct * 0, 0.95)["tau"] == NEVER


def test_dedupe_and_decontaminate():
    a = Record(id="1", text="distributor=X; qty=100; ref=A1")
    b = Record(id="2", text="distributor=X;  qty=100; ref=A1 ")
    c = Record(id="3", text="something entirely different here")
    kept, dropped = dedupe([a, b, c])
    assert [r.id for r in kept] == ["1", "3"] and dropped[0].id == "2"
    kept, leaked = decontaminate([a, c], [b])
    assert [r.id for r in kept] == ["3"]


def test_split_keeps_identical_texts_on_one_side():
    from s1gate.records import split

    header = "distributor=X; period=Period; customer=Customer; qty=Qty"
    rows = [Record(id=f"h{i}", text=(header if i % 2 else header.upper() + " "), labels={"row_action": "no_op"})
            for i in range(6)]
    rows += [Record(id=f"r{i}", text=f"distributor=X; qty={i}", labels={"row_action": "commit"}) for i in range(40)]
    for seed in range(20):
        rest, held = split(rows, 0.5, seed=seed, stratify_on="row_action")
        assert sorted(r.id for r in rest + held) == sorted(r.id for r in rows)
        heads = {r.id for r in held if r.id.startswith("h")}
        assert heads in (set(), {f"h{i}" for i in range(6)})  # all six copies move together
        assert sum(r.id.startswith("r") for r in held) == 20
    # with no repeated texts the split is the plain stratified shuffle it always was
    singles = rows[6:]
    rest, held = split(singles, 0.25, seed=3, stratify_on="row_action")
    import random
    order = singles[:]
    random.Random(3).shuffle(order)
    assert [r.id for r in held] == [r.id for r in order[:10]]


def test_gliner_parse_multilabel_recovers_softmax():
    # sigmoid(2)=0.881, sigmoid(0)=0.5 -> softmax over logits (2, 0) = (0.881, 0.119)
    out = _parse([{"label": "a", "confidence": 0.8807970779778823},
                  {"label": "b", "confidence": 0.5}], ["a", "b"])
    assert abs(out["a"] - 0.8807970779778823) < 1e-6


def test_cli_pipeline_with_keyword_backend(tmp_path):
    preds, report = tmp_path / "preds.jsonl", tmp_path / "report.json"
    assert main(["validate", "--schema", str(SCHEMA), "--data", str(EVAL)]) == 0
    assert main(["eval", "--schema", str(SCHEMA), "--data", str(EVAL), "--backend", "keyword",
                 "--preds-out", str(preds), "--report-out", str(report)]) == 0
    assert len(preds.read_text().splitlines()) == 26
    cal = tmp_path / "cal.json"
    assert main(["calibrate", "--schema", str(SCHEMA), "--preds", str(preds), "--out", str(cal)]) == 0
    thr = tmp_path / "thr.json"
    assert main(["thresholds", "--schema", str(SCHEMA), "--preds", str(preds), "--calibration", str(cal),
                 "--target", "0.9", "--min-support", "5", "--out", str(thr)]) == 0
    data = json.loads(thr.read_text())
    assert set(data["thresholds"]) == {"row_action", "needs_human", "severity"}


def test_gate_escalates_to_fallback():
    from s1gate.backends.keyword import KeywordBackend
    from s1gate.gate import Gate

    s = DecisionSchema.load(SCHEMA)
    gate = Gate(s, KeywordBackend(), thresholds={"row_action": NEVER})
    d = gate.decide(read_jsonl(EVAL)[0].text)
    assert d["row_action"].escalated and d["row_action"].option == "ask_human"
    assert not d["severity"].escalated


def test_http_service(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from s1gate.backends.keyword import KeywordBackend
    from s1gate.gate import Gate
    from s1gate.serve import create_app

    s = DecisionSchema.load(SCHEMA)
    gate = Gate(s, KeywordBackend(), thresholds={"row_action": NEVER}, log_path=tmp_path / "log.jsonl")
    c = TestClient(create_app(gate, str(tmp_path / "corr.jsonl")))
    assert c.get("/health").json()["fingerprint"] == s.fingerprint()
    r = c.post("/decide", json={"text": "qty=1100; spq=250"}).json()
    assert r["escalated"] and r["needs_person"] and r["decisions"]["row_action"]["option"] == "ask_human"
    assert r["decisions"]["row_action"]["decided_by"] == "fallback"
    assert c.post("/feedback", json={"id": "x", "text": "t", "labels": {"row_action": "duplicate"}}).json()["ok"]
    assert c.post("/feedback", json={"id": "x", "text": "t", "labels": {"row_action": "nope"}}).status_code == 422
    probs = c.post("/predict", json={"text": "t", "questions": [
        {"name": "q", "prompt": "?", "options": {"a": "", "b": ""}}]}).json()["probs"]["q"]
    assert abs(sum(probs.values()) - 1) < 1e-9
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 1
    assert read_jsonl(tmp_path / "corr.jsonl")[0].source == "correction"


class _FakeTeacher:
    """Writes records on request; when labelling blind, gets every third one wrong."""
    model = "fake"

    def __init__(self):
        self.n = 0

    def complete(self, system, user, temperature=1.0):
        import re as _re
        if "Records:" in user:  # blind labelling call
            idx = [int(i) for i in _re.findall(r"^\[(\d+)\]", user, flags=_re.MULTILINE)]
            return json.dumps([{"i": i, "row_action": "commit" if i % 3 else "duplicate"} for i in idx])
        k = int(_re.search(r"Write (\d+) distinct records", user).group(1))
        out = []
        for _ in range(k):
            self.n += 1
            out.append({"text": f"distributor=D{self.n}; qty={self.n * 250}", "labels": {"row_action": "commit"}})
        return "```json\n" + json.dumps(out) + "\n```"


def test_synth_loop_with_blind_verification():
    from s1gate.synth import Recipe, Scenario, generate

    s = DecisionSchema.load(SCHEMA)
    recipe = Recipe(domain="d", record_format="f", scenarios=[Scenario(target={"row_action": "commit"}, hint="h")],
                    n_per_scenario=6, batch_size=3)
    out = generate(s, recipe, _FakeTeacher(), log=lambda *_: None)
    assert len(out) == 6
    assert all(r.source == "synthetic" and r.labels == {"row_action": "commit"} for r in out)


def test_gliner_training_rows_shuffle_options():
    from s1gate.train_gliner import to_gliner_rows

    s = DecisionSchema.load(SCHEMA)
    rows = to_gliner_rows(read_jsonl(EVAL), s, augment=2)
    assert len(rows) == 52
    first = rows[0]["output"]["classifications"][0]
    assert first["task"] == "row_action" and first["true_label"] == ["commit"]
    assert sorted(first["labels"]) == sorted(s.question("row_action").labels)
    assert "label_descriptions" in first and "prompt" in first


class _TwoTeachers:
    """Generation and blind labelling by alternating fake teachers; pairs too."""

    def __init__(self, name):
        self.model = name
        self.calls = []

    def complete(self, system, user, temperature=1.0):
        import re as _re
        self.calls.append("label" if "Records:" in user else "gen")
        if "Records:" in user:
            idx = [int(i) for i in _re.findall(r"^\[(\d+)\]", user, flags=_re.MULTILINE)]
            return json.dumps([{"i": i, "row_action": "commit"} for i in idx])
        m = _re.search(r"Write (\d+) distinct PAIRS", user)
        if m:
            return json.dumps([{"a": f"pair {self.model} {len(self.calls)} {j} a", "b": f"pair {self.model} {len(self.calls)} {j} b"}
                               for j in range(int(m.group(1)))])
        k = int(_re.search(r"Write (\d+) distinct records", user).group(1))
        return json.dumps([{"text": f"row {self.model} {len(self.calls)} {j}"} for j in range(k)])


def test_synth_rotates_teachers_and_keeps_pairs_whole():
    from s1gate.synth import Pair, Recipe, Scenario, generate

    s = DecisionSchema.load(SCHEMA)
    commit = {"row_action": "commit"}
    recipe = Recipe(domain="d", record_format="f", scenarios=[Scenario(target=commit, hint="h", noise="none")],
                    pairs=[Pair(id="H", change="c", a=commit, b=commit)], n_per_scenario=4, pairs_per_type=2,
                    batch_size=2)
    a, b = _TwoTeachers("A"), _TwoTeachers("B")
    out = generate(s, recipe, [a, b], log=lambda *_: None)
    assert len([r for r in out if "pair" not in r.meta]) == 4
    pairs = [r for r in out if "pair" in r.meta]
    assert len(pairs) == 4 and {r.meta["side"] for r in pairs} == {"a", "b"}
    assert "gen" in a.calls and "label" in a.calls and "gen" in b.calls and "label" in b.calls
