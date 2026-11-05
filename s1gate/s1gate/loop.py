"""Step 8: close the loop - review queue, weekly report, next training set, versioned bundles.

    s1gate queue        escalated rows from the decision log -> a labelling queue
    s1gate label        a person labels the queue (writes corrections, source=correction)
    s1gate loop-report  coverage, escalation rate, overturned auto-decisions, per week
    s1gate build-train  corrections first (up-weighted), synthetic second, eval guarded
    s1gate bundle       model + calibration + thresholds + schema fingerprint under one name
    s1gate promote      a candidate bundle must match or beat the current one on frozen eval

The decision log holds row text: store it under the same retention and access rules as
the source POS files.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from .records import Record, dedupe, read_jsonl
from .schema import DecisionSchema


def read_log(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _day(ts: float) -> date:
    return datetime.fromtimestamp(ts, tz=timezone.utc).date()


def _period(ts: float, by: str) -> str:
    d = _day(ts)
    if by == "day":
        return d.isoformat()
    if by == "month":
        return f"{d.year}-{d.month:02d}"
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


# ---------------------------------------------------------------- review queue

def review_queue(log_rows: list[dict], only_escalated: bool = True, since: str | None = None,
                 questions: list[str] | None = None, exclude_ids: set[str] | None = None) -> list[Record]:
    """Rows for a person to label blind (no labels prefilled; what the gate said is in meta)."""
    since_d = date.fromisoformat(since) if since else None
    seen, out = set(exclude_ids or ()), []
    for row in log_rows:
        if row["id"] in seen or (since_d and _day(row["ts"]) < since_d):
            continue
        decisions = row["decisions"]
        qs = questions or list(decisions)
        if only_escalated and not any(decisions[q]["escalated"] for q in qs if q in decisions):
            continue
        seen.add(row["id"])
        out.append(Record(id=row["id"], text=row["text"], labels={}, source="real",
                          meta={"gate": {q: {k: d[k] for k in ("option", "model_option", "confidence",
                                                                "decided_by", "escalated") if k in d}
                                         for q, d in decisions.items()},
                                "logged": row["ts"], "schema_fingerprint": row.get("schema_fingerprint")}))
    return out


# ---------------------------------------------------------------- weekly report

def loop_report(log_rows: list[dict], corrections: list[Record], by: str = "week") -> dict:
    """Per period and question: coverage, escalations, and auto-decisions a person overturned.

    Falling coverage at a fixed tau is the earliest sign that distributor formats drifted.
    Overturned = a correction disagrees with an option the gate accepted on its own: that is
    an auto-commit error, the number the threshold target is supposed to bound.
    """
    latest: dict[str, Record] = {}
    for c in corrections:
        if c.id not in latest or c.meta.get("ts", 0) >= latest[c.id].meta.get("ts", 0):
            latest[c.id] = c
    periods: dict[str, dict] = defaultdict(lambda: defaultdict(Counter))
    overturned_conf: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in log_rows:
        p = _period(row["ts"], by)
        fix = latest.get(row["id"])
        for q, d in row["decisions"].items():
            c = periods[p][q]
            c["rows"] += 1
            how = d.get("decided_by") or ("fallback" if d["escalated"] else "gate")  # logs before decided_by
            c[how] += 1
            if fix and q in fix.labels:
                c["reviewed"] += 1
                if how == "gate" and fix.labels[q] != d["option"]:
                    c["overturned"] += 1
                    overturned_conf[(p, q)].append(float(d["confidence"]))
    out = {"by": by, "periods": {}}
    prev_cov: dict[str, float] = {}
    for p in sorted(periods):
        qs = {}
        for q, c in periods[p].items():
            n = c["rows"]
            cov = c["gate"] / n if n else 0.0
            auto = c["gate"]
            qs[q] = {"rows": n, "coverage": round(cov, 4),
                     "escalation_rate": round(1 - cov, 4),
                     "by": {k: c[k] for k in ("gate", "rule", "frontier", "fallback") if c[k]},
                     "reviewed": c["reviewed"], "overturned": c["overturned"],
                     "overturn_rate_of_auto": round(c["overturned"] / auto, 4) if auto else None,
                     "overturned_mean_confidence": (round(sum(overturned_conf[(p, q)]) / len(overturned_conf[(p, q)]), 4)
                                                    if overturned_conf[(p, q)] else None),
                     "coverage_change": round(cov - prev_cov[q], 4) if q in prev_cov else None}
            prev_cov[q] = cov
        out["periods"][p] = qs
    return out


# ---------------------------------------------------------------- next training set

def build_train(corrections: list[Record], synthetic: list[Record], correction_weight: int = 3,
                dedupe_threshold: float = 0.9) -> tuple[list[Record], dict]:
    """Corrections first (latest label per id, repeated `correction_weight` times), then synthetic.

    Corrections are real rows a person decided: worth more than any synthetic row, so they
    are up-weighted by repetition. Synthetic rows are deduped among themselves.
    """
    latest: dict[str, Record] = {}
    for c in corrections:
        if c.id not in latest or c.meta.get("ts", 0) >= latest[c.id].meta.get("ts", 0):
            latest[c.id] = c
    fixes = [c for c in latest.values() if c.labels]
    syn, dropped = dedupe(synthetic, dedupe_threshold) if synthetic else ([], [])
    out: list[Record] = []
    for k in range(max(1, correction_weight)):
        out.extend(c if k == 0 else c.model_copy(update={"id": f"{c.id}#w{k}"}) for c in fixes)
    out.extend(syn)
    stats = {"corrections": len(fixes), "correction_weight": correction_weight,
             "synthetic_in": len(synthetic), "synthetic_dropped_as_duplicates": len(dropped),
             "rows_out": len(out)}
    return out, stats


# ---------------------------------------------------------------- bundles and promotion

def bundle(name: str, schema: DecisionSchema, backend: str, model: str | None, calibration: str | None,
           thresholds: str | None, adapter: str | None = None, report: str | None = None,
           notes: str = "") -> dict:
    for path in (calibration, thresholds):
        if path:
            data = json.loads(Path(path).read_text())
            if data.get("schema_fingerprint") != schema.fingerprint():
                raise ValueError(f"{path} belongs to another schema version; refit before bundling")
    return {"name": name, "created": date.today().isoformat(), "schema": schema.name,
            "schema_version": schema.version, "schema_fingerprint": schema.fingerprint(),
            "backend": backend, "model": model, "adapter": adapter, "calibration": calibration,
            "thresholds": thresholds, "eval_report": report, "notes": notes}


def promote(candidate: dict, current: dict | None, nll_slack: float = 0.0,
            coverage_slack: float = 0.0, accuracy_slack: float = 0.0) -> tuple[bool, list[str]]:
    """Compare two eval reports (frozen eval, calibrated, with thresholds). Lower NLL and
    higher coverage at the deployed targets win; the candidate must not lose on either, and
    coverage bought with accuracy does not count: its auto-accepted rows must stay at or above
    the target (or at least as accurate as the current model's)."""
    if current is None:
        return True, ["no current model: candidate becomes the first version"]
    ok, notes = True, []
    cur = {q["question"]: q for q in current["questions"]}
    if candidate.get("data_sha256") and current.get("data_sha256") and candidate["data_sha256"] != current["data_sha256"]:
        return False, ["the two reports were computed on different eval files"]
    for q in candidate["questions"]:
        name = q["question"]
        if name not in cur:
            continue
        c = cur[name]
        if _num(q["nll"]) > _num(c["nll"]) + nll_slack:
            ok = False
            notes.append(f"{name}: NLL {q['nll']:.4f} worse than {c['nll']:.4f}")
        else:
            notes.append(f"{name}: NLL {q['nll']:.4f} vs {c['nll']:.4f} ok")
        cq, cc = q.get("at_threshold"), c.get("at_threshold")
        if cq and cc:
            if cq["coverage"] + coverage_slack < cc["coverage"]:
                ok = False
                notes.append(f"{name}: coverage {cq['coverage']:.3f} below {cc['coverage']:.3f}")
            else:
                notes.append(f"{name}: coverage {cq['coverage']:.3f} vs {cc['coverage']:.3f} ok")
            acc, cur_acc = cq.get("accuracy"), cc.get("accuracy")
            # the deployed target is the contract: a candidate fitted for a looser one is held to it
            targets = [t for t in (cq.get("target"), cc.get("target")) if t is not None]
            tgt = max(targets) if targets else None
            if cq.get("target") is not None and cc.get("target") is not None and cq["target"] < cc["target"]:
                notes.append(f"{name}: candidate thresholds fitted for target {cq['target']}, "
                             f"deployed target {cc['target']}; judged against {tgt}")
            if acc is not None:
                meets_target = tgt is not None and acc >= tgt
                # "not worse than current" only counts when the current model auto-accepts anything
                not_worse = cur_acc is not None and acc >= cur_acc - accuracy_slack
                if meets_target or not_worse:
                    notes.append(f"{name}: auto-accept accuracy {acc:.3f} ok")
                else:
                    ok = False
                    vs = "the current model auto-accepts nothing" if cur_acc is None else f"below the current {cur_acc:.3f}"
                    notes.append(f"{name}: auto-accept accuracy {acc:.3f} under target {tgt} and {vs} "
                                 f"({cq.get('auto_errors')} auto errors)")
        else:
            notes.append(f"{name}: no thresholds in one report; coverage not compared")
    return ok, notes


def _num(x) -> float:
    return float("inf") if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def load_corrections(path: str | Path | None) -> list[Record]:
    return read_jsonl(path) if path and Path(path).exists() else []
