"""Step 7: escalation thresholds.

For each question, pick the LOWEST confidence cut-off tau such that the answers the gate
auto-accepts (confidence >= tau) are right at least `target` of the time. Everything under
tau goes to the question's fallback (ASK_HUMAN, or a frontier model).

Lower tau = more rows automated (coverage) but more errors slip through. The target is a
business number: "we accept 1 wrong row in 100 auto-committed" means target 0.99.
`min_support` stops a tau being set from a handful of lucky rows.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from .calibrate import apply_temperature
from .metrics import top_and_correct
from .schema import DecisionSchema

NEVER = 1.01  # a tau no probability reaches: escalate everything


def pick_threshold(conf: np.ndarray, correct: np.ndarray, target: float,
                   min_support: int = 20) -> dict:
    order = np.argsort(-conf, kind="stable")
    c, ok = conf[order], correct[order]
    cum_acc = np.cumsum(ok) / np.arange(1, len(c) + 1)
    best_k = 0
    for k in range(min_support, len(c) + 1):
        # only cut between distinct confidence values, so tau keeps exactly the top k
        if k < len(c) and c[k] == c[k - 1]:
            continue
        if cum_acc[k - 1] >= target:
            best_k = k
    if best_k == 0:
        return {"tau": NEVER, "coverage": 0.0, "accuracy_at_tau": None,
                "note": f"no cut-off reaches {target} with >= {min_support} rows"}
    return {"tau": round(float(c[best_k - 1]), 4), "coverage": round(best_k / len(c), 4),
            "accuracy_at_tau": round(float(cum_acc[best_k - 1]), 4)}


def fit_all(schema: DecisionSchema, preds: list[dict], temperatures: dict[str, float],
            target: float | dict[str, float], min_support: int = 20) -> dict:
    out = {}
    for q in schema.questions:
        rows = [p for p in preds if q.name in p.get("gold", {}) and q.name in p["probs"]]
        tgt = target.get(q.name, 0.95) if isinstance(target, dict) else target
        if not rows:
            out[q.name] = {"tau": NEVER, "target": tgt, "note": "no labelled rows"}
            continue
        t = temperatures.get(q.name, 1.0)
        probs = [apply_temperature(r["probs"][q.name], t) for r in rows]
        conf, correct = top_and_correct(probs, [r["gold"][q.name] for r in rows], q.labels)
        out[q.name] = {"target": tgt, "n": len(rows), **pick_threshold(conf, correct, tgt, min_support)}
    return {"schema": schema.name, "schema_fingerprint": schema.fingerprint(),
            "fitted": date.today().isoformat(), "thresholds": out}


def load_thresholds(path: str | Path | None, schema: DecisionSchema) -> dict[str, float]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    if data.get("schema_fingerprint") != schema.fingerprint():
        raise ValueError(f"{path} was fitted for a different schema version; re-run thresholds")
    return {k: float(v["tau"]) for k, v in data["thresholds"].items()}
