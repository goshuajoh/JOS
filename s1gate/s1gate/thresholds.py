"""Step 7: escalation thresholds.

For each question, pick the LOWEST confidence cut-off tau such that the answers the gate
auto-accepts (confidence >= tau) are right at least `target` of the time. Everything under
tau goes to the question's fallback (ASK_HUMAN, or a frontier model).

Lower tau = more rows automated (coverage) but more errors slip through. The target is a
business number: "we accept 1 wrong row in 100 auto-committed" means target 0.99.
`min_support` stops a tau being set from a handful of lucky rows.

The other direction works too: `coverage` asks for the tau that automates a given share of
rows and reports the accuracy you get there. Every question also gets a trade-off table
(tau, coverage, accuracy at several targets) so the cut-off is chosen, not guessed.

Policy rules live next to the cut-offs and are applied by the gate after the model:
  always_escalate: options that always go to a person, whatever the confidence
  option_tau:      a cut-off for one option that differs from the question's tau
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from .calibrate import apply_calibration
from .metrics import top_and_correct
from .schema import DecisionSchema

NEVER = 1.01  # a tau no probability reaches: escalate everything
TABLE_TARGETS = (0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99)


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


def pick_for_coverage(conf: np.ndarray, correct: np.ndarray, coverage: float) -> dict:
    """The highest tau that still auto-accepts at least `coverage` of the rows."""
    if len(conf) == 0:
        return {"tau": NEVER, "coverage": 0.0, "accuracy_at_tau": None, "note": "no rows"}
    order = np.argsort(-conf, kind="stable")
    c, ok = conf[order], correct[order]
    k = min(len(c), max(1, int(np.ceil(coverage * len(c)))))
    while k < len(c) and c[k] == c[k - 1]:  # include ties: tau keeps everything >= it
        k += 1
    return {"tau": round(float(c[k - 1]), 4), "coverage": round(k / len(c), 4),
            "accuracy_at_tau": round(float(np.mean(ok[:k])), 4)}


def tradeoff_table(conf: np.ndarray, correct: np.ndarray, min_support: int = 20,
                   targets=TABLE_TARGETS) -> list[dict]:
    return [{"target": t, **{k: v for k, v in pick_threshold(conf, correct, t, min_support).items()
                             if k != "note"}} for t in targets]


def fit_all(schema: DecisionSchema, preds: list[dict], temperatures: dict[str, float],
            target: float | dict[str, float], min_support: int = 20,
            biases: dict[str, dict[str, float]] | None = None,
            coverage: dict[str, float] | None = None,
            rules: dict[str, dict] | None = None) -> dict:
    out = {}
    biases = biases or {}
    coverage = coverage or {}
    for q in schema.questions:
        rows = [p for p in preds if q.name in p.get("gold", {}) and q.name in p["probs"]]
        tgt = target.get(q.name, 0.95) if isinstance(target, dict) else target
        if not rows:
            out[q.name] = {"tau": NEVER, "target": tgt, "note": "no labelled rows"}
            continue
        t = temperatures.get(q.name, 1.0)
        probs = [apply_calibration(r["probs"][q.name], t, biases.get(q.name)) for r in rows]
        conf, correct = top_and_correct(probs, [r["gold"][q.name] for r in rows], q.labels)
        if q.name in coverage:
            picked = {"coverage_target": coverage[q.name], **pick_for_coverage(conf, correct, coverage[q.name])}
        else:
            picked = {"target": tgt, **pick_threshold(conf, correct, tgt, min_support)}
        out[q.name] = {"n": len(rows), **picked,
                       "tradeoff": tradeoff_table(conf, correct, min(min_support, len(rows)))}
    result = {"schema": schema.name, "schema_fingerprint": schema.fingerprint(),
              "fitted": date.today().isoformat(), "min_support": min_support, "thresholds": out}
    if rules:
        for qn, rule in rules.items():
            q = schema.question(qn)
            for o in list(rule.get("always_escalate", [])) + list(rule.get("option_tau", {})):
                if o not in q.options:
                    raise ValueError(f"rule for {qn}: {o!r} is not an option")
        result["rules"] = rules
    return result


def _load(path: str | Path, schema: DecisionSchema) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("schema_fingerprint") != schema.fingerprint():
        raise ValueError(f"{path} was fitted for a different schema version; re-run thresholds")
    return data


def load_thresholds(path: str | Path | None, schema: DecisionSchema) -> dict[str, float]:
    if not path:
        return {}
    return {k: float(v["tau"]) for k, v in _load(path, schema)["thresholds"].items()}


def load_rules(path: str | Path | None, schema: DecisionSchema) -> dict[str, dict]:
    """Policy rules stored with the thresholds ({} when none)."""
    if not path:
        return {}
    return _load(path, schema).get("rules", {})


def parse_rules(always_escalate: list[str] | None, option_tau: list[str] | None) -> dict[str, dict]:
    """CLI form: --always-escalate row_action=compliance_review  --option-tau row_action.normalize=0.8"""
    rules: dict[str, dict] = {}
    for item in always_escalate or []:
        q, _, opts = item.partition("=")
        rules.setdefault(q, {}).setdefault("always_escalate", []).extend(o for o in opts.split(",") if o)
    for item in option_tau or []:
        key, _, value = item.partition("=")
        q, _, o = key.partition(".")
        rules.setdefault(q, {}).setdefault("option_tau", {})[o] = float(value)
    return rules
