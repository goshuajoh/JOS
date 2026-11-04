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
from .schema import DecisionSchema

NEVER = 1.01  # a tau no probability reaches: escalate everything
TABLE_TARGETS = (0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99)


def gate_view(probs: list[dict[str, float]], golds: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(confidence, top option, correct) exactly as the gate computes them.

    The gate takes max(p, key=p.get) of the calibrated dict and compares that float with tau,
    so thresholds are fitted on the same floats (no re-normalisation, no rounding) and a stored
    tau means the same cut-off at runtime.
    """
    tops = [max(p, key=p.get) for p in probs]
    conf = np.array([p[t] for p, t in zip(probs, tops)], dtype=float)
    correct = np.array([float(t == g) for t, g in zip(tops, golds)])
    return conf, tops, correct


def auto_mask(conf: np.ndarray, tops: list[str], tau: float, rule: dict | None = None) -> np.ndarray:
    """Rows the gate auto-accepts under tau and the question's rules (same logic as Gate)."""
    rule = rule or {}
    always = set(rule.get("always_escalate", ()))
    otau = rule.get("option_tau", {})
    return np.array([t not in always and c >= float(otau.get(t, tau)) for c, t in zip(conf, tops)], dtype=bool)


def _partition(conf, tops, rule):
    """fixed: auto-accepted by an option's own tau whatever the question tau; free: decided by it."""
    n = len(conf)
    if not rule or tops is None:
        return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)
    always = set(rule.get("always_escalate", ()))
    otau = rule.get("option_tau", {})
    fixed = np.array([t not in always and t in otau and c >= float(otau[t]) for c, t in zip(conf, tops)], dtype=bool)
    free = np.array([t not in always and t not in otau for t in tops], dtype=bool)
    return fixed, free


def pick_threshold(conf: np.ndarray, correct: np.ndarray, target: float, min_support: int = 20,
                   tops: list[str] | None = None, rule: dict | None = None) -> dict:
    """Lowest tau whose auto-accepted rows (rules applied) are right at least `target` of the time."""
    n = len(conf)
    fixed, free = _partition(conf, tops, rule)
    f_n, f_ok = int(fixed.sum()), float(correct[fixed].sum())
    order = np.argsort(-conf[free], kind="stable")
    c, ok = conf[free][order], correct[free][order]
    cum_ok = np.concatenate([[0.0], np.cumsum(ok)])
    best_k = None
    for k in range(0, len(c) + 1):
        # only cut between distinct confidence values, so tau keeps exactly the top k
        if 0 < k < len(c) and c[k] == c[k - 1]:
            continue
        total = f_n + k
        if total < min_support or total == 0:
            continue
        if (f_ok + cum_ok[k]) / total >= target:
            best_k = k
    if best_k is None:
        out = {"tau": NEVER, "coverage": round(f_n / n, 4) if n else 0.0,
               "accuracy_at_tau": round(f_ok / f_n, 4) if f_n else None,
               "note": f"no cut-off reaches {target} with >= {min_support} rows"}
        if f_n:
            out["note"] += "; rows admitted by option_tau rules are still auto-accepted"
        return out
    tau = float(c[best_k - 1]) if best_k else NEVER
    total = f_n + best_k
    return {"tau": tau, "coverage": round(total / n, 4), "accuracy_at_tau": round((f_ok + cum_ok[best_k]) / total, 4)}


def pick_for_coverage(conf: np.ndarray, correct: np.ndarray, coverage: float,
                      tops: list[str] | None = None, rule: dict | None = None) -> dict:
    """The highest tau that still auto-accepts at least `coverage` of the rows (rules applied)."""
    n = len(conf)
    if n == 0:
        return {"tau": NEVER, "coverage": 0.0, "accuracy_at_tau": None, "note": "no rows"}
    fixed, free = _partition(conf, tops, rule)
    f_n, f_ok = int(fixed.sum()), float(correct[fixed].sum())
    order = np.argsort(-conf[free], kind="stable")
    c, ok = conf[free][order], correct[free][order]
    need = int(np.ceil(coverage * n - 1e-9))  # 0.07 * 100 is 7.000000000000001
    k = min(len(c), max(0, need - f_n))
    if k == 0 and f_n == 0 and len(c):
        k = 1
    while 0 < k < len(c) and c[k] == c[k - 1]:  # include ties: tau keeps everything >= it
        k += 1
    total = f_n + k
    note = {} if total >= need else {"note": f"rules leave at most {total / n:.1%} of rows automatable"}
    return {"tau": float(c[k - 1]) if k else NEVER, "coverage": round(total / n, 4),
            "accuracy_at_tau": round((f_ok + float(ok[:k].sum())) / total, 4) if total else None, **note}


def tradeoff_table(conf: np.ndarray, correct: np.ndarray, min_support: int = 20, targets=TABLE_TARGETS,
                   tops: list[str] | None = None, rule: dict | None = None) -> list[dict]:
    return [{"target": t, **{k: v for k, v in pick_threshold(conf, correct, t, min_support, tops, rule).items()
                             if k != "note"}} for t in targets]


def fit_all(schema: DecisionSchema, preds: list[dict], temperatures: dict[str, float],
            target: float | dict[str, float], min_support: int = 20,
            biases: dict[str, dict[str, float]] | None = None,
            coverage: dict[str, float] | None = None,
            rules: dict[str, dict] | None = None) -> dict:
    """Fit one tau per question. Rules are applied while fitting, exactly as the gate applies
    them, so the stored accuracy is the accuracy of what the gate will actually auto-accept."""
    out = {}
    biases = biases or {}
    coverage = coverage or {}
    rules = rules or {}
    for qn, rule in rules.items():
        q = schema.question(qn)
        for o in list(rule.get("always_escalate", [])) + list(rule.get("option_tau", {})):
            if o not in q.options:
                raise ValueError(f"rule for {qn}: {o!r} is not an option")
    for q in schema.questions:
        rows = [p for p in preds if q.name in p.get("gold", {}) and q.name in p["probs"]]
        tgt = target.get(q.name, 0.95) if isinstance(target, dict) else target
        if not rows:
            out[q.name] = {"tau": NEVER, "target": tgt, "note": "no labelled rows"}
            continue
        t = temperatures.get(q.name, 1.0)
        probs = [apply_calibration(r["probs"][q.name], t, biases.get(q.name)) for r in rows]
        conf, tops, correct = gate_view(probs, [r["gold"][q.name] for r in rows])
        rule = rules.get(q.name)
        if q.name in coverage:
            picked = {"coverage_target": coverage[q.name],
                      **pick_for_coverage(conf, correct, coverage[q.name], tops, rule)}
        else:
            picked = {"target": tgt, **pick_threshold(conf, correct, tgt, min_support, tops, rule)}
        out[q.name] = {"n": len(rows), **picked,
                       "tradeoff": tradeoff_table(conf, correct, min(min_support, len(rows)), tops=tops, rule=rule)}
    result = {"schema": schema.name, "schema_fingerprint": schema.fingerprint(),
              "fitted": date.today().isoformat(), "min_support": min_support, "thresholds": out}
    if rules:
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
