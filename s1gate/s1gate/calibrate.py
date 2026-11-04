"""Step 6: calibration by temperature scaling.

A model's raw probabilities are usually over- or under-confident. Temperature scaling fits
ONE number T per question on held-out data:  p_T(option) ∝ p(option) ** (1 / T)
T > 1 softens an overconfident model, T < 1 sharpens an underconfident one. The ranking of
options never changes, so accuracy is untouched; only the confidence becomes honest.

Fit T on a dev split that the model was NOT trained on, and not on the final eval set.
This is the cheap stand-in for TypeSafe's RLCD; see the guide for the RL upgrade path.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from .metrics import EPS, probs_matrix
from .schema import DecisionSchema


def _nll(logp: np.ndarray, gold: np.ndarray, t: float) -> float:
    z = logp / t
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(gold)), gold]))


def fit_temperature(P: np.ndarray, gold: np.ndarray) -> float:
    """Grid search on log T, then a golden-section refine. Robust, no extra dependencies."""
    logp = np.log(P + EPS)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 241))
    losses = [_nll(logp, gold, t) for t in grid]
    i = int(np.argmin(losses))
    lo, hi = np.log(grid[max(i - 1, 0)]), np.log(grid[min(i + 1, len(grid) - 1)])
    g = (5 ** 0.5 - 1) / 2
    for _ in range(40):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if _nll(logp, gold, np.exp(a)) < _nll(logp, gold, np.exp(b)):
            hi = b
        else:
            lo = a
    return float(np.exp((lo + hi) / 2))


def apply_temperature(probs: dict[str, float], t: float) -> dict[str, float]:
    if t == 1.0:
        return dict(probs)
    logs = {o: np.log(max(p, EPS)) / t for o, p in probs.items()}
    m = max(logs.values())
    exp = {o: float(np.exp(v - m)) for o, v in logs.items()}
    z = sum(exp.values())
    return {o: v / z for o, v in exp.items()}


def fit_all(schema: DecisionSchema, preds: list[dict]) -> dict:
    """preds = rows of a predictions file (from `s1gate eval --preds-out`)."""
    temps, detail = {}, {}
    for q in schema.questions:
        rows = [p for p in preds if q.name in p.get("gold", {}) and q.name in p["probs"]]
        if len(rows) < 20:
            temps[q.name] = 1.0
            detail[q.name] = {"n": len(rows), "note": "under 20 labelled rows; T left at 1.0"}
            continue
        P = probs_matrix([r["probs"][q.name] for r in rows], q.labels)
        gold = np.array([q.labels.index(r["gold"][q.name]) for r in rows])
        t = fit_temperature(P, gold)
        logp = np.log(P + EPS)
        temps[q.name] = round(t, 4)
        detail[q.name] = {"n": len(rows), "nll_before": round(_nll(logp, gold, 1.0), 4),
                          "nll_after": round(_nll(logp, gold, t), 4)}
    return {"schema": schema.name, "schema_fingerprint": schema.fingerprint(),
            "fitted": date.today().isoformat(), "temperatures": temps, "detail": detail}


def load_calibration(path: str | Path | None, schema: DecisionSchema) -> dict[str, float]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    if data.get("schema_fingerprint") != schema.fingerprint():
        raise ValueError(f"{path} was fitted for a different schema version; re-run calibrate")
    return {k: float(v) for k, v in data["temperatures"].items()}
