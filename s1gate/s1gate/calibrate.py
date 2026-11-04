"""Step 6: calibration by temperature scaling (and vector scaling when one option is off).

A model's raw probabilities are usually over- or under-confident. Temperature scaling fits
ONE number T per question on held-out data:  p_T(option) ∝ p(option) ** (1 / T)
T > 1 softens an overconfident model, T < 1 sharpens an underconfident one. The ranking of
options never changes, so accuracy is untouched; only the confidence becomes honest.

Vector scaling (`--method vector`) adds one bias per option:
    p(option) ∝ exp(log p(option) / T + b_option)
Use it when a single option is off on its own (compliance rows always under-confident,
say). Unlike temperature it CAN change the top option, so re-check accuracy after it.

Fit on a dev split that the model was NOT trained on, and never on the final eval set.
This is the cheap stand-in for TypeSafe's RLCD; see the guide for the RL upgrade path.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from .metrics import EPS, probs_matrix
from .schema import DecisionSchema

MIN_ROWS = 20


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


def _vector_nll(logp: np.ndarray, gold: np.ndarray, a: float, b: np.ndarray) -> float:
    z = logp * a + b
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    return float(np.mean(lse - z[np.arange(len(gold)), gold]))


def fit_vector(P: np.ndarray, gold: np.ndarray, l2: float = 1e-2, steps: int = 3000,
               lr: float = 0.05) -> tuple[float, np.ndarray]:
    """Fit a = 1/T and one bias per option by gradient descent on NLL + l2 * |b|^2.

    The problem is convex in (a, b) (multinomial logistic regression on fixed features), so
    plain Adam from the temperature solution converges; l2 keeps biases of options that are
    rare in dev from running off.
    """
    logp = np.log(P + EPS)
    n, k = logp.shape
    a = 1.0 / fit_temperature(P, gold)
    b = np.zeros(k)
    onehot = np.eye(k)[gold]
    m = np.zeros(k + 1)
    v = np.zeros(k + 1)
    for step in range(1, steps + 1):
        z = logp * a + b
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        err = (p - onehot) / n
        grad = np.concatenate([[float(np.sum(err * logp))], err.sum(axis=0) + 2 * l2 * b])
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        upd = lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
        a = max(a - upd[0], 1e-3)
        b = b - upd[1:]
    b = b - b.mean()  # softmax is shift-invariant; centre for readability
    return float(1.0 / a), b


def apply_calibration(probs: dict[str, float], t: float = 1.0,
                      bias: dict[str, float] | None = None) -> dict[str, float]:
    if t == 1.0 and not bias:
        return dict(probs)
    logs = {o: np.log(max(p, EPS)) / t + (bias.get(o, 0.0) if bias else 0.0) for o, p in probs.items()}
    m = max(logs.values())
    exp = {o: float(np.exp(v - m)) for o, v in logs.items()}
    z = sum(exp.values())
    return {o: v / z for o, v in exp.items()}


def apply_temperature(probs: dict[str, float], t: float) -> dict[str, float]:
    return apply_calibration(probs, t)


def fit_all(schema: DecisionSchema, preds: list[dict], method: str = "temperature") -> dict:
    """preds = rows of a predictions file (from `s1gate eval --preds-out`)."""
    if method not in ("temperature", "vector"):
        raise ValueError(f"unknown calibration method {method!r} (temperature, vector)")
    temps, biases, detail = {}, {}, {}
    for q in schema.questions:
        rows = [p for p in preds if q.name in p.get("gold", {}) and q.name in p["probs"]]
        if len(rows) < MIN_ROWS:
            temps[q.name] = 1.0
            detail[q.name] = {"n": len(rows), "note": f"under {MIN_ROWS} labelled rows; T left at 1.0"}
            continue
        P = probs_matrix([r["probs"][q.name] for r in rows], q.labels)
        gold = np.array([q.labels.index(r["gold"][q.name]) for r in rows])
        logp = np.log(P + EPS)
        before = _nll(logp, gold, 1.0)
        if method == "vector":
            t, b = fit_vector(P, gold)
            biases[q.name] = {o: round(float(x), 4) for o, x in zip(q.labels, b)}
            after = _vector_nll(logp, gold, 1.0 / t, b)
            acc_after = float(np.mean(np.argmax(logp / t + b, axis=1) == gold))
        else:
            t = fit_temperature(P, gold)
            after = _nll(logp, gold, t)
            acc_after = float(np.mean(np.argmax(P, axis=1) == gold))
        temps[q.name] = round(t, 4)
        detail[q.name] = {"n": len(rows), "nll_before": round(before, 4), "nll_after": round(after, 4),
                          "accuracy_before": round(float(np.mean(np.argmax(P, axis=1) == gold)), 4),
                          "accuracy_after": round(acc_after, 4)}
    out = {"schema": schema.name, "schema_fingerprint": schema.fingerprint(), "method": method,
           "fitted": date.today().isoformat(), "temperatures": temps, "detail": detail}
    if method == "vector":
        out["biases"] = biases
    return out


def _load(path: str | Path, schema: DecisionSchema) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("schema_fingerprint") != schema.fingerprint():
        raise ValueError(f"{path} was fitted for a different schema version; re-run calibrate")
    return data


def load_calibration(path: str | Path | None, schema: DecisionSchema) -> dict[str, float]:
    if not path:
        return {}
    return {k: float(v) for k, v in _load(path, schema)["temperatures"].items()}


def load_biases(path: str | Path | None, schema: DecisionSchema) -> dict[str, dict[str, float]]:
    """Per-option biases of a vector-scaling file ({} for plain temperature scaling)."""
    if not path:
        return {}
    data = _load(path, schema)
    return {q: {o: float(b) for o, b in bs.items()} for q, bs in data.get("biases", {}).items()}
