"""Metrics for a decision gate: is it right, and does its confidence mean anything?

accuracy / macro-F1  - how often the top option is right (macro-F1 weights rare options fairly)
NLL                  - average surprise at the gold option; punishes confident mistakes hard
Brier                - squared error of the whole probability vector
ECE                  - expected calibration error: |confidence - accuracy| averaged over bins
coverage curve       - accuracy you get if you only auto-accept answers above a confidence
MAE (score kind)     - distance from the gold rung on an ordinal scale
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

EPS = 1e-12


def probs_matrix(prob_dicts: list[dict[str, float]], labels: list[str]) -> np.ndarray:
    m = np.array([[d.get(l, 0.0) for l in labels] for d in prob_dicts], dtype=float)
    m = np.clip(m, 0.0, None)
    sums = m.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return m / sums


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(total)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        out.append({
            "lo": round(float(edges[b]), 3),
            "hi": round(float(edges[b + 1]), 3),
            "count": int(mask.sum()),
            "avg_confidence": float(conf[mask].mean()) if mask.any() else None,
            "accuracy": float(correct[mask].mean()) if mask.any() else None,
        })
    return out


def coverage_curve(conf: np.ndarray, correct: np.ndarray) -> list[dict]:
    """For each cut-off (highest confidence first): share auto-accepted and its accuracy."""
    order = np.argsort(-conf, kind="stable")
    c, ok = conf[order], correct[order]
    cum_ok = np.cumsum(ok)
    n = len(c)
    return [{"tau": float(c[k - 1]), "coverage": k / n, "accuracy": float(cum_ok[k - 1] / k)}
            for k in range(1, n + 1)]


def macro_f1(gold: np.ndarray, pred: np.ndarray, k: int) -> float:
    f1s = []
    for c in range(k):
        tp = np.sum((pred == c) & (gold == c))
        fp = np.sum((pred == c) & (gold != c))
        fn = np.sum((pred != c) & (gold == c))
        if tp + fp + fn == 0:
            continue  # option absent from gold and never predicted: no information
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


@dataclass
class QuestionReport:
    question: str
    n: int
    accuracy: float
    macro_f1: float
    nll: float
    brier: float
    ece: float
    mean_confidence: float
    mae: float | None = None
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    reliability: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_question(name: str, prob_dicts: list[dict[str, float]], golds: list[str],
                      labels: list[str], ordinal: bool = False) -> QuestionReport:
    if not golds:
        nan = float("nan")
        return QuestionReport(name, 0, nan, nan, nan, nan, nan, nan)
    P = probs_matrix(prob_dicts, labels)
    g = np.array([labels.index(x) for x in golds])
    pred = P.argmax(axis=1)
    conf = P.max(axis=1)
    correct = (pred == g).astype(float)
    onehot = np.eye(len(labels))[g]
    confusion = {labels[i]: {labels[j]: int(np.sum((g == i) & (pred == j))) for j in range(len(labels))}
                 for i in range(len(labels))}
    mae = None
    if ordinal:
        values = np.array([int(l) for l in labels], dtype=float)
        mae = float(np.mean(np.abs(values[pred] - values[g])))
    return QuestionReport(
        question=name,
        n=len(g),
        accuracy=float(correct.mean()),
        macro_f1=macro_f1(g, pred, len(labels)),
        nll=float(-np.mean(np.log(P[np.arange(len(g)), g] + EPS))),
        brier=float(np.mean(np.sum((P - onehot) ** 2, axis=1))),
        ece=ece(conf, correct),
        mean_confidence=float(conf.mean()),
        mae=mae,
        confusion=confusion,
        reliability=reliability_bins(conf, correct),
    )


def top_and_correct(prob_dicts: list[dict[str, float]], golds: list[str],
                    labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    P = probs_matrix(prob_dicts, labels)
    g = np.array([labels.index(x) for x in golds])
    return P.max(axis=1), (P.argmax(axis=1) == g).astype(float)
