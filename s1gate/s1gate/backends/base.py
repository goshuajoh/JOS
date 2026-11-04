"""The one interface every model sits behind.

A backend reads one state and returns, for each question, a probability for every option.
Everything downstream (metrics, calibration, thresholds, the gate) only sees this shape,
so swapping GLiNER for Qwen or a remote endpoint is one CLI flag.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod

from ..schema import DecisionSchema

Probs = dict[str, dict[str, float]]  # question -> option -> probability


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def predict(self, text: str, schema: DecisionSchema) -> Probs:
        ...

    def predict_batch(self, texts: list[str], schema: DecisionSchema) -> list[Probs]:
        return [self.predict(t, schema) for t in texts]


def softmax(scores: dict[str, float]) -> dict[str, float]:
    m = max(scores.values())
    exp = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exp.values())
    return {k: v / z for k, v in exp.items()}
