"""A deliberately dumb baseline: word overlap between the state and each option's words.

Two uses: it exercises the whole harness with no model download, and it is the floor any
real model has to beat. If GLiNER barely beats this on your eval set, the schema's
option descriptions are probably the problem, not the model.
"""
from __future__ import annotations

import re

from ..schema import DecisionSchema
from .base import Backend, Probs, softmax

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "is", "be", "it", "this", "row", "for",
         "but", "as", "on", "in", "by", "with", "then", "not", "no", "any", "one"}


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


class KeywordBackend(Backend):
    name = "keyword"

    def __init__(self, sharpness: float = 2.0):
        self.sharpness = sharpness

    def predict(self, text: str, schema: DecisionSchema) -> Probs:
        state = _words(text)
        out: Probs = {}
        for q in schema.questions:
            scores = {}
            for option, desc in q.options.items():
                words = _words(option.replace("_", " ") + " " + desc)
                scores[option] = self.sharpness * len(state & words) / (len(words) ** 0.5 or 1)
            out[q.name] = softmax(scores)
        return out
