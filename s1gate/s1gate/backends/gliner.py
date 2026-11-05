"""GLiNER2.5-Decide (or any GLiNER2 checkpoint, including your fine-tune) as a backend.

Getting a full distribution out of gliner2:
`classify_text(..., include_confidence=True)` on a single-label task returns only the top
label and its confidence. To score EVERY option, each task is sent as multi-label with
cls_threshold 0.0 (every option is returned) and class_act "softmax", so the scores are
gliner2's own softmax over the per-option logits: the single-label distribution, computed by
gliner2, the same objective `train_gliner` trains with.

A gliner2 without the class_act option returns float32 sigmoids instead. Those are turned
back into logits (log p / (1 - p)) and soft-maxed, which matches the softmax until the
sigmoids saturate at 1.0 (logits above about 16.6); then the confident options tie.

If a gliner2 release changes the payload, `_parse` falls back to top-label-only and
spreads the remainder evenly; `s1gate eval` will then show a flat calibration curve,
which is the cue to check this module.
"""
from __future__ import annotations

import math

from ..schema import DecisionSchema
from .base import Backend, Probs, softmax

DEFAULT_MODEL = "fastino/GLiNER2.5-Decide"
# gliner2 returns float32 sigmoids: keep their whole range (logits about -103..16.6) so confident
# outputs do not collapse into ties; only exact 0 and 1 are nudged inside the open interval.
_P_MIN, _P_MAX = 1e-45, 1 - 2 ** -24


def _has_class_act() -> bool:
    try:
        from gliner2.inference.schema_model import ClassificationInput
    except ImportError:
        return False
    return "class_act" in getattr(ClassificationInput, "model_fields", {})


class GlinerBackend(Backend):
    name = "gliner"

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None,
                 use_prompts: bool = True, batch_size: int = 8):
        from gliner2 import AutoExtractor  # optional dependency: pip install "gliner2[local]"

        self.model = AutoExtractor.from_pretrained(model_id)
        if device and hasattr(self.model, "to"):
            self.model = self.model.to(device)
        self.model_id = model_id
        self.use_prompts = use_prompts
        self.batch_size = batch_size
        self._task_cache: dict[str, dict] = {}
        self.activation = "softmax" if _has_class_act() else "sigmoid"

    def tasks(self, schema: DecisionSchema) -> dict:
        key = schema.fingerprint()
        if key not in self._task_cache:
            tasks = {}
            for q in schema.questions:
                described = any(q.options.values())
                spec: dict = {
                    "labels": dict(q.options) if described else q.labels,
                    "multi_label": True,
                    "cls_threshold": 0.0,
                }
                if self.activation == "softmax":
                    spec["class_act"] = "softmax"
                if self.use_prompts:
                    spec["prompt"] = q.prompt
                tasks[q.name] = spec
            self._task_cache[key] = tasks
        return self._task_cache[key]

    def predict(self, text: str, schema: DecisionSchema) -> Probs:
        raw = self.model.classify_text(text, self.tasks(schema), include_confidence=True)
        return self._to_probs(raw, schema)

    def predict_batch(self, texts: list[str], schema: DecisionSchema) -> list[Probs]:
        raws = self.model.batch_classify_text(texts, self.tasks(schema),
                                              batch_size=self.batch_size, include_confidence=True)
        return [self._to_probs(r, schema) for r in raws]

    def _to_probs(self, raw: dict, schema: DecisionSchema) -> Probs:
        return {q.name: _parse(raw.get(q.name), q.labels, self.activation) for q in schema.questions}


def _parse(value, labels: list[str], activation: str = "sigmoid") -> dict[str, float]:
    # Multi-label payload: [{"label": ..., "confidence": ...}, ...]
    if isinstance(value, list) and value and isinstance(value[0], dict):
        scores = {item["label"]: float(item["confidence"]) for item in value if "label" in item}
        if activation == "softmax" and all(l in scores for l in labels):
            p = {l: max(scores[l], 0.0) for l in labels}  # already a softmax over the options
            z = sum(p.values())
            if z > 0:
                return {l: v / z for l, v in p.items()}
        sig = scores
        logits = {}
        for l in labels:
            p = min(max(sig.get(l, _P_MIN), _P_MIN), _P_MAX)
            logits[l] = math.log(p / (1 - p))
        return softmax(logits)
    # Single-label payload: {"label": ..., "confidence": ...}
    if isinstance(value, dict) and "label" in value:
        top, conf = value["label"], float(value.get("confidence", 1.0))
        rest = (1 - conf) / max(len(labels) - 1, 1)
        return {l: (conf if l == top else rest) for l in labels}
    # Bare label string, or nothing: one-hot / uniform.
    if isinstance(value, str) and value in labels:
        return {l: (1.0 if l == value else 0.0) for l in labels}
    return {l: 1 / len(labels) for l in labels}
