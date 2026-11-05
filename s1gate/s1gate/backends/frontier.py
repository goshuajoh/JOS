"""A frontier LLM as a backend: the escalation target for rows System 1 is unsure about.

Slow and remote, so it is never the gate itself. Plug it into `Gate(escalator=...)` (or
`s1gate serve --escalate-to frontier`) and it is asked only about the questions that fell
under their threshold. The answer is still typed: anything that is not a schema option is
treated as "no answer" and the row keeps its fallback (ask_human).

Probabilities are built from the model's stated confidence (top option = confidence, the
rest share the remainder). Verbalised confidence is not calibrated; the gate only uses it
as a floor (`escalator_min_confidence`). Calibrate it on dev like any other backend
(`s1gate eval --backend frontier ...` then `s1gate calibrate`) before trusting the number.

Bank data rule: rows sent here leave the process. Use only a deployment approved for the
data class (an internal Azure OpenAI deployment, say), with masking where it applies.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..schema import DecisionSchema
from .base import Backend, Probs

SYSTEM = ("You are a careful reviewer for a POS-ingestion pipeline. Answer every question "
          "strictly by the option definitions. Return only JSON, no prose, no code fences.")


def _describe(schema: DecisionSchema) -> str:
    parts = []
    for q in schema.questions:
        opts = "\n".join(f"    - {o}: {d}" if d else f"    - {o}" for o, d in q.options.items())
        parts.append(f"- {q.name}: {q.prompt}\n{opts}")
    return "\n".join(parts)


class FrontierBackend(Backend):
    name = "frontier"

    def __init__(self, llm, guide_path: str | None = None, max_guide_chars: int = 20000):
        self.llm = llm
        self.guide = ""
        if guide_path and Path(guide_path).exists():
            self.guide = Path(guide_path).read_text(encoding="utf-8")[:max_guide_chars]

    def prompt(self, text: str, schema: DecisionSchema) -> str:
        keys = ", ".join(f'"{q.name}": {{"option": "<option>", "confidence": <0..1>}}' for q in schema.questions)
        guide = f"Labelling guide:\n{self.guide}\n\n" if self.guide else ""
        return (f"{guide}Questions:\n{_describe(schema)}\n\nRecord:\n{text}\n\n"
                f"Return one JSON object: {{{keys}}}")

    def predict(self, text: str, schema: DecisionSchema) -> Probs:
        raw = self.llm.complete(SYSTEM, self.prompt(text, schema), temperature=0.0)
        return parse_answer(raw, schema)


def _confidence(value) -> float:
    """A stated confidence in [0, 1]; 0.5 when there is none usable.

    No stated number counts as a coin flip, so it never clears an escalation floor by default:
    a missing value, true/false, text that is not a number, NaN, infinity and any number outside
    [0, 100] give 0.5. A number above 1 (up to 100) is read as a percentage.
    """
    if value is None or isinstance(value, bool):
        return 0.5
    try:
        conf = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.5
    if not math.isfinite(conf):
        return 0.5
    if 1 < conf <= 100:
        conf /= 100
    return conf if 0 <= conf <= 1 else 0.5  # 150, -3, 1e300: not a probability


def parse_answer(raw: str, schema: DecisionSchema) -> Probs:
    """JSON answer -> probabilities; anything unusable becomes a uniform (= unsure) row."""
    try:
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1]) if start != -1 and end > start else {}
    except json.JSONDecodeError:
        data = {}
    out: Probs = {}
    for q in schema.questions:
        k = len(q.labels)
        item = data.get(q.name) if isinstance(data, dict) else None
        option = item.get("option") if isinstance(item, dict) else item
        option = "yes" if option is True else "no" if option is False else (str(option) if option is not None else None)
        if option not in q.options:
            out[q.name] = {o: 1 / k for o in q.labels}
            continue
        conf = _confidence(item.get("confidence") if isinstance(item, dict) else None)
        # the stated option stays strictly on top, however low the stated confidence
        conf = min(max(conf, 1 / k + 0.01), 0.999)
        rest = (1 - conf) / max(k - 1, 1)
        out[q.name] = {o: (conf if o == option else rest) for o in q.labels}
    return out
