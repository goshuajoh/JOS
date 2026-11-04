"""Step 8: the runtime gate the FSM calls in its Classify state.

    gate = Gate.from_files("configs/schema.pos.yaml", make_backend("gliner", model="runs/gliner/final"),
                           calibration="runs/calibration.json", thresholds="runs/thresholds.json",
                           log_path="runs/decisions.jsonl")
    d = gate.decide(row_text)
    if d["row_action"].escalated: -> Escalate state
    else: route on d["row_action"].option

Every decision can be appended to a JSONL log. Corrections made downstream (a person
overriding the gate) go to a corrections file via `record_correction`; both feed the
next training round.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .backends import Backend
from .calibrate import apply_temperature, load_calibration
from .records import Record, stable_id
from .schema import DecisionSchema
from .thresholds import load_thresholds


@dataclass
class Decision:
    question: str
    option: str          # what the FSM should act on (the fallback when escalated)
    model_option: str    # the model's own top option, kept for audit
    confidence: float    # calibrated probability of model_option
    tau: float
    escalated: bool
    probs: dict[str, float]


class Gate:
    def __init__(self, schema: DecisionSchema, backend: Backend,
                 temperatures: dict[str, float] | None = None,
                 thresholds: dict[str, float] | None = None,
                 log_path: str | Path | None = None):
        self.schema = schema
        self.backend = backend
        self.temperatures = temperatures or {}
        self.thresholds = thresholds or {}
        self.log_path = Path(log_path) if log_path else None

    @classmethod
    def from_files(cls, schema_path: str, backend: Backend, calibration: str | None = None,
                   thresholds: str | None = None, log_path: str | None = None) -> "Gate":
        schema = DecisionSchema.load(schema_path)
        return cls(schema, backend, load_calibration(calibration, schema),
                   load_thresholds(thresholds, schema), log_path)

    def decide(self, text: str, record_id: str | None = None) -> dict[str, Decision]:
        return self.decide_batch([text], [record_id] if record_id else None)[0]

    def decide_batch(self, texts: list[str], ids: list[str] | None = None) -> list[dict[str, Decision]]:
        start = time.perf_counter()
        raw = self.backend.predict_batch(texts, self.schema)
        ms = (time.perf_counter() - start) * 1000 / max(len(texts), 1)
        results = []
        for i, (text, probs) in enumerate(zip(texts, raw)):
            out = {}
            for q in self.schema.questions:
                p = apply_temperature(probs[q.name], self.temperatures.get(q.name, 1.0))
                top = max(p, key=p.get)
                tau = self.thresholds.get(q.name, 0.0)
                escalated = p[top] < tau
                option = (q.fallback or top) if escalated else top
                out[q.name] = Decision(q.name, option, top, round(p[top], 4), tau, escalated,
                                       {k: round(v, 4) for k, v in p.items()})
            results.append(out)
            self._log(ids[i] if ids else stable_id(text), text, out, ms)
        return results

    def _log(self, record_id: str, text: str, decisions: dict[str, Decision], ms: float) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"id": record_id, "ts": time.time(), "text": text, "latency_ms": round(ms, 2),
               "schema_fingerprint": self.schema.fingerprint(), "backend": self.backend.name,
               "decisions": {k: asdict(v) for k, v in decisions.items()}}
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_correction(path: str | Path, record_id: str, text: str, labels: dict[str, str],
                      reviewer: str | None = None) -> None:
    """Append a human-confirmed label. These are the most valuable rows you will ever get."""
    rec = Record(id=record_id, text=text, labels=labels, source="correction",
                 meta={"reviewer": reviewer, "ts": time.time()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(rec.model_dump_json() + "\n")
