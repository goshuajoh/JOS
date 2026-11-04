"""Step 8: the runtime gate the FSM calls in its Classify state.

    gate = Gate.from_files("configs/schema.pos.yaml", make_backend("gliner", model="runs/gliner/best"),
                           calibration="runs/calibration.json", thresholds="runs/thresholds.json",
                           log_path="runs/decisions.jsonl")
    d = gate.decide(row_text)
    if d["row_action"].escalated: -> Escalate state
    else: route on d["row_action"].option

How one question is decided (`Decision.decided_by`):
  gate      calibrated confidence >= tau (the option's own tau if a rule sets one): auto-accept
  rule      the top option is listed in `always_escalate`: a person sees it whatever the confidence
  frontier  confidence < tau and an escalator (a frontier model) answered with enough confidence
  fallback  confidence < tau and no confident escalator: option = the question's fallback

`escalated` is True for everything except `gate`: System 1 did not decide it alone.
The escalator is off by default; rows sent to it leave the process (and, for a hosted
model, the bank network), so only wire in an approved deployment.

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
from .calibrate import apply_calibration, load_biases, load_calibration
from .records import Record, stable_id
from .schema import DecisionSchema
from .thresholds import load_rules, load_thresholds


@dataclass
class Decision:
    question: str
    option: str          # what the FSM should act on (the fallback when escalated)
    model_option: str    # the model's own top option, kept for audit
    confidence: float    # calibrated probability of model_option
    tau: float
    escalated: bool
    probs: dict[str, float]
    decided_by: str = "gate"   # gate | rule | frontier | fallback


class Gate:
    def __init__(self, schema: DecisionSchema, backend: Backend,
                 temperatures: dict[str, float] | None = None,
                 thresholds: dict[str, float] | None = None,
                 log_path: str | Path | None = None,
                 biases: dict[str, dict[str, float]] | None = None,
                 rules: dict[str, dict] | None = None,
                 escalator: Backend | None = None,
                 escalator_min_confidence: float = 0.8):
        self.schema = schema
        self.backend = backend
        self.temperatures = temperatures or {}
        self.thresholds = thresholds or {}
        self.biases = biases or {}
        self.rules = rules or {}
        self.escalator = escalator
        self.escalator_min_confidence = escalator_min_confidence
        self.log_path = Path(log_path) if log_path else None
        for qn, rule in self.rules.items():
            q = schema.question(qn)
            for o in list(rule.get("always_escalate", [])) + list(rule.get("option_tau", {})):
                if o not in q.options:
                    raise ValueError(f"rule for {qn}: {o!r} is not an option")

    @classmethod
    def from_files(cls, schema_path: str, backend: Backend, calibration: str | None = None,
                   thresholds: str | None = None, log_path: str | None = None,
                   escalator: Backend | None = None, escalator_min_confidence: float = 0.8) -> "Gate":
        schema = DecisionSchema.load(schema_path)
        return cls(schema, backend, load_calibration(calibration, schema),
                   load_thresholds(thresholds, schema), log_path,
                   biases=load_biases(calibration, schema), rules=load_rules(thresholds, schema),
                   escalator=escalator, escalator_min_confidence=escalator_min_confidence)

    def decide(self, text: str, record_id: str | None = None) -> dict[str, Decision]:
        return self.decide_batch([text], [record_id] if record_id else None)[0]

    def _decide_one(self, q, probs: dict[str, float]) -> Decision:
        p = apply_calibration(probs, self.temperatures.get(q.name, 1.0), self.biases.get(q.name))
        top = max(p, key=p.get)
        rule = self.rules.get(q.name, {})
        tau = float(rule.get("option_tau", {}).get(top, self.thresholds.get(q.name, 0.0)))
        if top in rule.get("always_escalate", ()):
            how = "rule"
        elif p[top] < tau:
            how = "fallback"
        else:
            how = "gate"
        escalated = how != "gate"
        option = (q.fallback or top) if escalated else top
        return Decision(q.name, option, top, round(p[top], 4), tau, escalated,
                        {k: round(v, 4) for k, v in p.items()}, how)

    def _ask_escalator(self, text: str, out: dict[str, Decision]) -> None:
        """Let the frontier model answer the questions System 1 escalated on confidence."""
        pending = [d for d in out.values() if d.decided_by == "fallback"]
        if not pending or self.escalator is None:
            return
        try:
            answer = self.escalator.predict(text, self.schema)
        except Exception:  # an unreachable escalator must never block the FSM: keep the fallback
            return
        for d in pending:
            probs = answer.get(d.question) or {}
            if not probs:
                continue
            top = max(probs, key=probs.get)
            if top in self.schema.question(d.question).options and probs[top] >= self.escalator_min_confidence:
                d.option, d.decided_by = top, "frontier"

    def decide_batch(self, texts: list[str], ids: list[str] | None = None) -> list[dict[str, Decision]]:
        start = time.perf_counter()
        raw = self.backend.predict_batch(texts, self.schema)
        ms = (time.perf_counter() - start) * 1000 / max(len(texts), 1)
        results = []
        for i, (text, probs) in enumerate(zip(texts, raw)):
            out = {q.name: self._decide_one(q, probs[q.name]) for q in self.schema.questions}
            self._ask_escalator(text, out)
            results.append(out)
            self._log(ids[i] if ids else stable_id(text), text, out, ms)
        return results

    def decision_model(self, decisions: dict[str, Decision]):
        """The decisions as the schema's typed answer model (one Enum per question)."""
        return self.schema.output_model()(**{k: v.option for k, v in decisions.items()})

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
                      reviewer: str | None = None, meta: dict | None = None) -> None:
    """Append a human-confirmed label. These are the most valuable rows you will ever get."""
    rec = Record(id=record_id, text=text, labels=labels, source="correction",
                 meta={"reviewer": reviewer, "ts": time.time(), **(meta or {})})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(rec.model_dump_json() + "\n")
