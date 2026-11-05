"""Step 8: a small HTTP service around the gate (FastAPI).

POST /decide    {"text": "...", "id": "optional"}           -> calibrated decisions, escalated, needs_person
POST /predict   {"text": "...", "questions": [...]}         -> raw probabilities (EndpointBackend wire format)
POST /feedback  {"id": "...", "text": "...", "labels": {...}} -> appends a correction
GET  /health                                                 -> schema name, fingerprint, backend

Binds to 127.0.0.1 by default. Put it behind the bank's gateway/auth before exposing it.
"""
from __future__ import annotations

from dataclasses import asdict

from pydantic import BaseModel

from .gate import Gate, record_correction
from .schema import DecisionSchema, Question


class DecideIn(BaseModel):
    text: str
    id: str | None = None


class PredictQuestion(BaseModel):
    name: str
    prompt: str
    options: dict[str, str]


class PredictIn(BaseModel):
    text: str
    questions: list[PredictQuestion]


class FeedbackIn(BaseModel):
    id: str
    text: str
    labels: dict[str, str]
    reviewer: str | None = None


def create_app(gate: Gate, corrections_path: str = "data/corrections.jsonl"):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="s1gate", version=str(gate.schema.version))

    @app.get("/health")
    def health():
        return {"schema": gate.schema.name, "version": gate.schema.version,
                "fingerprint": gate.schema.fingerprint(), "backend": gate.backend.name}

    @app.post("/decide")
    def decide(body: DecideIn):
        result = gate.decide(body.text, body.id)
        # escalated: System 1 did not decide every question alone (a frontier model may have).
        # needs_person: some question fell back to the fallback option or hit an always-escalate rule.
        return {"decisions": {k: asdict(v) for k, v in result.items()},
                "escalated": any(v.escalated for v in result.values()),
                "needs_person": any(v.decided_by in ("fallback", "rule") for v in result.values())}

    @app.post("/predict")
    def predict(body: PredictIn):
        schema = DecisionSchema(name="adhoc", questions=[
            Question(name=q.name, prompt=q.prompt, options=q.options) for q in body.questions])
        return {"probs": gate.backend.predict(body.text, schema)}

    @app.post("/feedback")
    def feedback(body: FeedbackIn):
        for q_name, option in body.labels.items():
            try:
                if option not in gate.schema.question(q_name).options:
                    raise HTTPException(422, f"{q_name}={option!r} is not an option")
            except KeyError:
                raise HTTPException(422, f"unknown question {q_name!r}") from None
        record_correction(corrections_path, body.id, body.text, body.labels, body.reviewer)
        return {"ok": True}

    return app
