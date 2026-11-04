"""Call a decision model over HTTP.

Default wire format = this project's own `s1gate serve` (/predict), so a GPU box can host
the Qwen backend while laptops and the FSM call it.

For a TypeSafe-compatible server (KEV, JevK5, Decima all ship one), subclass and override
`build_request` and `parse_response` using that project's README for the exact field names;
the rest of the harness does not change.
"""
from __future__ import annotations

import json
import os
import urllib.request

from ..schema import DecisionSchema
from .base import Backend, Probs


class EndpointBackend(Backend):
    name = "endpoint"

    def __init__(self, url: str, api_key: str | None = None, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.api_key = api_key or os.environ.get("S1GATE_ENDPOINT_KEY")
        self.timeout = timeout

    def build_request(self, text: str, schema: DecisionSchema) -> tuple[str, dict]:
        payload = {
            "text": text,
            "questions": [{"name": q.name, "prompt": q.prompt, "options": q.options}
                          for q in schema.questions],
        }
        return self.url + "/predict", payload

    def parse_response(self, body: dict, schema: DecisionSchema) -> Probs:
        return {q.name: {o: float(p) for o, p in body["probs"][q.name].items()}
                for q in schema.questions}

    def predict(self, text: str, schema: DecisionSchema) -> Probs:
        url, payload = self.build_request(text, schema)
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return self.parse_response(json.loads(resp.read()), schema)
