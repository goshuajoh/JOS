"""Step 1: the decision schema.

A schema is a list of questions. Each question has a fixed set of options; the gate must
return exactly one of them. The schema is the contract between the model and the FSM:
Validate/Resolve only ever see these enum values.
"""
from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, create_model, field_validator, model_validator

Kind = Literal["choice", "noul", "score"]


def _enum_member(option: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", option).strip("_").upper() or "EMPTY"
    return f"V_{name}" if name[0].isdigit() else name


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


class Question(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: Kind = "choice"
    prompt: str
    options: dict[str, str]
    fallback: str | None = None

    @field_validator("options", mode="before")
    @classmethod
    def _coerce_options(cls, value):
        # Accept a plain list of names, or {name: description}. YAML may hand back
        # non-string keys (0, True), so everything is forced to str.
        if isinstance(value, list):
            return {str(o): "" for o in value}
        if isinstance(value, dict):
            return {_yaml_key(k): ("" if d is None else str(d)) for k, d in value.items()}
        raise TypeError("options must be a list or a mapping")

    @model_validator(mode="after")
    def _check(self):
        if len(self.options) < 2:
            raise ValueError(f"{self.name}: needs at least two options")
        if self.kind == "noul" and set(self.options) != {"yes", "no"}:
            raise ValueError(f"{self.name}: a noul question takes exactly 'yes' and 'no'")
        if self.kind == "score":
            try:
                values = [int(o) for o in self.options]
            except ValueError as exc:
                raise ValueError(f"{self.name}: score options must be integers") from exc
            if values != sorted(values):
                raise ValueError(f"{self.name}: score options must be in ascending order")
        if self.fallback is not None and self.fallback not in self.options:
            raise ValueError(f"{self.name}: fallback {self.fallback!r} is not an option")
        return self

    @property
    def labels(self) -> list[str]:
        return list(self.options)

    def as_enum(self) -> type[Enum]:
        return Enum(_camel(self.name), [(_enum_member(o), o) for o in self.options], type=str)


def _yaml_key(key) -> str:
    if key is True:
        return "yes"
    if key is False:
        return "no"
    return str(key)


class DecisionSchema(BaseModel):
    name: str
    version: int = 1
    description: str = ""
    questions: list[Question]

    @model_validator(mode="after")
    def _unique(self):
        names = [q.name for q in self.questions]
        if len(names) != len(set(names)):
            raise ValueError("question names must be unique")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "DecisionSchema":
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def question(self, name: str) -> Question:
        for q in self.questions:
            if q.name == name:
                return q
        raise KeyError(name)

    @property
    def names(self) -> list[str]:
        return [q.name for q in self.questions]

    def fingerprint(self) -> str:
        """Changes whenever a question, option or description changes.

        Calibration and threshold files store it, so a stale file is caught at load time.
        """
        blob = json.dumps(self.model_dump(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def output_model(self) -> type[BaseModel]:
        """A Pydantic model of one gate answer, with an Enum per question.

        Its JSON schema can drive constrained decoding (XGrammar / Outlines) if a
        generative fallback is ever used for the same contract.
        """
        fields = {q.name: (q.as_enum(), ...) for q in self.questions}
        return create_model(_camel(self.name) + "Decision", **fields)

    def json_schema(self) -> dict:
        return self.output_model().model_json_schema()
