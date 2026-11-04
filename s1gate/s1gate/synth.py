"""Step 3: synthetic training data from a frontier "teacher" model.

Recipe (configs/synth.*.yaml):
  scenarios  - target answers to cover, each with a hint of what such a row looks like
  variation  - axes sampled at random per call (formats, naming styles, noise) for diversity

Loop per scenario:
  1. generate: the teacher writes `batch_size` records whose correct answers are the target
  2. validate: drop records whose labels are not schema options or drift from the target
  3. verify (optional, recommended): a second, BLIND call labels the records without
     seeing the target; keep only records where blind labels agree with the target.
     This removes the rows where the generator wrote something other than it was asked.

Then run `s1gate clean` to dedupe and decontaminate against the eval set.
"""
from __future__ import annotations

import json
import os
import random
import re
from typing import Protocol

import yaml
from pydantic import BaseModel, Field, field_validator

from .records import Record, stable_id, validate_records
from .schema import DecisionSchema, _yaml_key

# ---------------------------------------------------------------- teacher clients


class LLM(Protocol):
    def complete(self, system: str, user: str, temperature: float = 1.0) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 8000):
        import anthropic

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get("S1GATE_SYNTH_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, temperature: float = 1.0) -> str:
        msg = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                          temperature=temperature, system=system,
                                          messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAICompatLLM:
    """Any OpenAI-compatible endpoint: Azure OpenAI, vLLM, Ollama, LM Studio.

    For Azure OpenAI, set base_url to https://<resource>.openai.azure.com/openai/v1/ and
    OPENAI_API_KEY to the resource key, or swap in openai.AzureOpenAI here.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        import openai

        self.client = openai.OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                                    api_key=api_key or os.environ.get("OPENAI_API_KEY", "none"))
        self.model = model

    def complete(self, system: str, user: str, temperature: float = 1.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return resp.choices[0].message.content or ""


def make_llm(provider: str, model: str | None = None, base_url: str | None = None) -> LLM:
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "openai":
        if not model:
            raise ValueError("--model is required for the openai provider")
        return OpenAICompatLLM(model, base_url=base_url)
    raise ValueError(f"unknown provider {provider!r} (anthropic, openai)")


# ---------------------------------------------------------------- recipe


class Scenario(BaseModel):
    target: dict[str, str]
    hint: str
    weight: float = 1.0

    @field_validator("target", mode="before")
    @classmethod
    def _str_values(cls, v):
        return {str(k): _yaml_key(val) for k, val in v.items()}


class Recipe(BaseModel):
    domain: str
    record_format: str
    example: str = ""
    variation: dict[str, list[str]] = Field(default_factory=dict)
    scenarios: list[Scenario]
    n_per_scenario: int = 40
    batch_size: int = 10
    temperature: float = 1.0

    @classmethod
    def load(cls, path: str) -> "Recipe":
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))


# ---------------------------------------------------------------- prompts

GEN_SYSTEM = ("You write realistic, varied synthetic records for training a classifier. "
              "Return only a JSON array, no prose, no code fences.")
LABEL_SYSTEM = ("You are a careful annotator. Label each record strictly by the option "
                "definitions given. Return only a JSON array, no prose, no code fences.")


def describe_schema(schema: DecisionSchema) -> str:
    parts = []
    for q in schema.questions:
        opts = "\n".join(f"    - {o}: {d}" if d else f"    - {o}" for o, d in q.options.items())
        parts.append(f"- {q.name} ({q.kind}): {q.prompt}\n{opts}")
    return "\n".join(parts)


def generation_prompt(schema: DecisionSchema, recipe: Recipe, scenario: Scenario,
                      k: int, rng: random.Random) -> str:
    axes = "\n".join(f"- {axis}: {rng.choice(values)}" for axis, values in recipe.variation.items())
    target = json.dumps(scenario.target)
    return f"""Domain:
{recipe.domain.strip()}

Record format:
{recipe.record_format.strip()}

Example record:
{recipe.example.strip()}

Questions the classifier answers:
{describe_schema(schema)}

Write {k} distinct records whose correct answers are exactly: {target}
What such a record looks like: {scenario.hint}

Style for this batch:
{axes or "- free"}

Rules:
- Each record must be unambiguous under the option definitions above.
- Vary names, numbers, references and wording; never repeat a record.
- Do not mention the answer or the question inside the record.

Return a JSON array of {k} objects: {{"text": "<record>", "labels": {target}}}"""


def labelling_prompt(schema: DecisionSchema, texts: list[str], questions: list[str]) -> str:
    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    keys = ", ".join(f'"{q}": <option>' for q in questions)
    return f"""Questions:
{describe_schema(schema)}

Records:
{numbered}

For each record, in order, return {{"i": <index>, {keys}}}. Answer only with option names.
Return a JSON array with {len(texts)} objects."""


def extract_json_array(text: str) -> list:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------- the loop


def generate(schema: DecisionSchema, recipe: Recipe, llm: LLM, seed: int = 7,
             verify: bool = True, scale: float = 1.0, log=print) -> list[Record]:
    rng = random.Random(seed)
    kept: list[Record] = []
    for s_idx, scenario in enumerate(recipe.scenarios):
        want = max(1, round(recipe.n_per_scenario * scenario.weight * scale))
        got: list[Record] = []
        attempts = 0
        while len(got) < want and attempts < 3 * (want // recipe.batch_size + 1):
            attempts += 1
            k = recipe.batch_size  # always a full batch: verification drops some, extras are trimmed
            raw = llm.complete(GEN_SYSTEM, generation_prompt(schema, recipe, scenario, k, rng),
                               temperature=recipe.temperature)
            batch = []
            for item in extract_json_array(raw):
                if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                    continue
                text = str(item["text"]).strip()
                batch.append(Record(id=stable_id(text, "syn"), text=text,
                                    labels=dict(scenario.target), source="synthetic",
                                    meta={"scenario": s_idx, "teacher": getattr(llm, "model", "?")}))
            batch = [r for r in batch if not validate_records([r], schema)]
            if verify and batch:
                batch = _blind_verify(schema, llm, batch, list(scenario.target))
            got.extend(batch)
        log(f"scenario {s_idx}: kept {len(got)}/{want} after {attempts} calls - {scenario.hint[:60]}")
        kept.extend(got[:want])
    return kept


def _blind_verify(schema: DecisionSchema, llm: LLM, batch: list[Record],
                  questions: list[str]) -> list[Record]:
    raw = llm.complete(LABEL_SYSTEM, labelling_prompt(schema, [r.text for r in batch], questions),
                       temperature=0.0)
    answers = {}
    for item in extract_json_array(raw):
        if isinstance(item, dict) and "i" in item:
            try:
                answers[int(item["i"])] = {q: _yaml_key(item.get(q)) for q in questions}
            except (TypeError, ValueError):
                continue
    out = []
    for i, r in enumerate(batch):
        blind = answers.get(i)
        if blind and all(blind[q] == r.labels[q] for q in questions):
            out.append(r)
    return out
