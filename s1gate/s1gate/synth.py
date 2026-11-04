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
from typing import Literal, Protocol

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

NoiseMode = Literal["none", "optional", "required"]


def _targets(v):
    return {str(k): _yaml_key(val) for k, val in v.items()}


class Scenario(BaseModel):
    target: dict[str, str]
    hint: str
    weight: float = 1.0
    id: str | None = None
    noise: NoiseMode = "optional"   # none = canonical formats only (e.g. commit rows)

    @field_validator("target", mode="before")
    @classmethod
    def _str_values(cls, v):
        return _targets(v)


class Pair(BaseModel):
    """Two records identical except for `change`, with targets a and b: a hard pair."""
    id: str
    change: str
    a: dict[str, str]
    b: dict[str, str]
    weight: float = 1.0

    @field_validator("a", "b", mode="before")
    @classmethod
    def _str_values(cls, v):
        return _targets(v)


class Recipe(BaseModel):
    domain: str
    record_format: str
    example: str = ""
    guide: str | None = None        # labelling guide sent to teacher and blind labeller
    parts: str | None = None        # part catalogue (YAML with a `parts` list)
    pools: str | None = None        # TRAIN-side company pool (JSON, region -> companies)
    variation: dict[str, list[str]] = Field(default_factory=dict)   # label-neutral axes
    noise: dict[str, list[str]] = Field(default_factory=dict)       # format-noise axes
    scenarios: list[Scenario]
    pairs: list[Pair] = Field(default_factory=list)
    n_per_scenario: int = 40
    pairs_per_type: int = 20
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


def _read(path: str | None, limit: int = 30000) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read()[:limit]


def style_card(recipe: Recipe, rng: random.Random, mode: str) -> dict[str, str]:
    card = {axis: rng.choice(values) for axis, values in recipe.variation.items() if values}
    if mode == "none" or not recipe.noise:
        card["noise"] = "none - every value in canonical form, except what the hint itself requires"
    else:
        k = rng.choice([1, 1, 2]) if mode == "required" else rng.choice([0, 0, 1, 2])
        axes = rng.sample(sorted(recipe.noise), k=min(k, len(recipe.noise)))
        card["noise"] = " + ".join(rng.choice(recipe.noise[a]) for a in axes) or "none"
    return card


def _pool_sample(recipe: Recipe, region: str | None, rng: random.Random) -> str:
    raw = _read(recipe.pools, limit=10_000_000)
    if not raw:
        return ""
    pools = json.loads(raw)
    key = (region or "").lower().replace(" and ", "_").replace(" ", "_")
    block = pools.get(key) or pools[rng.choice(sorted(pools))]
    dists = rng.sample(block["distributors"], k=min(3, len(block["distributors"])))
    custs = rng.sample(block["customers"], k=min(12, len(block["customers"])))
    return ("Companies to use (invented; master customer = one of these names):\n"
            + "\n".join(f"- distributor {d['name']} (refs like {d.get('ref_style', 'INV-######')})" for d in dists)
            + "\n" + "\n".join(f"- customer {c['name']} ({c.get('country', '')})" for c in custs))


def _context(recipe: Recipe, schema: DecisionSchema, card: dict[str, str], rng: random.Random) -> str:
    guide = _read(recipe.guide)
    parts = _read(recipe.parts)
    blocks = [f"Domain:\n{recipe.domain.strip()}", f"Record format:\n{recipe.record_format.strip()}",
              f"Example record:\n{recipe.example.strip()}", f"Questions the classifier answers:\n{describe_schema(schema)}"]
    if guide:
        blocks.append(f"Labelling guide (the rules every label must follow):\n{guide}")
    if parts:
        blocks.append(f"Part catalogue (use these parts with exactly these spq/moq in master data):\n{parts}")
    pool = _pool_sample(recipe, card.get("region"), rng)
    if pool:
        blocks.append(pool)
    blocks.append("Style for this batch:\n" + "\n".join(f"- {k}: {v}" for k, v in card.items()))
    return "\n\n".join(blocks)


RULES = """Rules:
- Each record must be unambiguous under the option definitions and the guide.
- Keep the record format: same keys, same order; only values vary.
- Vary names, numbers, references and wording; never repeat a record.
- Do not add anything (a format quirk, a renamed customer, a seen ref, a restricted flag, a
  missing value) that would move the answer away from the target. Check SPQ/MOQ arithmetic.
- Do not mention the answer or the question inside the record."""


def generation_prompt(schema: DecisionSchema, recipe: Recipe, scenario: Scenario,
                      k: int, rng: random.Random) -> str:
    card = style_card(recipe, rng, scenario.noise)
    target = json.dumps(scenario.target)
    return f"""{_context(recipe, schema, card, rng)}

Write {k} distinct records whose correct answers are exactly: {target}
What such a record looks like: {scenario.hint}

{RULES}

Return a JSON array of {k} objects: {{"text": "<record>", "labels": {target}}}"""


def pair_prompt(schema: DecisionSchema, recipe: Recipe, pair: Pair, k: int, rng: random.Random) -> str:
    card = style_card(recipe, rng, "optional")
    return f"""{_context(recipe, schema, card, rng)}

Write {k} distinct PAIRS of records. In each pair the two records are identical except for
this one change: {pair.change}
Record a must have exactly the answers {json.dumps(pair.a)}; record b exactly {json.dumps(pair.b)}.

{RULES}

Return a JSON array of {k} objects: {{"a": "<record a>", "b": "<record b>"}}"""


def labelling_prompt(schema: DecisionSchema, texts: list[str], questions: list[str],
                     guide: str = "") -> str:
    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    keys = ", ".join(f'"{q}": <option>' for q in questions)
    rules = f"Labelling guide:\n{guide}\n\n" if guide else ""
    return f"""{rules}Questions:
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


def _rule_filter(batch: list[Record], rules, catalogue) -> list[Record]:
    if rules is None:
        return batch
    return [r for r in batch if not any(f.level == "violation" for f in rules.check_labels(r.text, r.labels, catalogue))]


def generate(schema: DecisionSchema, recipe: Recipe, llm: "LLM | list[LLM]", seed: int = 7,
             verify: bool = True, scale: float = 1.0, log=print, rules=None,
             catalogue: set[str] | None = None) -> list[Record]:
    """Teacher loop over scenarios and hard pairs.

    llm may be a list of teachers ("several teachers": rows from two model families are more
    varied than twice as many from one). Calls rotate through them, and the blind check of a
    batch uses the NEXT teacher, so a model never verifies its own rows when it has company.
    `rules` (a domain module, e.g. s1gate.domains.pos) drops rows that break a hard rule.
    """
    teachers = llm if isinstance(llm, list) else [llm]
    rng = random.Random(seed)
    guide = _read(recipe.guide)
    calls = 0
    kept: list[Record] = []

    def pick():
        nonlocal calls
        g = teachers[calls % len(teachers)]
        v = teachers[(calls + 1) % len(teachers)]
        calls += 1
        return g, v

    for s_idx, scenario in enumerate(recipe.scenarios):
        sid = scenario.id or str(s_idx)
        want = max(1, round(recipe.n_per_scenario * scenario.weight * scale))
        got: list[Record] = []
        attempts = 0
        while len(got) < want and attempts < 3 * (want // recipe.batch_size + 1):
            attempts += 1
            k = recipe.batch_size  # always a full batch: verification drops some, extras are trimmed
            gen, ver = pick()
            raw = gen.complete(GEN_SYSTEM, generation_prompt(schema, recipe, scenario, k, rng),
                               temperature=recipe.temperature)
            batch = []
            for item in extract_json_array(raw):
                if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                    continue
                text = str(item["text"]).strip()
                batch.append(Record(id=stable_id(text, "syn"), text=text,
                                    labels=dict(scenario.target), source="synthetic",
                                    meta={"scenario": sid, "teacher": getattr(gen, "model", "?")}))
            batch = [r for r in batch if not validate_records([r], schema)]
            batch = _rule_filter(batch, rules, catalogue)
            if verify and batch:
                batch = _blind_verify(schema, ver, batch, list(scenario.target), guide)
            got.extend(batch)
        log(f"scenario {sid}: kept {len(got)}/{want} after {attempts} calls - {scenario.hint[:60]}")
        kept.extend(got[:want])

    for pair in recipe.pairs:
        want = max(1, round(recipe.pairs_per_type * pair.weight * scale))
        got_pairs = 0
        attempts = 0
        k = max(1, recipe.batch_size // 2)
        while got_pairs < want and attempts < 3 * (want // k + 1):
            attempts += 1
            gen, ver = pick()
            raw = gen.complete(GEN_SYSTEM, pair_prompt(schema, recipe, pair, k, rng),
                               temperature=recipe.temperature)
            for item in extract_json_array(raw):
                if got_pairs >= want or not isinstance(item, dict):
                    continue
                ta, tb = str(item.get("a", "")).strip(), str(item.get("b", "")).strip()
                if not ta or not tb or ta == tb:
                    continue
                pid = f"{pair.id}-{stable_id(ta + tb, '')[1:]}"
                both = [Record(id=stable_id(t, "syn"), text=t, labels=dict(lab), source="synthetic",
                               meta={"scenario": pair.id, "pair": pid, "side": side,
                                     "teacher": getattr(gen, "model", "?")})
                        for t, lab, side in ((ta, pair.a, "a"), (tb, pair.b, "b"))]
                both = _rule_filter([r for r in both if not validate_records([r], schema)], rules, catalogue)
                if verify and both:
                    both = _blind_verify(schema, ver, both, list(pair.a), guide)
                if len(both) == 2:  # keep a pair only whole: half a pair teaches no boundary
                    kept.extend(both)
                    got_pairs += 1
        log(f"pair {pair.id}: kept {got_pairs}/{want} pairs after {attempts} calls - {pair.change[:60]}")
    return kept


def _blind_verify(schema: DecisionSchema, llm: LLM, batch: list[Record],
                  questions: list[str], guide: str = "") -> list[Record]:
    raw = llm.complete(LABEL_SYSTEM, labelling_prompt(schema, [r.text for r in batch], questions, guide),
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


# ---------------------------------------------------------------- blind verification by other agents
# When the teacher is not called through `generate` (rows written by agents or another tool),
# the same blind check runs in two halves: export shuffled, text-only batches that MIX
# scenarios (so a labeller cannot infer the target from its neighbours), have independent
# annotators label them, then merge: keep a row only when every blind label equals its target.


def blind_batches(records: list[Record], batch_size: int = 150, seed: int = 0,
                  anonymize: bool = True) -> tuple[list[list[dict]], dict[str, str]]:
    """Shuffled id+text batches: no labels, no meta, scenarios mixed, opaque ids.

    Returns (batches, key) where key maps each opaque id back to the record id. Ids such as
    "t19-0087" would tell an annotator which rows came from the same scenario, so they are
    replaced; keep the key away from the annotators and pass it to `merge_blind`.
    """
    import hashlib

    rng = random.Random(seed)
    rows, key = [], {}
    for r in records:
        anon = "b" + hashlib.sha1(f"{seed}:{r.id}".encode()).hexdigest()[:10] if anonymize else r.id
        key[anon] = r.id
        rows.append({"id": anon, "text": r.text})
    rng.shuffle(rows)
    return [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)], key


def merge_blind(records: list[Record], blind: dict[str, dict[str, str]], schema: DecisionSchema,
                questions: list[str] | None = None) -> tuple[list[Record], list[Record], dict]:
    """Keep records whose blind labels equal the target on every question; report the rest.

    Returns (kept, rejected, stats). Rejected rows carry meta.blind and meta.reject so the
    confusion between target and blind label can be mined (step 3: "mine the confusion").
    """
    from collections import Counter

    questions = questions or schema.names
    kept, rejected = [], []
    confusion: dict[str, Counter] = {q: Counter() for q in questions}
    missing = 0
    for r in records:
        b = blind.get(r.id)
        if b is None:
            missing += 1
            rejected.append(r.model_copy(update={"meta": {**r.meta, "reject": "no blind label"}}))
            continue
        diff = [q for q in questions if q in r.labels and b.get(q) != r.labels[q]]
        for q in questions:
            if q in r.labels:
                confusion[q][(r.labels[q], b.get(q))] += 1
        if diff:
            rejected.append(r.model_copy(update={"meta": {**r.meta, "blind": {q: b.get(q) for q in questions},
                                                          "reject": "blind disagrees on " + ",".join(diff)}}))
        else:
            kept.append(r.model_copy(update={"meta": {**r.meta, "blind_verified": True}}))
    stats = {"rows": len(records), "kept": len(kept), "rejected": len(rejected), "no_blind_label": missing,
             "agreement": {q: round(sum(n for (t, b), n in c.items() if t == b) / max(1, sum(c.values())), 4)
                           for q, c in confusion.items()},
             "top_disagreements": {q: [{"target": t, "blind": b, "count": n}
                                       for (t, b), n in c.most_common() if t != b][:8]
                                   for q, c in confusion.items()}}
    return kept, rejected, stats


# ---------------------------------------------------------------- mine the confusion matrix
# Step 3, "ways to make the data better": every confusion in an eval report becomes a higher
# weight on the scenarios that teach the confused option, plus a hard-pair stub whose
# `change` a person (or the teacher) fills in with the one field that separates the two.


def mine_confusions(report: dict, recipe_data: dict, min_count: int = 3, boost: float = 2.0,
                    max_factor: float = 3.0) -> tuple[dict, list[dict]]:
    """Return (new recipe dict, confusions found). Does not touch the input dict."""
    import copy

    out = copy.deepcopy(recipe_data)
    found = []
    for q in report.get("questions", []):
        conf = q.get("confusion") or {}
        for gold, row in conf.items():
            total = sum(row.values())
            for pred, n in row.items():
                if pred != gold and n >= min_count and total:
                    found.append({"question": q["question"], "gold": gold, "pred": pred,
                                  "count": n, "rate": round(n / total, 3)})
    found.sort(key=lambda c: -c["count"])
    for s in out.get("scenarios", []):
        target = {str(k): _yaml_key(v) for k, v in s.get("target", {}).items()}
        rate = sum(c["rate"] for c in found if target.get(c["question"]) == c["gold"])
        if rate:
            s["weight"] = round(min(float(s.get("weight", 1.0)) * (1 + boost * rate), max_factor * float(s.get("weight", 1.0))), 3)
    existing = {(p.get("a", {}).get("row_action"), p.get("b", {}).get("row_action")) for p in out.get("pairs", [])}
    stubs = []
    for c in found:
        if c["question"] != "row_action" or (c["gold"], c["pred"]) in existing:
            continue
        existing.add((c["gold"], c["pred"]))
        stubs.append({"id": f"M{len(stubs) + 1:02d}", "a": {"row_action": c["gold"]}, "b": {"row_action": c["pred"]},
                      "change": f"TODO: the one field that turns {c['gold']} into {c['pred']} "
                                f"(seen {c['count']}x, {c['rate']:.0%} of {c['gold']} rows)"})
    if stubs:
        out["mined_pairs_todo"] = stubs  # not read by `synth` until moved into `pairs` with full targets
    return out, found
