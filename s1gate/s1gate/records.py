"""Labelled records: JSONL I/O, validation, splitting, dedupe and decontamination.

One record = one state the gate reads, plus the gold option for each question it was
labelled on. A record may leave a question unlabelled; metrics and training skip it.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from .schema import DecisionSchema

Source = Literal["real", "synthetic", "correction"]


class Record(BaseModel):
    id: str
    text: str
    labels: dict[str, str] = Field(default_factory=dict)
    source: Source = "real"
    meta: dict[str, Any] = Field(default_factory=dict)


def read_jsonl(path: str | Path) -> list[Record]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(Record.model_validate_json(line))
            except Exception as exc:  # surface the line number, then stop
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[Record]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")
            n += 1
    return n


def stable_id(text: str, prefix: str = "r") -> str:
    return f"{prefix}-{hashlib.sha1(text.encode()).hexdigest()[:10]}"


def validate_records(records: list[Record], schema: DecisionSchema) -> list[str]:
    """Every problem found, as one line each. Empty list = clean."""
    errors: list[str] = []
    seen: set[str] = set()
    for r in records:
        if r.id in seen:
            errors.append(f"{r.id}: duplicate id")
        seen.add(r.id)
        if not r.text.strip():
            errors.append(f"{r.id}: empty text")
        for q_name, option in r.labels.items():
            try:
                q = schema.question(q_name)
            except KeyError:
                errors.append(f"{r.id}: unknown question {q_name!r}")
                continue
            if option not in q.options:
                errors.append(f"{r.id}: {q_name}={option!r} is not an option ({', '.join(q.labels)})")
    return errors


def label_counts(records: list[Record], schema: DecisionSchema) -> dict[str, Counter]:
    counts = {q.name: Counter() for q in schema.questions}
    for r in records:
        for q_name, option in r.labels.items():
            if q_name in counts:
                counts[q_name][option] += 1
    return counts


def split(records: list[Record], frac: float, seed: int = 13,
          stratify_on: str | None = None) -> tuple[list[Record], list[Record]]:
    """Split into (rest, held_out). Stratified on one question's label when given."""
    rng = random.Random(seed)
    groups: dict[str, list[Record]] = {}
    for r in records:
        key = r.labels.get(stratify_on, "") if stratify_on else ""
        groups.setdefault(key, []).append(r)
    rest, held = [], []
    for group in groups.values():
        rng.shuffle(group)
        k = round(len(group) * frac)
        held.extend(group[:k])
        rest.extend(group[k:])
    return rest, held


# ---------------------------------------------------------------- near-duplicate checks

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def shingles(text: str, n: int = 5) -> frozenset[str]:
    t = normalize(text)
    if len(t) <= n:
        return frozenset([t])
    return frozenset(t[i:i + n] for i in range(len(t) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def dedupe(records: list[Record], threshold: float = 0.9) -> tuple[list[Record], list[Record]]:
    """Drop exact and near-duplicate texts, keeping the first seen.

    Pairwise, so fine to roughly 10k rows. Beyond that, swap in MinHash LSH
    (the `datasketch` package) with the same threshold.
    """
    kept: list[Record] = []
    kept_sh: list[frozenset] = []
    dropped: list[Record] = []
    exact: set[str] = set()
    for r in records:
        key = normalize(r.text)
        if key in exact:
            dropped.append(r)
            continue
        sh = shingles(r.text)
        if any(_maybe_similar(sh, other, threshold) and jaccard(sh, other) >= threshold
               for other in kept_sh):
            dropped.append(r)
            continue
        exact.add(key)
        kept.append(r)
        kept_sh.append(sh)
    return kept, dropped


def decontaminate(train: list[Record], held_out: list[Record],
                  threshold: float = 0.8) -> tuple[list[Record], list[Record]]:
    """Remove training records that are near copies of any held-out record.

    Without this, eval accuracy measures memory, not generalisation.
    """
    held_sh = [shingles(r.text) for r in held_out]
    kept, dropped = [], []
    for r in train:
        sh = shingles(r.text)
        hit = any(_maybe_similar(sh, h, threshold) and jaccard(sh, h) >= threshold for h in held_sh)
        (dropped if hit else kept).append(r)
    return kept, dropped


def _maybe_similar(a: frozenset, b: frozenset, threshold: float) -> bool:
    # Jaccard can never exceed min/max of the set sizes; skip pairs that cannot pass.
    small, large = sorted((len(a), len(b)))
    return large == 0 or small / large >= threshold
