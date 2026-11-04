"""Labelled records: JSONL I/O, validation, splitting, dedupe and decontamination.

One record = one state the gate reads, plus the gold option for each question it was
labelled on. A record may leave a question unlabelled; metrics and training skip it.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from .schema import DecisionSchema

Source = Literal["real", "synthetic", "correction", "proxy"]  # proxy = hand-made stand-in for real rows


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


def _token_order(sets: list[frozenset]) -> dict:
    """Global order for prefix filtering: rarest shingle first (ties broken by the shingle)."""
    freq = Counter(t for sh in sets for t in sh)
    return {t: (n, t) for t, n in freq.items()}


def _prefix(sh: frozenset, order: dict, threshold: float) -> list:
    """The shortest prefix (in global order) two sets must share to reach `threshold` Jaccard."""
    toks = sorted(sh, key=order.__getitem__)
    keep = len(toks) - math.ceil(threshold * len(toks) - 1e-9) + 1  # 0.55 * 100 is 55.000000000000007
    return toks[:max(1, keep)]


def dedupe(records: list[Record], threshold: float = 0.9,
           label_aware: bool = True) -> tuple[list[Record], list[Record]]:
    """Drop exact and near-duplicate texts, keeping the first seen.

    label_aware (default): a near-duplicate is dropped only when its labels equal the kept
    row's. Two rows that differ in one field and carry different answers are hard pairs,
    the most useful rows in the set, so both stay. Exact text copies with conflicting labels
    cannot both be right and are dropped (the first copy too).

    Exact, and fast: a prefix-filter similarity join (rare shingles first) means a row is only
    compared with rows that share one of its rarest shingles.
    """
    # exact copies (byte-identical text) that disagree on labels cannot all be right: drop every
    # copy, whatever order they come in. Case or spacing differences are NOT copies: a lower-case
    # part number is the very thing that separates commit from normalize.
    by_text: dict[str, set] = {}
    for r in records:
        by_text.setdefault(r.text, set()).add(json.dumps(r.labels, sort_keys=True))
    conflicted_texts = {t for t, ls in by_text.items() if label_aware and len(ls) > 1}
    sets = [shingles(r.text) for r in records]
    order = _token_order(sets)
    index: dict[str, list[int]] = {}
    kept_idx: list[int] = []
    dropped: list[Record] = []
    exact: set[str] = set()
    for i, r in enumerate(records):
        if r.text in conflicted_texts or r.text in exact:
            dropped.append(r)
            continue
        sh = sets[i]
        pre = _prefix(sh, order, threshold)
        cands = {j for t in pre for j in index.get(t, ())}
        dup = False
        for j in cands:
            if label_aware and records[j].labels != r.labels:
                continue
            if _maybe_similar(sh, sets[j], threshold) and jaccard(sh, sets[j]) >= threshold:
                dup = True
                break
        if dup:
            dropped.append(r)
            continue
        exact.add(r.text)
        kept_idx.append(i)
        for t in pre:
            index.setdefault(t, []).append(i)
    return [records[i] for i in kept_idx], dropped


def decontaminate(train: list[Record], held_out: list[Record],
                  threshold: float = 0.8) -> tuple[list[Record], list[Record]]:
    """Remove training records that are near copies of any held-out record.

    Without this, eval accuracy measures memory, not generalisation.
    """
    held_sh = [shingles(r.text) for r in held_out]
    train_sh = [shingles(r.text) for r in train]
    order = _token_order(held_sh + train_sh)
    index: dict[str, list[int]] = {}
    for j, sh in enumerate(held_sh):
        for t in _prefix(sh, order, threshold):
            index.setdefault(t, []).append(j)
    kept, dropped = [], []
    for r, sh in zip(train, train_sh):
        cands = {j for t in _prefix(sh, order, threshold) for j in index.get(t, ())}
        hit = any(_maybe_similar(sh, held_sh[j], threshold) and jaccard(sh, held_sh[j]) >= threshold
                  for j in cands)
        (dropped if hit else kept).append(r)
    return kept, dropped


def _maybe_similar(a: frozenset, b: frozenset, threshold: float) -> bool:
    # Jaccard can never exceed min/max of the set sizes; skip pairs that cannot pass.
    small, large = sorted((len(a), len(b)))
    return large == 0 or small / large >= threshold
