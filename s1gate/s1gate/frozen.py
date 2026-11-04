"""Step 1: freeze the eval set, then keep it out of everything except the final report.

    s1gate freeze --schema configs/schema.pos.yaml --data data/eval/eval.jsonl
        -> data/eval/eval.frozen.json  (sha256 of the file, ids, text hashes, label counts)

After freezing:
  * `s1gate eval` on the file checks it is byte-identical to what was frozen;
  * training commands refuse any record whose text matches a frozen row;
  * `calibrate` and `thresholds` refuse predictions made on frozen rows (fitting on eval
    turns it into a dev set - pitfall 1 in the guide).

Manifests are found next to the data (`*.frozen.json`) or passed explicitly with --frozen.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

from .records import Record, normalize, read_jsonl
from .schema import DecisionSchema

DEFAULT_DIR = Path("data/eval")


class FrozenOverlapError(RuntimeError):
    pass


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode()).hexdigest()[:16]


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_path(data_path: str | Path) -> Path:
    p = Path(data_path)
    return p.with_name(p.name.removesuffix(".jsonl") + ".frozen.json")


def freeze(data_path: str | Path, schema: DecisionSchema, note: str = "") -> dict:
    records = read_jsonl(data_path)
    counts: dict[str, Counter] = {q.name: Counter() for q in schema.questions}
    for r in records:
        for q, o in r.labels.items():
            if q in counts:
                counts[q][o] += 1
    manifest = {
        "file": Path(data_path).name,
        "sha256": file_sha256(data_path),
        "n": len(records),
        "frozen": date.today().isoformat(),
        "schema": schema.name,
        "schema_version": schema.version,
        "schema_fingerprint": schema.fingerprint(),
        "sources": dict(Counter(r.source for r in records)),
        "note": note,
        "label_counts": {q: dict(c) for q, c in counts.items()},
        "ids": [r.id for r in records],
        "text_hashes": sorted({text_hash(r.text) for r in records}),
    }
    out = manifest_path(data_path)
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return manifest


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def discover(paths: list[str] | None = None, root: str | Path = DEFAULT_DIR) -> list[Path]:
    """Explicit manifests if given, else every *.frozen.json under `root` (may be empty)."""
    if paths:
        return [Path(p) for p in paths]
    root = Path(root)
    return sorted(root.glob("*.frozen.json")) if root.exists() else []


def verify_file(data_path: str | Path, manifests: list[Path] | None = None) -> tuple[dict | None, str | None]:
    """(manifest, problem) for a data file; (None, None) when the file was never frozen."""
    mp = manifest_path(data_path)
    candidates = [mp] if mp.exists() else []
    candidates += [m for m in (manifests or []) if load(m).get("file") == Path(data_path).name and m != mp]
    for m in candidates:
        man = load(m)
        if man.get("file") != Path(data_path).name:
            continue
        if file_sha256(data_path) != man["sha256"]:
            return man, (f"{data_path} changed since it was frozen on {man['frozen']} "
                         f"(sha256 mismatch) - an edited eval set is a dev set")
        return man, None
    return None, None


def overlapping(records: list[Record], manifests: list[Path]) -> list[tuple[str, str]]:
    """(record id, manifest file) for every record whose text matches a frozen row."""
    hits = []
    for m in manifests:
        man = load(m)
        frozen = set(man.get("text_hashes", []))
        for r in records:
            if text_hash(r.text) in frozen:
                hits.append((r.id, str(m)))
    return hits


def guard_training(records: list[Record], manifests: list[Path], what: str = "training") -> None:
    hits = overlapping(records, manifests)
    if hits:
        sample = ", ".join(i for i, _ in hits[:5])
        raise FrozenOverlapError(f"{len(hits)} {what} records copy frozen eval rows ({sample}...); "
                                 "remove them (s1gate clean --held-out <eval>) - never train on eval")


def guard_preds(preds: list[dict], manifests: list[Path], what: str = "fitting") -> None:
    """Refuse predictions on frozen rows, matched by id or (when the row carries one) text hash."""
    frozen_ids: set[str] = set()
    frozen_text: set[str] = set()
    for m in manifests:
        man = load(m)
        frozen_ids |= set(man.get("ids", []))
        frozen_text |= set(man.get("text_hashes", []))
    hits = [p.get("id", "?") for p in preds if p.get("id") in frozen_ids or p.get("text_hash") in frozen_text]
    if hits:
        raise FrozenOverlapError(f"{len(hits)} predictions are on frozen eval rows ({', '.join(hits[:5])}...); "
                                 f"{what} must use dev predictions, never eval")
