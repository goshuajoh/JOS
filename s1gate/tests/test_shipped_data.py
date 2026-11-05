"""The shipped datasets keep their promises: valid, rule-clean, frozen, and free of eval leakage.

Each test skips when its file is not there (e.g. in a checkout without data/).
"""
from collections import Counter
from pathlib import Path

import pytest

from s1gate.domains import pos
from s1gate.frozen import file_sha256, load, manifest_path, text_hash
from s1gate.records import decontaminate, read_jsonl, validate_records
from s1gate.schema import DecisionSchema

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
SCHEMA = DecisionSchema.load(ROOT / "configs" / "schema.pos.yaml")
CATALOGUE = pos.load_catalogue(ROOT / "configs" / "pools" / "parts.yaml")
TRAIN, DEV, EVAL = D / "synth" / "train.jsonl", D / "dev.jsonl", D / "eval" / "eval.jsonl"


def need(*paths):
    for p in paths:
        if not p.exists():
            pytest.skip(f"{p.relative_to(ROOT)} not shipped")
    return [read_jsonl(p) for p in paths]


@pytest.mark.parametrize("path", [TRAIN, DEV, EVAL, D / "synth" / "procedural.jsonl",
                                  D / "synth" / "teacher.verified.jsonl"])
def test_valid_and_rule_clean(path):
    (rows,) = need(path)
    assert validate_records(rows, SCHEMA) == []
    bad = [(r.id, f.message) for r in rows for f in pos.check_labels(r.text, r.labels, CATALOGUE)
           if f.level == "violation"]
    assert bad == []
    assert all(tuple(r.labels[q] for q in SCHEMA.names) in pos.VALID_COMBOS for r in rows)


def test_eval_is_frozen_and_untouched():
    need(EVAL)
    man = manifest_path(EVAL)
    assert man.exists(), "run s1gate freeze on the eval set"
    assert load(man)["sha256"] == file_sha256(EVAL)
    assert load(man)["schema_fingerprint"] == SCHEMA.fingerprint()


def test_no_eval_or_dev_rows_in_training():
    train, dev, ev = need(TRAIN, DEV, EVAL)
    held = {text_hash(r.text) for r in dev + ev}
    assert not [r.id for r in train if text_hash(r.text) in held]
    kept, leaked = decontaminate(train, dev + ev, 0.8)
    assert leaked == []
    assert not {r.id for r in dev} & {r.id for r in ev}
    # repeated header/blank rows must not straddle the split: calibrate would refuse dev preds
    assert not {text_hash(r.text) for r in dev} & {text_hash(r.text) for r in ev}


def test_held_out_companies_never_appear_in_training():
    train, dev, ev = need(TRAIN, DEV, EVAL)
    import json
    pools = json.loads((D / "pools" / "eval" / "companies.json").read_text())
    names = {c["name"] for b in pools.values() for c in b["customers"] + b["distributors"]}
    hits = [r.id for r in train if any(f"={n};" in r.text for n in names)]
    assert hits == []


def test_training_covers_every_combination_and_scenario():
    (train,) = need(TRAIN)
    combos = Counter(tuple(r.labels[q] for q in SCHEMA.names) for r in train)
    assert set(combos) == pos.VALID_COMBOS
    assert min(combos.values()) >= 50
    import yaml
    recipe = yaml.safe_load((ROOT / "configs" / "synth.pos.yaml").read_text())
    ids = {s["id"] for s in recipe["scenarios"]} | {p["id"] for p in recipe["pairs"]}
    assert ids <= {r.meta.get("scenario") for r in train}


def test_dev_and_eval_cover_every_row_action():
    dev, ev = need(DEV, EVAL)
    for rows in (dev, ev):
        acts = Counter(r.labels["row_action"] for r in rows)
        assert set(acts) == set(SCHEMA.question("row_action").labels)
        assert min(acts.values()) >= 5
