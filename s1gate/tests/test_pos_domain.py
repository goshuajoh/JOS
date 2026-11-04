"""POS domain pack: value readers, the rule checker and the procedural generator."""
from collections import Counter
from pathlib import Path

import pytest

from s1gate.domains import pos, pos_gen
from s1gate.records import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = pos.load_catalogue(ROOT / "configs" / "pools" / "parts.yaml")
EVAL = ROOT / "data" / "eval" / "eval.sample.jsonl"
POOLS = ROOT / "tests" / "fixtures" / "pools_small.json"

ROW = ("distributor=Northwind Components; period=2026-09; customer=Acme Controls Ltd; country=PL; "
       "part=WM-S3-N8R8; qty={qty}; unit_price=2.31 USD; ref={ref} || master: customer=Acme Controls Ltd; "
       "spq=250; moq=500; restricted={restricted}; seen_refs={seen}")


def row(qty="1000", ref="INV-1/1", restricted="no", seen="none", **swap):
    text = ROW.format(qty=qty, ref=ref, restricted=restricted, seen=seen)
    for k, v in swap.items():
        text = text.replace(f"; {k}=" + pos.parse(text).reported[k] + ";", f"; {k}={v};", 1)
    return text


def labels(action, nh, sev):
    return {"row_action": action, "needs_human": nh, "severity": sev}


@pytest.mark.parametrize("raw,value,status", [
    ("1000", 1000, "canonical"), ("1.000", 1000, "formatted"), ("1,000", 1000, "formatted"),
    ("1 000", 1000, "formatted"), ("1000.00", 1000, "formatted"), ("1.000,00", 1000, "formatted"),
    ("1,00,000", 100000, "formatted"), ("2.5k", 2500, "formatted"), ("4,000 EA", 4000, "formatted"),
    ("4000pcs", 4000, "formatted"), ("  2,000 ", 2000, "formatted"), ("1.5", None, "impossible"),
    ("0", 0, "impossible"), ("-500", None, "impossible"), ("see attachment", None, "invalid"),
    ("1000/1200", None, "invalid"), ("", None, "missing"),
])
def test_read_qty(raw, value, status):
    assert pos.read_qty(raw) == (value, status)


@pytest.mark.parametrize("raw,status", [
    ("2026-09", "canonical"), ("2026-13", "impossible"), ("30.09.2026", "formatted"),
    ("Sep-26", "formatted"), ("2026/09/15", "formatted"), ("2026-9", "formatted"),
    ("September 2026", "formatted"), ("Eylül 2026", "formatted"), ("Q3 2026", "not_month"),
    ("FY26", "not_month"), ("2026", "not_month"), ("03/04/2026", "unknown"), ("", "missing"),
])
def test_read_period(raw, status):
    assert pos.read_period(raw)[1] == status


@pytest.mark.parametrize("raw,status", [
    ("2.31 USD", "canonical"), ("2,31 USD", "formatted"), ("2.31 usd", "formatted"),
    ("$2.31", "formatted"), ("USD 2.31", "formatted"), ("0.00 USD", "impossible"),
    ("TBD", "invalid"), ("2.31", "unknown"),
])
def test_read_price(raw, status):
    assert pos.read_price(raw)[1] == status


def test_name_relation_ignores_case_spacing_punctuation_accents_only():
    assert pos.name_relation("Acme Controls Ltd", "Acme Controls Ltd") == "identical"
    assert pos.name_relation("ACME CONTROLS LTD.", "Acme Controls Ltd") == "format_only"
    assert pos.name_relation("Vega Beyaz Eşya A.Ş.", "Vega Beyaz Esya A.S.") == "format_only"
    assert pos.name_relation("Vega Beyaz Esya AS", "Vega Beyaz Esya A.S.") == "format_only"
    assert pos.name_relation("ACME CTRL LTD", "Acme Controls Ltd") == "different"
    assert pos.name_relation("Merlion SmartHome", "Merlion Smart Home Pte Ltd") == "different"


def test_hand_written_sample_rows_pass_the_rules():
    for r in read_jsonl(EVAL):
        assert [f for f in pos.check_labels(r.text, r.labels, CATALOGUE)] == [], r.id


@pytest.mark.parametrize("text,lab,rule", [
    (row(qty="1100"), labels("commit", "no", "0"), "pack_qty"),
    (row(qty="1000"), labels("pack_qty_issue", "yes", "2"), "pack_qty"),
    (row(restricted="yes"), labels("commit", "no", "0"), "compliance"),
    (row(ref="INV-1/1", seen="INV-1/1"), labels("commit", "no", "0"), "duplicate"),
    (row(ref="inv-1/1", seen="INV-1/1, INV-2/2"), labels("pack_qty_issue", "yes", "2"), "duplicate"),
    (row(ref="INV-1/1", seen="INV-1/11"), labels("duplicate", "no", "3"), "duplicate"),
    (row(qty="1.000"), labels("commit", "no", "0"), "normalize"),
    (row(), labels("normalize", "no", "1"), "normalize"),
    (row(customer="ACME CTRL LTD"), labels("normalize", "no", "1"), "vendor_rename"),
    (row(customer="ACME CONTROLS LTD"), labels("vendor_rename", "no", "2"), "vendor_rename"),
    (row(qty="-500"), labels("commit", "no", "0"), "ask_human"),
    (row(ref="-"), labels("commit", "no", "0"), "ask_human"),
    (row(), labels("commit", "yes", "0"), "combo"),
    (row(customer="Customer Name", part="Part No", qty="Qty"), labels("ask_human", "yes", "3"), "no_op"),
])
def test_check_labels_catches_rule_breaks(text, lab, rule):
    found = {f.rule for f in pos.check_labels(text, lab, CATALOGUE) if f.level == "violation"}
    assert rule in found, found


@pytest.mark.parametrize("text,lab", [
    (row(qty="1000"), labels("commit", "no", "0")),
    (row(qty="500"), labels("commit", "no", "0")),                     # exactly MOQ
    (row(qty="1.100"), labels("pack_qty_issue", "yes", "2")),          # pack beats normalize
    (row(qty="750", customer="ACME CTRL LTD"), labels("vendor_rename", "no", "2")),
    (row(qty="1100", restricted="yes", seen="INV-1/1"), labels("compliance_review", "yes", "4")),
    (row(ref="INV-1/1", seen="INV-9/9, inv-1/1", qty="1100"), labels("duplicate", "no", "3")),
    (row(ref="INV-1/1", seen="INV-1/11"), labels("commit", "no", "0")),
    (row(period="Sep-26", part="wm-s3-n8r8"), labels("normalize", "no", "1")),
])
def test_check_labels_accepts_correct_labels(text, lab):
    assert [f for f in pos.check_labels(text, lab, CATALOGUE) if f.level == "violation"] == []


def test_procedural_generator_labels_hold_and_cover_every_combo():
    recs = pos_gen.generate(str(POOLS), str(ROOT / "configs/pools/parts.yaml"),
                            str(ROOT / "configs/synth.pos.yaml"), str(ROOT / "configs/pools/regions.yaml"),
                            n_per_scenario=8, pairs_per_type=4, seed=5, log=lambda *_: None)
    combos = Counter((r.labels["row_action"], r.labels["needs_human"], r.labels["severity"]) for r in recs)
    assert set(combos) == pos.VALID_COMBOS
    for r in recs:
        assert not [f for f in pos.check_labels(r.text, r.labels, CATALOGUE) if f.level == "violation"], r.text
    assert len({r.text for r in recs}) == len(recs)
    pairs = Counter(r.meta["pair"] for r in recs if "pair" in r.meta)
    assert pairs and set(pairs.values()) == {2}


def test_procedural_generator_is_deterministic():
    kw = dict(n_per_scenario=3, pairs_per_type=2, seed=9, log=lambda *_: None)
    args = (str(POOLS), str(ROOT / "configs/pools/parts.yaml"), str(ROOT / "configs/synth.pos.yaml"),
            str(ROOT / "configs/pools/regions.yaml"))
    a = [r.text for r in pos_gen.generate(*args, **kw)]
    b = [r.text for r in pos_gen.generate(*args, **kw)]
    assert a == b
