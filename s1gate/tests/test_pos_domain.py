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


# ---------------------------------------------------------------- regressions from the review

@pytest.mark.parametrize("raw,value", [
    ("16.1k", 16100), ("64.35k", 64350), ("1,00,000.00", 100000), ("1.000 Stk.", 1000),
    ("1000 Nos.", 1000), ("4,000 EA.", 4000),
])
def test_qty_formats_from_review(raw, value):
    assert pos.read_qty(raw) == (value, "formatted")


@pytest.mark.parametrize("raw,status", [
    ("5 220,00 HUF", "formatted"), ("2,31 Euro", "formatted"), ("41,90 TL", "formatted"),
    ("$2.31 USD", "formatted"), ("2,31€", "formatted"), ("2.31/2.40 USD", "invalid"),
    ("-2.31 USD", "impossible"), ("2.31 kr", "unknown"), ("2.31 USD EUR", "unknown"),
])
def test_price_formats_from_review(raw, status):
    assert pos.read_price(raw)[1] == status


@pytest.mark.parametrize("customer,country,lab", [
    ("Total Power Solutions Ltd", "GB", labels("commit", "no", "0")),     # not a subtotal
    ("Check Point Software Ltd.", "IL", labels("commit", "no", "0")),     # not a note
    ("Acme Controls Ltd", "NA", labels("commit", "no", "0")),             # Namibia, not n/a
    ("Acme Controls Ltd", "Korea, Republic of", labels("normalize", "no", "1")),
])
def test_correct_rows_from_review_pass(customer, country, lab):
    text = ROW.format(qty="1000", ref="INV-1/1", restricted="no", seen="none").replace(
        "customer=Acme Controls Ltd; country=PL", f"customer={customer}; country={country}").replace(
        "master: customer=Acme Controls Ltd", f"master: customer={customer}")
    assert [f.message for f in pos.check_labels(text, lab, CATALOGUE) if f.level == "violation"] == []


def test_review_misses_are_now_violations():
    unread = row(country="Korea Rep")
    assert "normalize" in {f.rule for f in pos.check_labels(unread, labels("commit", "no", "0"), CATALOGUE)}
    sales = row(unit_price="")
    assert "no_op" in {f.rule for f in pos.check_labels(sales, labels("no_op", "no", "3"), CATALOGUE)
                       if f.level == "violation"}
    accents = row(customer="Dong Son Audio Company Limited").replace(
        "master: customer=Acme Controls Ltd", "master: customer=Đông Sơn Audio Company Limited")
    assert not [f for f in pos.check_labels(accents, labels("normalize", "no", "1"), CATALOGUE) if f.level == "violation"]
    odd_digits = row().replace("spq=250", "spq=²50")
    pos.check_labels(odd_digits, labels("commit", "no", "0"), CATALOGUE)  # must not raise


def _gen(seed, n=6, pairs=6, recipe=None):
    return pos_gen.generate(str(POOLS), str(ROOT / "configs/pools/parts.yaml"),
                            recipe or str(ROOT / "configs/synth.pos.yaml"), str(ROOT / "configs/pools/regions.yaml"),
                            n_per_scenario=n, pairs_per_type=pairs, seed=seed, log=lambda *_: None)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_procgen_pairs_are_never_orphaned(seed):
    recs = _gen(seed)
    sides = Counter(r.meta["pair"] for r in recs if "pair" in r.meta)
    assert set(sides.values()) == {2}
    assert len({r.text for r in recs}) == len(recs)


def test_procgen_skips_unknown_recipe_ids(tmp_path):
    import yaml
    recipe = yaml.safe_load((ROOT / "configs/synth.pos.yaml").read_text())
    recipe["scenarios"].append({"id": "C99", "target": {"row_action": "commit"}, "hint": "new"})
    recipe["pairs"].append({"id": "M01", "change": "x", "a": {"row_action": "commit"}, "b": {"row_action": "duplicate"}})
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(recipe))
    recs = _gen(1, n=2, pairs=1, recipe=str(path))
    assert recs and not {r.meta["scenario"] for r in recs} & {"C99", "M01"}


def test_procgen_scenario_kinds_mean_what_they_say():
    g = pos_gen.Generator(__import__("json").loads(POOLS.read_text()),
                          pos_gen.load_parts(ROOT / "configs/pools/parts.yaml"), seed=3)
    for _ in range(20):
        r, _ = g.scenario("V2")   # suffix change or drop: the name core survives
        core = g.split_suffix(r.m_customer)[0]
        assert r.customer.startswith(core) and r.customer != r.m_customer
        r, _ = g.scenario("D3")   # a real case/space variant, never the ref verbatim
        seen = r.seen_refs.split(", ")
        assert r.ref not in seen and pos.norm_ref(r.ref) in {pos.norm_ref(x) for x in seen}


# ---------------------------------------------------------------- final verification round

def _named(customer, master=None, **swap):
    text = row(**swap).replace("customer=Acme Controls Ltd; country", f"customer={customer}; country", 1)
    return text.replace("master: customer=Acme Controls Ltd", f"master: customer={master or customer}", 1)


def _violations(text, lab):
    return [f.message for f in pos.check_labels(text, lab, CATALOGUE) if f.level == "violation"]


@pytest.mark.parametrize("name", ["Summa Technologies Inc", "Calidad Total", "Totaal Techniek B.V.", "Kids-Total",
                                  "Toplam Enerji A.S.", "Total General Contracting LLC", "Total Power Solutions Ltd"])
def test_marker_words_inside_a_real_customer_name_are_the_name(name):
    for broken in (dict(ref=""), dict(part="WM-S3-N8R8 / WM-S3-N8R9")):
        text = _named(name, **broken)
        assert pos.facts(text, CATALOGUE).nonsales is None
        assert not _violations(text, labels("ask_human", "yes", "3"))
    renamed = _named("Total Power Solutions", "Total Power Solutions Ltd", ref="")
    assert pos.facts(renamed, CATALOGUE).nonsales is None


@pytest.mark.parametrize("customer,master", [("Subtotal", "unknown"), ("Total", "Total Power Solutions Ltd"),
                                             ("Acme Controls Ltd Total", "Acme Controls Ltd"),
                                             ("Total Acme Controls Ltd", "Acme Controls Ltd")])
def test_subtotal_rows_still_read_as_subtotals(customer, master):
    text = ("distributor=Northwind Components; period=2026-09; customer=%s; country=; part=; qty=12,500; unit_price=; "
            "ref= || master: customer=%s; spq=?; moq=?; restricted=no; seen_refs=none" % (customer, master))
    assert pos.facts(text, CATALOGUE).nonsales == "subtotal"
    assert not _violations(text, labels("no_op", "no", "3"))


def test_no_op_on_a_sales_line_with_a_failed_lookup_is_a_violation():
    text = row().replace("master: customer=Acme Controls Ltd; spq=250; moq=500", "master: customer=unknown; spq=?; moq=?")
    assert _violations(text, labels("no_op", "no", "1"))
    assert not _violations(text, labels("ask_human", "yes", "3"))


def test_notes_in_the_customer_field_are_judged_against_the_master_name():
    assert _violations(_named("Acme Controls Ltd ??", "Acme Controls Ltd"), labels("normalize", "no", "1"))
    assert not _violations(_named("Check Point Software", "Check Point Software Ltd."), labels("vendor_rename", "no", "2"))
    assert not _violations(_named("TBC Corporation", "TBC Corp"), labels("vendor_rename", "no", "2"))
    assert _violations(_named("Check Point Software (TBC)", "Check Point Software Ltd."), labels("vendor_rename", "no", "2"))


@pytest.mark.parametrize("raw,status", [
    ("−2.31 USD", "impossible"), ("(2.31) USD", "impossible"), ("($2.31)", "impossible"), ("USD (2.31)", "impossible"),
    ("R$0,00", "impossible"), ("US$0.00", "impossible"), ("S$0.00", "impossible"),
    ("2.31-2.40 USD", "invalid"), ("2.31 2.40 USD", "invalid"), ("2.31 or 2.40 USD", "invalid"),
    ("approx 2.31 USD", "invalid"), ("~2.31 USD", "invalid"), ("ca. 2.31 EUR", "invalid"),
    ("R$10,50", "formatted"), ("US$2.31", "formatted"), ("CA$2.31", "formatted"), ("5 220,00 HUF", "formatted"),
    ("12 345,67 CZK", "formatted"), ("2.31 USD (net 30)", "unknown"),
])
def test_price_reader_final_round(raw, status):
    assert pos.read_price(raw)[1] == status


def test_procgen_reports_real_data_errors_instead_of_skipping(tmp_path):
    import json
    pools = json.loads(POOLS.read_text())
    region = next(iter(pools))
    pools[region]["customers"].append({"Name": "Typo Key Customer Ltd", "country": "IN"})
    bad = tmp_path / "pools.json"
    bad.write_text(json.dumps(pools))
    with pytest.raises(KeyError):
        for seed in range(5):  # the bad entry is drawn sooner or later
            pos_gen.generate(str(bad), str(ROOT / "configs/pools/parts.yaml"), str(ROOT / "configs/synth.pos.yaml"),
                             str(ROOT / "configs/pools/regions.yaml"), n_per_scenario=6, pairs_per_type=3, seed=seed,
                             log=lambda *_: None)


def test_procgen_rename_scenarios_meet_their_definitions():
    import json
    g = pos_gen.Generator(json.loads(POOLS.read_text()), pos_gen.load_parts(ROOT / "configs/pools/parts.yaml"), seed=5)
    for _ in range(40):
        r, _ = g.scenario("VU1")   # a sister company: shares the first word of a multi-word core
        core = g.split_suffix(r.m_customer)[0].split()
        assert len(core) >= 2 and r.customer.split()[0] == core[0]
        built = g.pair("H06")
        if built:
            (_, _), (b, _) = built  # row b is an abbreviation, never a typo or a suffix change
            assert g.split_suffix(b.customer)[0] != g.split_suffix(b.m_customer)[0]
            assert len(b.customer) < len(b.m_customer) or b.customer.isupper() or any(w.isupper() for w in b.customer.split())
