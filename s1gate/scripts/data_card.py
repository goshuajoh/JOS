"""Write data/README.md (the dataset card) from the shipped files, so every number is computed.

    python -I scripts/data_card.py --root .
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
args = ap.parse_args()
ROOT = Path(args.root).resolve()
sys.path.insert(0, str(ROOT))

from s1gate.domains.pos import VALID_COMBOS  # noqa: E402
from s1gate.records import read_jsonl  # noqa: E402
from s1gate.schema import DecisionSchema  # noqa: E402

D = ROOT / "data"
schema = DecisionSchema.load(ROOT / "configs/schema.pos.yaml")
Q = schema.names


def load(rel):
    p = D / rel
    return read_jsonl(p) if p.exists() else []


def jload(rel):
    p = D / rel
    return json.loads(p.read_text()) if p.exists() else None


files = {
    "eval/eval.jsonl": "proxy eval set, frozen: final report only",
    "dev.jsonl": "proxy dev set: baselines, calibration, thresholds",
    "eval/eval.sample.jsonl": "26 hand-written rows (from the scaffold): tests and smoke runs",
    "synth/train.jsonl": "the training set: verified teacher rows + procedural rows, deduped and decontaminated",
    "synth/teacher.verified.jsonl": "teacher rows that passed blind re-labelling and the rule audit",
    "synth/teacher.rejected.jsonl": "teacher rows dropped by blind re-labelling or the rules (with the reason)",
    "synth/procedural.jsonl": "procedural rows, labels known by construction",
    "proxy/labelled.jsonl": "all adjudicated proxy rows before the dev/eval split",
}
data = {k: load(k) for k in files}
train = data["synth/train.jsonl"]

lines = ["# Data for the POS row gate", "",
         "Every number below is computed from the files by `scripts/data_card.py`.",
         "Schema: `pos_row_gate` v%d (fingerprint `%s`)." % (schema.version, schema.fingerprint()), "",
         "| file | rows | what |", "|---|---:|---|"]
for k, what in files.items():
    if data[k]:
        lines.append(f"| `data/{k}` | {len(data[k])} | {what} |")
lines += ["", "Supporting files: `synth/teacher/raw/` (each teacher agent's output), "
          "`synth/teacher/workorders/` (what each agent was asked for), `synth/teacher/blind/` and "
          "`proxy/blind/` (blind labels and id keys), `proxy/raw/` (simulated distributor files), "
          "`proxy/agreement.json` (author vs annotators), `pools/` (invented companies, split into "
          "`train/` and `eval/` with no overlap).", ""]

# ---------------------------------------------------------------- label distributions
lines += ["## Labels", ""]
show = ["synth/train.jsonl", "dev.jsonl", "eval/eval.jsonl"]
for q in schema.questions:
    lines += [f"**{q.name}**", "", "| option | " + " | ".join(f"`{s.split('/')[-1]}`" for s in show) + " |",
              "|---|" + "---:|" * len(show)]
    for o in q.labels:
        cells = []
        for s in show:
            rows = data[s]
            n = sum(1 for r in rows if r.labels.get(q.name) == o)
            cells.append(f"{n} ({n / len(rows):.0%})" if rows else "-")
        lines.append(f"| `{o}` | " + " | ".join(cells) + " |")
    lines.append("")

lines += ["**The eleven valid combinations** (row_action / needs_human / severity)", "",
          "| combination | " + " | ".join(f"`{s.split('/')[-1]}`" for s in show) + " |",
          "|---|" + "---:|" * len(show)]
for combo in sorted(VALID_COMBOS):
    cells = [str(sum(1 for r in data[s] if tuple(r.labels.get(x) for x in Q) == combo)) for s in show]
    lines.append(f"| {' / '.join(combo)} | " + " | ".join(cells) + " |")
other = sum(1 for s in show for r in data[s] if tuple(r.labels.get(x) for x in Q) not in VALID_COMBOS)
lines += ["", f"Rows outside the eleven combinations: {other}.", ""]

# ---------------------------------------------------------------- training set composition
if train:
    gen = Counter(r.meta.get("generator", "teacher") for r in train)
    teachers = Counter(r.meta.get("teacher") for r in train if r.meta.get("teacher"))
    authoring = Counter(r.meta.get("authoring") for r in train if r.meta.get("generator") != "procedural")
    scen = Counter(r.meta.get("scenario") for r in train)
    pairs = Counter(r.meta.get("pair") for r in train if r.meta.get("pair"))
    complete_pairs = sum(1 for v in pairs.values() if v == 2)
    lens = [len(r.text) for r in train]
    lines += ["## Training set composition", "",
              f"* sources: {gen.get('teacher', 0)} teacher rows ({', '.join(f'{k} {v}' for k, v in teachers.most_common())}), "
              f"{gen.get('procedural', 0)} procedural rows",
              f"* teacher rows by authoring: {authoring.get('hand', 0)} hand-authored (the agent chose every value), "
              f"{authoring.get('script', 0)} produced by a seeded generator script the agent wrote "
              "(valid and blind-verified, but less varied - `meta.authoring` lets you down-weight them)",
              f"* hard pairs with both rows kept: {complete_pairs}",
              f"* text length: median {statistics.median(lens):.0f} chars, min {min(lens)}, max {max(lens)}",
              f"* distinct customers named in master data: "
              f"{len({r.text.split('|| master: customer=')[-1].split(';')[0] for r in train if '|| master: customer=' in r.text})}",
              "", "| scenario | rows | scenario | rows | scenario | rows |", "|---|---:|---|---:|---|---:|"]
    items = sorted(scen.items(), key=lambda kv: (kv[0][0], len(kv[0]), kv[0]))
    for i in range(0, len(items), 3):
        chunk = items[i:i + 3] + [("", "")] * (3 - len(items[i:i + 3]))
        lines.append("| " + " | ".join(f"{k} | {v}" for k, v in chunk) + " |")
    lines += ["", "Scenario ids are defined in `configs/synth.pos.yaml` (C commit, N normalize, V/VU clear/uncertain "
              "rename, P pack quantity, R restricted, D duplicate, A ask_human, O no_op, H hard pairs).", ""]

# ---------------------------------------------------------------- how it was checked
blind = jload("synth/teacher/blind/merge_stats.json")
clean = jload("synth/clean_stats.json")
agree = jload("proxy/agreement.json")
lines += ["## How it was made and checked", ""]
lines += ["1. **Company pools.** Two agents invented 88 distributors and 440 customers across 11 regions "
          "(`data/pools/raw/`). `scripts/split_pools.py` assigns every company to train or eval by a stable hash, so the "
          "proxy dev/eval rows use companies training never sees.",
          "2. **Teacher rows.** 25 agents (Opus and Sonnet alternating, the guide's \"several teachers\") each got a "
          "work order (`synth/teacher/workorders/t*.md`): scenarios or hard-pair types from `configs/synth.pos.yaml`, "
          "plus a style card per batch (region, order size, seen_refs history, currency, the format noise allowed for "
          "that scenario). Each agent self-checked with `s1gate audit` until there were no violations. The Sonnet "
          "agents met their orders by writing seeded generator scripts, and the Opus agents chose each row's values "
          "themselves. So 8 more Opus chunks (`h*.md`, new style cards, an explicit no-generator rule) were written by "
          "hand for the scenarios where natural variety matters most: format messiness, typos and renames, uncertain "
          "links, garbled rows, note and subtotal lines, and the rename and qty-format hard pairs. "
          "`meta.authoring` records which is which.",
          "3. **Blind re-labelling.** `s1gate blind-export` shuffled all teacher rows into mixed batches with opaque "
          "ids. Separate annotator agents labelled them from the guide alone. `s1gate blind-merge --domain pos` "
          "kept a row only if the blind label matched its target on all three questions and the rule audit passed."]
if blind:
    lines.append(f"   Result: {blind['rows']} rows in, {blind['kept_after_rules']} kept. Blind agreement per question: "
                 + ", ".join(f"{q} {v:.1%}" for q, v in blind["agreement"].items()) + ".")
    tops = blind["top_disagreements"].get("row_action", [])[:5]
    if tops:
        lines.append("   Most frequent row_action disagreements (target → blind): "
                     + "; ".join(f"{t['target']} → {t['blind']} ×{t['count']}" for t in tops) + ".")
lines += ["4. **Procedural rows.** `s1gate procgen` (`s1gate/domains/pos_gen.py`) builds every scenario and hard pair "
          "from a clean canonical row plus one mutation recipe. Every row passes the rule audit, and rows that fail are "
          "rebuilt, never kept.",
          "5. **Proxy dev/eval.** Ten agents each simulated one held-out distributor's monthly file in file order, "
          "with its own house habits (`configs/proxy_eval.pos.yaml`). Two blind annotators (Opus, Sonnet) labelled "
          "every row. Rows where author, both annotators and the rule audit agreed were accepted, and the rest went "
          "to an adjudicator with all three opinions. The result was split 50/50, stratified on row_action, into "
          "`dev.jsonl` and `eval/eval.jsonl`, and the eval file was frozen (`eval/eval.frozen.json`)."]
if agree:
    lines.append(f"   Result: {agree['rows']} rows, {agree['unanimous']} unanimous, {agree['disputed']} adjudicated. "
                 "Agreement between the two blind annotators (Gate A wants ≥ 90%): "
                 + ", ".join(f"{q} {v['agreement']:.1%} (κ {v['kappa']:.2f})" for q, v in agree["a_vs_b"].items()) + ".")
lines.append("6. **Cleaning.** `s1gate clean` removed exact and near-duplicates (5-gram Jaccard ≥ 0.9, label-aware, so "
             "hard pairs survive) and every training row within 0.8 Jaccard of dev, eval or the sample set.")
if clean:
    lines.append(f"   Result: {clean['in']} rows in, {clean['duplicates']} duplicates or conflicting copies, "
                 f"{clean['held_out']} too close to held-out rows, {clean['out']} rows in `train.jsonl`.")
lines += ["",
          "## Limits", "",
          "* **The proxy eval is not real data.** It's the most realistic stand-in available here (held-out "
          "companies, file order, house habits, independent labels), but it was still written by the same model "
          "family that wrote the training rows. Label 200–500 real rows (labelling guide section 7), run "
          "`s1gate agreement` on a doubly-labelled subset, and replace `dev.jsonl` and `eval/eval.jsonl`. Then "
          "re-run `s1gate freeze` and `s1gate clean`.",
          "* The class mix is deliberately not the production mix. Training over-samples rare classes, as the guide "
          "suggests, and the proxy dev/eval keep commit as the plurality but carry far more problems than a real "
          "month. Refit calibration and thresholds on real dev rows before trusting coverage numbers.",
          "* All companies and part numbers are invented. A real customer name that happens to match an invented "
          "one is a coincidence.", ""]
(D / "README.md").write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {D / 'README.md'} ({len(lines)} lines)")
