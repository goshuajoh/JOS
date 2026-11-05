# Data for the POS row gate

Every number below is computed from the files by `scripts/data_card.py`.
Schema: `pos_row_gate` v2 (fingerprint `19f942aef48c`).

| file | rows | what |
|---|---:|---|
| `data/eval/eval.jsonl` | 199 | proxy eval set, frozen: final report only |
| `data/dev.jsonl` | 201 | proxy dev set: baselines, calibration, thresholds |
| `data/eval/eval.sample.jsonl` | 26 | 26 hand-written rows (from the scaffold): tests and smoke runs |
| `data/synth/train.jsonl` | 8905 | the training set: verified teacher rows + procedural rows, deduped and decontaminated |
| `data/synth/teacher.verified.jsonl` | 3956 | teacher rows that passed blind re-labelling and the rule audit |
| `data/synth/procedural.jsonl` | 4962 | procedural rows, labels known by construction |
| `data/proxy/labelled.jsonl` | 400 | all adjudicated proxy rows before the dev/eval split |

Supporting files: `synth/teacher/raw/` (each teacher agent's output), `synth/teacher/workorders/` (what each agent was asked for), `synth/teacher/blind/` and `proxy/blind/` (blind labels and id keys), `proxy/raw/` (simulated distributor files), `proxy/agreement.json` (author vs annotators), `pools/` (invented companies, split into `train/` and `eval/` with no overlap).

## Labels

**row_action**

| option | `train.jsonl` | `dev.jsonl` | `eval.jsonl` |
|---|---:|---:|---:|
| `commit` | 1471 (17%) | 66 (33%) | 66 (33%) |
| `normalize` | 1554 (17%) | 37 (18%) | 36 (18%) |
| `vendor_rename` | 1623 (18%) | 27 (13%) | 27 (14%) |
| `pack_qty_issue` | 1014 (11%) | 19 (9%) | 20 (10%) |
| `compliance_review` | 543 (6%) | 12 (6%) | 12 (6%) |
| `duplicate` | 726 (8%) | 11 (5%) | 10 (5%) |
| `no_op` | 728 (8%) | 15 (7%) | 14 (7%) |
| `ask_human` | 1246 (14%) | 14 (7%) | 14 (7%) |

**needs_human**

| option | `train.jsonl` | `dev.jsonl` | `eval.jsonl` |
|---|---:|---:|---:|
| `yes` | 3372 (38%) | 50 (25%) | 52 (26%) |
| `no` | 5533 (62%) | 151 (75%) | 147 (74%) |

**severity**

| option | `train.jsonl` | `dev.jsonl` | `eval.jsonl` |
|---|---:|---:|---:|
| `0` | 1471 (17%) | 66 (33%) | 66 (33%) |
| `1` | 1899 (21%) | 45 (22%) | 41 (21%) |
| `2` | 2068 (23%) | 41 (20%) | 41 (21%) |
| `3` | 2816 (32%) | 34 (17%) | 33 (17%) |
| `4` | 651 (7%) | 15 (7%) | 18 (9%) |

**The eleven valid combinations** (row_action / needs_human / severity)

| combination | `train.jsonl` | `dev.jsonl` | `eval.jsonl` |
|---|---:|---:|---:|
| ask_human / yes / 3 | 1246 | 14 | 14 |
| commit / no / 0 | 1471 | 66 | 66 |
| compliance_review / yes / 4 | 543 | 12 | 12 |
| duplicate / no / 3 | 726 | 11 | 10 |
| no_op / no / 1 | 345 | 8 | 5 |
| no_op / no / 3 | 275 | 4 | 3 |
| no_op / no / 4 | 108 | 3 | 6 |
| normalize / no / 1 | 1554 | 37 | 36 |
| pack_qty_issue / yes / 2 | 1014 | 19 | 20 |
| vendor_rename / no / 2 | 1054 | 22 | 21 |
| vendor_rename / yes / 3 | 569 | 5 | 6 |

Rows outside the eleven combinations: 0.

## Training set composition

* sources: 3955 teacher rows (opus 2494, sonnet 1461), 4950 procedural rows
* teacher rows by authoring: 2494 hand-authored (the agent chose every value), 1461 produced by a seeded generator script the agent wrote (valid and blind-verified, but less varied - `meta.authoring` lets you down-weight them)
* hard pairs with both rows kept: 975
* text length: median 292 chars, min 168, max 400
* distinct customers named in master data (a failed lookup's 'unknown' not counted): 308

| scenario | rows | scenario | rows | scenario | rows |
|---|---:|---|---:|---|---:|
| A1 | 180 | A2 | 250 | A3 | 250 |
| A4 | 126 | A5 | 90 | C1 | 180 |
| C2 | 108 | C3 | 200 | C4 | 125 |
| C5 | 150 | C6 | 108 | D1 | 144 |
| D2 | 180 | D3 | 108 | D4 | 144 |
| H01 | 150 | H02 | 150 | H03 | 150 |
| H04 | 150 | H05 | 150 | H06 | 150 |
| H07 | 200 | H08 | 200 | H09 | 200 |
| H10 | 150 | H11 | 150 | H12 | 150 |
| N1 | 180 | N2 | 250 | N3 | 200 |
| N4 | 144 | N5 | 180 | N6 | 250 |
| N7 | 100 | O1 | 108 | O2 | 115 |
| O3 | 122 | O4 | 175 | O5 | 108 |
| P1 | 180 | P2 | 180 | P3 | 108 |
| P4 | 144 | P5 | 144 | P6 | 108 |
| R1 | 180 | R2 | 180 | R3 | 108 |
| V1 | 180 | V2 | 180 | V3 | 200 |
| V4 | 175 | V5 | 144 | VU1 | 144 |
| VU2 | 175 | VU3 | 150 |  |  |

Scenario ids are defined in `configs/synth.pos.yaml` (C commit, N normalize, V/VU clear/uncertain rename, P pack quantity, R restricted, D duplicate, A ask_human, O no_op, H hard pairs).

## How it was made and checked

1. **Company pools.** Two agents invented 88 distributors and 440 customers across 11 regions (`data/pools/raw/`). `scripts/split_pools.py` assigns every company to train or eval by a stable hash, so the proxy dev/eval rows use companies training never sees.
2. **Teacher rows.** 25 agents (Opus and Sonnet alternating, the guide's "several teachers") each got a work order (`synth/teacher/workorders/t*.md`): scenarios or hard-pair types from `configs/synth.pos.yaml`, plus a style card per batch (region, order size, seen_refs history, currency, the format noise allowed for that scenario). Each agent self-checked with `s1gate audit` until there were no violations. The Sonnet agents met their orders by writing seeded generator scripts, and the Opus agents chose each row's values themselves. So 8 more Opus chunks (`h*.md`, new style cards, an explicit no-generator rule) were written by hand for the scenarios where natural variety matters most: format messiness, typos and renames, uncertain links, garbled rows, note and subtotal lines, and the rename and qty-format hard pairs. `meta.authoring` records which is which.
3. **Blind re-labelling.** `s1gate blind-export` shuffled all teacher rows into mixed batches with opaque ids. Separate annotator agents labelled them from the guide alone. `s1gate blind-merge --domain pos` kept a row only if the blind label matched its target on all three questions and the rule audit passed.
   Result: 3956 rows in, 3956 kept. Blind agreement per question: row_action 100.0%, needs_human 100.0%, severity 100.0%.
4. **Procedural rows.** `s1gate procgen` (`s1gate/domains/pos_gen.py`) builds every scenario and hard pair from a clean canonical row plus one mutation recipe. Every row passes the rule audit, and rows that fail are rebuilt, never kept.
5. **Proxy dev/eval.** Ten agents each simulated one held-out distributor's monthly file in file order, with its own house habits (`configs/proxy_eval.pos.yaml`). Two blind annotators (Opus, Sonnet) labelled every row. Rows where author, both annotators and the rule audit agreed were accepted, and the rest went to an adjudicator with all three opinions. The result was split 50/50, stratified on row_action, into `dev.jsonl` and `eval/eval.jsonl`, with identical texts (a header row a file repeats) kept on one side, and the eval file was frozen (`eval/eval.frozen.json`).
   Result: 400 rows, 398 unanimous, 2 adjudicated. Agreement between the two blind annotators (Gate A wants ≥ 90%): row_action 99.5% (κ 0.99), needs_human 99.5% (κ 0.99), severity 99.8% (κ 1.00).
6. **Cleaning.** `s1gate clean` removed exact and near-duplicates (5-gram Jaccard ≥ 0.9, label-aware, so hard pairs survive) and every training row within 0.8 Jaccard of dev, eval or the sample set.
   Result: 8918 rows in, 13 duplicates or conflicting copies, 0 too close to held-out rows, 8905 rows in `train.jsonl`.

## Limits

* **The proxy eval is not real data.** It's the most realistic stand-in available here (held-out companies, file order, house habits, independent labels), but it was still written by the same model family that wrote the training rows. Label 200–500 real rows (labelling guide section 7), run `s1gate agreement` on a doubly-labelled subset, and replace `dev.jsonl` and `eval/eval.jsonl`. Then re-run `s1gate freeze` and `s1gate clean`.
* The class mix is deliberately not the production mix. Training over-samples rare classes, as the guide suggests, and the proxy dev/eval keep commit as the plurality but carry far more problems than a real month. Refit calibration and thresholds on real dev rows before trusting coverage numbers.
* Blind agreement on the teacher rows is perfect, which says the rows follow the guide, not that the task is easy: authors and annotators are the same model family and read the same guide, so they share its blind spots. Real rows will hold cases the guide does not settle; adjudicate those and extend the guide.
* All companies and part numbers are invented. A real customer name that happens to match an invented one is a coincidence.
