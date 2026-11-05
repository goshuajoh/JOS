# s1gate

A System 1 decision gate: a small local model that reads one record and returns one option
per question, with a calibrated probability, in a single forward pass. It's built for the
Classify state of the POS ingestion FSM. The schema and the POS domain pack
(`s1gate/domains/`) are the only POS-specific parts.

The full walkthrough is the guide doc, "Building a System 1 Decision Gate". The labelling rules
are in [`docs/labeling_guide.md`](docs/labeling_guide.md). The shipped data is described in
[`data/README.md`](data/README.md). This file is the quick reference.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # core: schema, metrics, calibration, CLI, POS data tools
pip install -e ".[gliner]"              # GLiNER2.5-Decide backend (CPU is fine)
pip install -e ".[gliner-train]"        # fine-tune GLiNER
pip install -e ".[qwen]"                # Qwen option-logit backend + LoRA training (GPU advised)
pip install -e ".[synth]"               # teacher / frontier models via Anthropic or OpenAI-compatible APIs
pip install -e ".[serve]"               # HTTP service
pytest -q                               # no model download needed; ML smoke tests skip without torch
```

## The eight steps, and what does each one

| Step | What | Commands | Files |
|---|---|---|---|
| 1 | Decision schema as Pydantic enums, incl. `NO_OP` and `ASK_HUMAN`; a frozen eval set | `validate`, `export-enums`, `label`, `agreement`, `split`, `freeze`, `audit` | `configs/schema.pos.yaml`, `s1gate/generated/pos_row_gate.py`, `docs/labeling_guide.md`, `data/eval/` |
| 2 | Zero-shot baselines: accuracy, ECE, CPU latency | `eval`, `compare` | `s1gate/backends/`, `s1gate/metrics.py` |
| 3 | Synthetic data: teacher rows, procedural rows, blind verification, dedupe against eval | `synth`, `procgen`, `blind-export`, `blind-merge`, `clean` | `configs/synth.pos.yaml`, `s1gate/synth.py`, `s1gate/domains/` |
| 4–5 | Backbone; cross-entropy over option scores, option order shuffled every epoch | `train-gliner`, `train-qwen` | `s1gate/train_gliner.py`, `s1gate/train_qwen.py` |
| 6 | Calibration: temperature (or vector) scaling, ECE, reliability plots | `calibrate`, `report` | `s1gate/calibrate.py`, `s1gate/report.py` |
| 7 | Escalation thresholds τ for a target accuracy or coverage, plus per-option rules | `thresholds` | `s1gate/thresholds.py` |
| 8 | Serve inside the FSM, log, relabel, retrain | `serve`, `queue`, `label`, `loop-report`, `build-train`, `bundle`, `promote` | `s1gate/gate.py`, `s1gate/serve.py`, `s1gate/loop.py`, `examples/pos_fsm.py` |

### Step 1: schema, enums, eval set

* `row_action` has eight options, among them `no_op` (not a sales line) and `ask_human` (the
  fallback). Training shuffles option order, so the prompt states precedence by name:
  `compliance_review > no_op > ask_human > duplicate > pack_qty_issue > vendor_rename >
  normalize > commit`.
* The YAML is the source of truth. `s1gate export-enums` writes `RowAction`, `NeedsHuman`,
  `Severity` and `PosRowGateDecision` as plain Pydantic/Enum code for the FSM, stamped with
  the schema fingerprint. A test fails if the module goes stale.
* `s1gate freeze` hashes the eval file. After that, `train-*`, `clean` and `build-train` refuse
  rows that copy it, `calibrate` and `thresholds` refuse predictions on it, and `eval` warns if
  the file changes.
* `s1gate agreement --a A.jsonl --b B.jsonl` gives per-question agreement, Cohen's kappa and
  the top disagreement pairs. That is Gate A, which wants at least 90%.
* `s1gate audit` checks labels against the structural rules of the guide: restricted means
  compliance, a seen ref means duplicate, SPQ/MOQ arithmetic, missing or impossible values,
  canonical formats, non-sales lines and the eleven valid label combinations.

### Step 3: data

The guide calls this step the one that decides how good the gate gets. `data/` ships three sources:

* **teacher rows**: written by frontier-model agents from `configs/synth.pos.yaml` (44
  scenarios, 12 hard-pair types, style cards per batch). Each agent self-checked with `audit`,
  then independent annotators labelled the rows blind, in shuffled mixed batches with opaque
  ids, and a row was kept only when the blind labels matched its target and the rule audit
  passed. `meta.authoring` says whether the agent wrote each row itself (`hand`) or wrote a
  seeded generator script (`script`). Down-weight the script rows if they turn out too uniform.
* **procedural rows** (`s1gate procgen`): labels known by construction, for exact SPQ/MOQ
  boundaries, near-miss refs, flags, stacked problems and every hard pair. Every row passes the
  rule audit.
* **proxy dev/eval**: a stand-in for real labelled rows. Ten simulated distributor files in
  file order with house habits, using a held-out company pool that training never sees,
  labelled by their author plus two blind annotators, then adjudicated. **Replace it with real
  rows as soon as they exist.**

`clean` dedupes with an exact prefix-filter Jaccard join. It is label-aware: near-duplicates
with *different* answers are hard pairs and both stay. Training rows within 0.8 Jaccard of
dev/eval are dropped.

## The loop in commands

```bash
S=configs/schema.pos.yaml
# 1  schema + eval set
s1gate export-enums --schema $S --check
s1gate audit --data data/dev.jsonl data/eval/eval.jsonl
s1gate freeze --schema $S --data data/eval/eval.jsonl

# 2  zero-shot baselines on DEV (keyword = the floor)
s1gate eval --schema $S --data data/dev.jsonl --backend keyword --report-out runs/keyword0.json
s1gate eval --schema $S --data data/dev.jsonl --backend gliner --report-out runs/gliner0.json
s1gate eval --schema $S --data data/dev.jsonl --backend qwen --model Qwen/Qwen3-0.6B --report-out runs/qwen0.json
s1gate compare --reports runs/keyword0.json runs/gliner0.json runs/qwen0.json --target-q row_action=0.99

# 3  synthetic data (shipped already; to regenerate or extend)
s1gate procgen --out data/synth/procedural.jsonl --n-per-scenario 110 --pairs 50
s1gate synth --schema $S --recipe configs/synth.pos.yaml --out data/synth/raw.jsonl --scale 0.1
s1gate clean --train data/synth/teacher.verified.jsonl data/synth/procedural.jsonl \
             --held-out data/dev.jsonl data/eval/eval.jsonl data/eval/eval.sample.jsonl \
             --out data/synth/train.jsonl

# 4-5  fine-tune (route A first)
s1gate train-gliner --schema $S --train data/synth/train.jsonl --dev data/dev.jsonl --out runs/gliner
s1gate train-qwen   --schema $S --train data/synth/train.jsonl --dev data/dev.jsonl --out runs/qwen

# 6-7  calibrate + thresholds on DEV predictions, then report on EVAL
s1gate eval --schema $S --data data/dev.jsonl --backend gliner --model runs/gliner/selected \
            --preds-out runs/dev_preds.jsonl
s1gate calibrate  --schema $S --preds runs/dev_preds.jsonl --out runs/calibration.json
s1gate thresholds --schema $S --preds runs/dev_preds.jsonl --calibration runs/calibration.json \
                  --target 0.97 --target-q row_action=0.99 needs_human=0.98 severity=0.90 \
                  --always-escalate row_action=compliance_review --out runs/thresholds.json
s1gate report --schema $S --preds runs/dev_preds.jsonl --calibration runs/calibration.json \
              --thresholds runs/thresholds.json --out runs/report.html
s1gate eval --schema $S --data data/eval/eval.jsonl --backend gliner --model runs/gliner/selected \
            --calibration runs/calibration.json --thresholds runs/thresholds.json --report-out runs/eval_v1.json

# 8  serve, then close the loop
s1gate bundle --schema $S --name v1 --backend gliner --model runs/gliner/selected \
              --calibration runs/calibration.json --thresholds runs/thresholds.json --report runs/eval_v1.json
s1gate serve --schema $S --backend gliner --model runs/gliner/selected \
             --calibration runs/calibration.json --thresholds runs/thresholds.json
s1gate loop-report --log runs/decisions.jsonl --corrections data/corrections.jsonl   # weekly
s1gate queue --log runs/decisions.jsonl --out data/review_queue.jsonl
s1gate label --schema $S --data data/review_queue.jsonl --out data/corrections.jsonl --source correction
s1gate build-train --corrections data/corrections.jsonl --synthetic data/synth/train.jsonl --out data/train.r2.jsonl
s1gate promote --candidate runs/eval_v2.json --current runs/eval_v1.json             # monthly
```

`examples/pos_fsm.py` is the Classify state wired to the typed enums, with shadow mode and a
simulated reviewer. It runs on the keyword floor with no model download.

Escalation to a frontier model is opt-in (`serve --escalate-to frontier --frontier-base-url
...`). It is asked only about questions under τ, and its answer must be a schema option, or
the row keeps `ask_human`. Rows sent there leave the process, so use only a deployment that
is approved for the data class.

## What was and was not tested

* **Tested here.** About 160 unit, CLI and data tests (`pytest -q`). They cover:
  * the schema and enums, records, and dedupe/decontamination (checked against brute force)
  * metrics, temperature and vector calibration, and thresholds with rules. A property test checks
    that the coverage and accuracy in `thresholds.json` are exactly what the `Gate` does.
  * the gate's escalation paths, the HTTP service, the loop tools and the HTML report
  * the POS rule checker and the procedural generator: every combination is covered and every
    label holds
  * the shipped data: it is rule-clean, the eval file is frozen and untouched, and no eval or
    dev rows or held-out companies appear in training
* **Smoke-tested on toy models.** The Qwen backend and LoRA training ran end to end on a tiny
  random Qwen3. Route A ran end to end on a tiny random GLiNER2, both full fine-tuning and LoRA:
  softmax loss patch, per-epoch checkpoints, LoRA merge, dev-NLL selection, then loading
  `selected/` in the backend. All training rows pass gliner2 2.1.0's own loader, with option
  order re-drawn on every read.
* **Reviewed.** Independent review agents went over the rule checker, generator, calibration,
  thresholds, CLI and loop code. Every defect they found is fixed and has a regression test.
* **Not run here.** GLiNER2.5-Decide and real Qwen3 checkpoints. Hugging Face was not reachable
  from the build environment, so there are no zero-shot or fine-tuned numbers yet, only the
  keyword floor in `reports/`. Your first `scripts/run_baselines.sh` on `data/dev.jsonl` is the
  real check. See `backends/gliner.py` for what to look at if the calibration curve comes out flat.
