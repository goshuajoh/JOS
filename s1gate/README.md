# s1gate

A System 1 decision gate: a small local model that reads one record and returns one option
per question, with a calibrated probability, in a single forward pass. Built for the
Classify state of the POS ingestion FSM; the schema is the only POS-specific part.

The full walkthrough lives in the guide doc ("Building a System 1 Decision Gate").
This file is the quick reference.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # core: schema, metrics, calibration, CLI
pip install -e ".[gliner]"              # GLiNER2.5-Decide backend (CPU is fine)
pip install -e ".[gliner-train]"        # fine-tune GLiNER
pip install -e ".[qwen]"                # Qwen option-logit backend + LoRA training (GPU advised)
pip install -e ".[synth]"               # synthetic data via Anthropic or OpenAI-compatible APIs
pip install -e ".[serve]"               # HTTP service
pytest -q                               # 14 tests, no model download needed
```

## Layout

```
configs/schema.pos.yaml     step 1  questions + options (the contract)
configs/synth.pos.yaml      step 3  synthetic-data recipe
data/eval/                  step 1  hand-labelled eval set (sample included; replace it)
s1gate/schema.py            step 1  schema model, Enum + JSON-schema export
s1gate/records.py           1, 3    JSONL records, validation, split, dedupe, decontamination
s1gate/backends/            step 2  keyword (floor), gliner, qwen (option-logit), endpoint
s1gate/metrics.py           step 2  accuracy, macro-F1, NLL, Brier, ECE, reliability, coverage
s1gate/synth.py             step 3  teacher LLM generation + blind verification
s1gate/train_gliner.py      4-5     GLiNER fine-tune (option order shuffled)
s1gate/train_qwen.py        4-5     Qwen LoRA option scorer (CE over option letters)
s1gate/calibrate.py         step 6  temperature scaling per question
s1gate/thresholds.py        step 7  escalation cut-off per question for a target accuracy
s1gate/gate.py              step 8  runtime Gate for the FSM + decision log + corrections
s1gate/serve.py             step 8  FastAPI: /decide /predict /feedback /health
s1gate/cli.py               all     `s1gate <command>`
```

## The loop in commands

```bash
S=configs/schema.pos.yaml
# 1  schema + eval set
s1gate validate --schema $S --data data/eval/eval.jsonl
s1gate split --data data/labelled.jsonl --frac 0.5 --stratify row_action \
             --out-rest data/dev.jsonl --out-held data/eval/eval.jsonl

# 2  zero-shot baselines (keyword = the floor)
s1gate eval --schema $S --data data/eval/eval.jsonl --backend keyword
s1gate eval --schema $S --data data/eval/eval.jsonl --backend gliner --report-out runs/gliner0.json
s1gate eval --schema $S --data data/eval/eval.jsonl --backend qwen --model Qwen/Qwen3-0.6B

# 3  synthetic data, then clean it against dev + eval
s1gate synth --schema $S --recipe configs/synth.pos.yaml --out data/synth/raw.jsonl --scale 0.1
s1gate clean --train data/synth/raw.jsonl --held-out data/dev.jsonl data/eval/eval.jsonl \
             --out data/synth/train.jsonl

# 4-5  fine-tune (pick one)
s1gate train-gliner --schema $S --train data/synth/train.jsonl --dev data/dev.jsonl --out runs/gliner
s1gate train-qwen   --schema $S --train data/synth/train.jsonl --dev data/dev.jsonl --out runs/qwen

# 6-7  calibrate + thresholds on DEV predictions, then report on EVAL
s1gate eval --schema $S --data data/dev.jsonl --backend gliner --model runs/gliner/final \
            --preds-out runs/dev_preds.jsonl
s1gate calibrate  --schema $S --preds runs/dev_preds.jsonl --out runs/calibration.json
s1gate thresholds --schema $S --preds runs/dev_preds.jsonl --calibration runs/calibration.json \
                  --target 0.97 --target-q row_action=0.99 --out runs/thresholds.json
s1gate eval --schema $S --data data/eval/eval.jsonl --backend gliner --model runs/gliner/final \
            --calibration runs/calibration.json

# 8  serve
s1gate serve --schema $S --backend gliner --model runs/gliner/final \
             --calibration runs/calibration.json --thresholds runs/thresholds.json
```

## What was and was not tested

- Tested here: schema, records, metrics, calibration, thresholds, gate, HTTP service, the
  synth loop (with a fake teacher), the Qwen backend and LoRA training end to end on a tiny
  random Qwen3 model, and GLiNER training rows against gliner2 2.1.0's own loader.
- Not run here (model weights were unreachable from the build sandbox): GLiNER2.5-Decide
  inference and fine-tuning, real Qwen3 checkpoints, real teacher-model calls. The first
  `s1gate eval --backend gliner` run is the check; see `backends/gliner.py` for what to
  look at if the calibration curve comes out flat.
