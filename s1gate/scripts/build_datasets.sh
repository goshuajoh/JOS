#!/usr/bin/env bash
# How the shipped datasets were assembled from the agent outputs (step 1 + step 3).
# Re-run after adding teacher files, blind labels or adjudications; every step is deterministic.
#
#   scripts/build_datasets.sh
#
# Inputs (already in data/):
#   synth/teacher/raw/*.jsonl          teacher rows with target labels (one file per agent)
#   synth/teacher/blind/*.labels.jsonl blind labels, opaque ids; key_r*.json map them back
#   proxy/raw/*.jsonl                  simulated distributor files with the author's labels
#   proxy/blind/*.{A,B}.labels.jsonl   two independent blind annotators; key_r*.json
#   proxy/adjudicated.jsonl            adjudicator decisions for disputed proxy rows
set -euo pipefail
cd "$(dirname "$0")/.."
S=configs/schema.pos.yaml

# --- teacher rows: keep a row only when the blind label equals its target and the rules pass
keys=$(ls data/synth/teacher/blind/key_r*.json)
s1gate blind-merge --schema "$S" --data data/synth/teacher/raw/*.jsonl \
    --labels data/synth/teacher/blind/*.labels.jsonl --key $(echo $keys | cut -d' ' -f1) \
    --extra-keys $(echo $keys | cut -d' ' -f2-) --domain pos \
    --out data/synth/teacher.verified.jsonl --rejected-out data/synth/teacher.rejected.jsonl \
    --stats-out data/synth/teacher/blind/merge_stats.json
python -I scripts/tag_authoring.py data/synth/teacher.verified.jsonl data/synth/teacher.rejected.jsonl

# --- procedural rows (labels by construction; seeded)
s1gate procgen --pools data/pools/train/companies.json --out data/synth/procedural.jsonl \
    --n-per-scenario 110 --pairs 50 --seed 1

# --- proxy dev/eval: author + two blind annotators -> unanimous / disputed -> adjudicated
python -I scripts/proxy_adjudication.py prep --raw data/proxy/raw/*.jsonl \
    --a data/proxy/blind/*.A.labels.jsonl --b data/proxy/blind/*.B.labels.jsonl \
    --keys data/proxy/blind/key_r*.json --out-dir data/proxy
python -I scripts/proxy_adjudication.py final --out-dir data/proxy --adjudicated data/proxy/adjudicated.jsonl
s1gate audit --data data/proxy/labelled.jsonl
s1gate split --data data/proxy/labelled.jsonl --frac 0.5 --stratify row_action --seed 13 \
    --out-rest data/dev.jsonl --out-held data/eval/eval.jsonl
s1gate validate --schema "$S" --data data/dev.jsonl data/eval/eval.jsonl
s1gate freeze --schema "$S" --data data/eval/eval.jsonl \
    --note "PROXY eval: 10 simulated distributor files, held-out companies, author + 2 blind annotators + adjudication. Replace with real rows."

# --- training set: dedupe (label-aware) + decontaminate against everything held out
s1gate clean --train data/synth/teacher.verified.jsonl data/synth/procedural.jsonl \
    --held-out data/dev.jsonl data/eval/eval.jsonl data/eval/eval.sample.jsonl \
    --out data/synth/train.jsonl --stats-out data/synth/clean_stats.json
s1gate audit --data data/synth/train.jsonl

# --- the floor on the proxy sets (step 2 starts here; real models need Hugging Face)
mkdir -p reports
s1gate eval --schema "$S" --data data/dev.jsonl --backend keyword \
    --preds-out runs/keyword_dev.preds.jsonl --report-out reports/keyword_dev.json
s1gate calibrate --schema "$S" --preds runs/keyword_dev.preds.jsonl --out reports/keyword_calibration.json
s1gate thresholds --schema "$S" --preds runs/keyword_dev.preds.jsonl --calibration reports/keyword_calibration.json \
    --target 0.97 --target-q row_action=0.99 needs_human=0.98 severity=0.90 --out reports/keyword_thresholds.json
s1gate eval --schema "$S" --data data/eval/eval.jsonl --backend keyword --calibration reports/keyword_calibration.json \
    --thresholds reports/keyword_thresholds.json --report-out reports/keyword_eval.json
s1gate report --schema "$S" --preds runs/keyword_dev.preds.jsonl --calibration reports/keyword_calibration.json \
    --thresholds reports/keyword_thresholds.json --title "keyword floor on proxy dev" --out reports/keyword_dev_report.html

python -I scripts/data_card.py --root .
