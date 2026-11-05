#!/usr/bin/env bash
# Step 2: zero-shot baselines on DEV, then Gate B. Run on the machine the gate will run on
# (CPU latency is part of the result). Needs: pip install -e ".[gliner,qwen]"
#
#   scripts/run_baselines.sh                       # keyword, GLiNER2.5-Decide, Qwen3 0.6B and 1.7B
#   KEV_MODEL=<hf id of the KEV 0.8B / JevK5 checkpoint> scripts/run_baselines.sh
#   DATA=data/dev.jsonl DEVICE=cpu scripts/run_baselines.sh
set -euo pipefail
cd "$(dirname "$0")/.."

S=${SCHEMA:-configs/schema.pos.yaml}
DATA=${DATA:-data/dev.jsonl}
DEVICE=${DEVICE:-cpu}
OUT=${OUT:-runs/baselines}
mkdir -p "$OUT"

REPORTS=()  # only this run's eval reports go to compare (not calibration files or older runs)
run() {  # name, then backend args
  local name=$1; shift
  echo "=== $name"
  if s1gate eval --schema "$S" --data "$DATA" --device "$DEVICE" "$@" \
      --preds-out "$OUT/$name.preds.jsonl" --report-out "$OUT/$name.json"; then
    REPORTS+=("$OUT/$name.json")
  else
    echo "!!! $name failed"
  fi
}

run keyword  --backend keyword
run gliner   --backend gliner --model "${GLINER_MODEL:-fastino/GLiNER2.5-Decide}" --batch-size 8
[ -n "${GLINER_MULTI:-}" ] && run gliner_multi --backend gliner --model fastino/GLiNER2.5-multi-Decide --batch-size 8
run qwen06   --backend qwen --model Qwen/Qwen3-0.6B --batch-size 4
run qwen17   --backend qwen --model Qwen/Qwen3-1.7B --batch-size 2
[ -n "${KEV_MODEL:-}" ] && run kev --backend qwen --model "$KEV_MODEL" --batch-size 4

[ ${#REPORTS[@]} -gt 0 ] || { echo "no baseline finished"; exit 1; }
s1gate compare --reports "${REPORTS[@]}" --target 0.97 --target-q row_action=0.99 needs_human=0.98 severity=0.90

# reliability plots for the best zero-shot model, calibrated on these dev predictions
best=${BEST:-gliner}
if [[ " ${REPORTS[*]} " == *" $OUT/$best.json "* ]]; then  # only if $best ran in this run
  s1gate calibrate --schema "$S" --preds "$OUT/$best.preds.jsonl" --out "$OUT/$best.calibration.json"
  s1gate report --schema "$S" --preds "$OUT/$best.preds.jsonl" --calibration "$OUT/$best.calibration.json" \
      --title "zero-shot $best on $(basename "$DATA")" --out "$OUT/$best.report.html"
else
  echo "$best did not run in this run: no calibration or reliability report (set BEST=<a backend that ran>)"
fi
