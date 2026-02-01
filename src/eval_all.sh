#!/usr/bin/env bash
set -euo pipefail

# Edit these paths
CKPT_DIR="./saved_model_normal"
TEST_DIR="../VCTK-DEMAND/test"
OUT_FILE="./eval_results.txt"

# Clear/create output file
: > "$OUT_FILE"

shopt -s nullglob

for ckpt in "$CKPT_DIR"/ckpt_epoch_*.pt; do
  echo "Evaluating: $ckpt" >&2

  # Run evaluation, capture stdout+stderr, then keep only the last line that contains metrics
  raw_out="$(python3 evaluation.py --model_path "$ckpt" --test_dir "$TEST_DIR" 2>&1)"

  # Grab the line containing "pesq:" (your script prints metrics once)
  metrics_line="$(echo "$raw_out" | grep -E "pesq:" | tail -n 1 | tr -s '[:space:]' ' ' | sed 's/^ *//;s/ *$//')"

  # If it didn't find metrics, store the error message instead
  if [[ -z "${metrics_line}" ]]; then
    metrics_line="ERROR: $(echo "$raw_out" | tail -n 5 | tr -s '[:space:]' ' ')"
  fi

  # Format: "file name that got weights from: results"
  echo "$(basename "$ckpt"): $metrics_line" >> "$OUT_FILE"
done

echo "Done. Results saved to $OUT_FILE" >&2