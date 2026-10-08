#!/usr/bin/env bash
set -euo pipefail

: "${LLAMA2_7B_PATH:?Please export LLAMA2_7B_PATH=/absolute/path/to/local/Llama-2-7b-hf}"
TRAIN_PATH=${TRAIN_PATH:-data/raw/train.json}
TEST_PATH=${TEST_PATH:-data/raw/test.json}

python src/fewshot_event_infer.py \
  --base_model "$LLAMA2_7B_PATH" \
  --train_path "$TRAIN_PATH" \
  --test_path "$TEST_PATH" \
  --k 4 \
  --output_file outputs/event_fewshot_predictions.jsonl \
  --max_new_tokens 16 \
  --temperature 0
