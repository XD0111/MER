#!/usr/bin/env bash
set -euo pipefail

: "${LLAMA2_7B_PATH:?Please export LLAMA2_7B_PATH=/absolute/path/to/local/Llama-2-7b-hf}"
python src/infer.py \
  --base_model "$LLAMA2_7B_PATH" \
  --adapter_path outputs/llama2_7b_lora_event_sft \
  --input_file data/processed_event/test.jsonl \
  --output_file outputs/event_test_predictions.jsonl \
  --max_new_tokens 16 \
  --temperature 0

python src/evaluate_generation.py --pred_file outputs/event_test_predictions.jsonl
