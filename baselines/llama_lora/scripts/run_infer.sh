#!/usr/bin/env bash
set -euo pipefail
python src/infer.py \
  --base_model "${LLAMA2_7B_PATH}" \
  --adapter_path outputs/llama2_7b_lora_sft \
  --instruction "请解释什么是法律判决预测。" \
  --max_new_tokens 256
