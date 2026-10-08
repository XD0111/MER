#!/usr/bin/env bash
set -euo pipefail

: "${LLAMA2_7B_PATH:?Please export LLAMA2_7B_PATH=/absolute/path/to/local/Llama-2-7b-hf}"
accelerate launch src/train_lora_sft.py --config configs/llama2_7b_lora_event_local.yaml
