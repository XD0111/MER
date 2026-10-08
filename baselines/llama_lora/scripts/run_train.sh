#!/usr/bin/env bash
set -euo pipefail
accelerate launch src/train_lora_sft.py --config configs/llama2_7b_lora.yaml
