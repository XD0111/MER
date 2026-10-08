#!/usr/bin/env bash
set -euo pipefail
python src/prepare_data.py \
  --input data/raw_demo.jsonl \
  --output_dir data/processed \
  --valid_ratio 0.05 \
  --test_ratio 0.05
