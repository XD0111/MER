#!/usr/bin/env bash
set -euo pipefail

# Replace these paths with your original project's train/valid/test JSON files.
TRAIN_PATH=${TRAIN_PATH:-data/raw/train.json}
VALID_PATH=${VALID_PATH:-data/raw/valid.json}
TEST_PATH=${TEST_PATH:-data/raw/test.json}

python src/prepare_event_relation_data.py \
  --train_path "$TRAIN_PATH" \
  --valid_path "$VALID_PATH" \
  --test_path "$TEST_PATH" \
  --output_dir data/processed_event \
  --single 1 \
  --seed 42
