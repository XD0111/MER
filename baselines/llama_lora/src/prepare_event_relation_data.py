from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from event_relation_prompt import (
    add_few_shot_examples,
    get_mention_schema,
    load_json_or_jsonl,
    trans_llm,
    write_jsonl,
)


def convert_split(path: str, seed: int, eos_token: str, single: bool):
    raw = load_json_or_jsonl(path)
    with_schema = get_mention_schema(raw, seed=seed)
    return trans_llm(with_schema, eos_token=eos_token, single=single, include_meta=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert event-relation graph data into TRL prompt/completion JSONL for Llama2 LoRA SFT."
    )
    parser.add_argument("--train_path", required=True, help="Raw train JSON/JSONL path used by the original project.")
    parser.add_argument("--valid_path", required=True, help="Raw valid/dev JSON/JSONL path used by the original project.")
    parser.add_argument("--test_path", required=True, help="Raw test JSON/JSONL path used by the original project.")
    parser.add_argument("--output_dir", default="data/processed_event", help="Output directory for train/valid/test JSONL.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eos_token", default="</s>")
    parser.add_argument("--single", type=int, default=1, help="1 means ask the model to answer with a single word/phrase.")
    parser.add_argument(
        "--few_shot_k",
        type=int,
        default=0,
        help="Optional: prepend k train demonstrations to valid/test prompts. Keep 0 for normal LoRA fine-tuning.",
    )
    args = parser.parse_args()

    single = bool(args.single)
    train_rows = convert_split(args.train_path, args.seed, args.eos_token, single)
    valid_rows = convert_split(args.valid_path, args.seed + 1, args.eos_token, single)
    test_rows = convert_split(args.test_path, args.seed + 2, args.eos_token, single)

    if args.few_shot_k > 0:
        valid_rows = add_few_shot_examples(valid_rows, train_rows, args.few_shot_k, seed=args.seed)
        test_rows = add_few_shot_examples(test_rows, train_rows, args.few_shot_k, seed=args.seed)

    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    print(f"Saved event-relation SFT data to {out_dir}")
    print(f"train={len(train_rows)}, valid={len(valid_rows)}, test={len(test_rows)}")
    print("Note: each original event-pair sample is expanded into 4 relation-classification instruction samples.")


if __name__ == "__main__":
    main()
