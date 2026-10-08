from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from prompt_utils import build_alpaca_prompt


INSTRUCTION_KEYS = ["instruction", "question", "query", "task"]
INPUT_KEYS = ["input", "context", "case_fact", "facts", "fact", "source"]
OUTPUT_KEYS = ["output", "answer", "response", "target", "label"]


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_existing(example: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for key in keys:
        if key in example and example[key] is not None:
            value = example[key]
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)
    return default


def normalize_one(example: Dict[str, Any], eos_token: str = "</s>") -> Dict[str, str]:
    """Convert common raw formats into TRL prompt-completion format.

    Supported raw formats:
    1. {"instruction": ..., "input": ..., "output": ...}
    2. {"question": ..., "answer": ...}
    3. Already normalized {"prompt": ..., "completion": ...}
    """
    if "prompt" in example and "completion" in example:
        prompt = str(example["prompt"]).strip()
        completion = str(example["completion"]).strip()
    else:
        instruction = first_existing(example, INSTRUCTION_KEYS)
        input_text = first_existing(example, INPUT_KEYS)
        completion = first_existing(example, OUTPUT_KEYS)
        if not instruction or not completion:
            raise ValueError(
                "Each row must contain instruction/output, question/answer, or prompt/completion. "
                f"Bad row keys: {list(example.keys())}"
            )
        prompt = build_alpaca_prompt(instruction, input_text)
        completion = completion.strip()

    if eos_token and not completion.endswith(eos_token):
        completion = completion + eos_token
    return {"prompt": prompt, "completion": completion}


def split_rows(
    rows: List[Dict[str, str]],
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)

    n = len(rows)
    n_test = int(round(n * test_ratio))
    n_valid = int(round(n * valid_ratio))

    # 小数据集保留至少一个训练样本，避免 demo 数据被切空。
    if n <= 5:
        n_valid = 0
        n_test = 0

    test_rows = rows[:n_test]
    valid_rows = rows[n_test : n_test + n_valid]
    train_rows = rows[n_test + n_valid :]
    return train_rows, valid_rows, test_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw instruction data into TRL prompt-completion JSONL.")
    parser.add_argument("--input", required=True, help="Raw JSONL file.")
    parser.add_argument("--output_dir", default="data/processed", help="Directory for train/valid/test JSONL files.")
    parser.add_argument("--valid_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eos_token", default="</s>", help="EOS token appended to each completion. Use '' to disable.")
    args = parser.parse_args()

    raw_rows = list(read_jsonl(args.input))
    normalized = [normalize_one(row, eos_token=args.eos_token) for row in raw_rows]
    train_rows, valid_rows, test_rows = split_rows(normalized, args.valid_ratio, args.test_ratio, args.seed)

    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    print(f"Loaded {len(raw_rows)} raw rows")
    print(f"Saved train={len(train_rows)}, valid={len(valid_rows)}, test={len(test_rows)} to {out_dir}")


if __name__ == "__main__":
    main()
