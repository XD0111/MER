from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from event_relation_prompt import (
    add_few_shot_examples,
    get_mention_schema,
    load_json_or_jsonl,
    parse_candidate_label,
    trans_llm,
    write_jsonl,
)
from prompt_utils import strip_eos


def expand_path(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(path))


def load_model(base_model: str, adapter_path: Optional[str] = None, load_in_4bit: bool = True):
    base_model = expand_path(base_model)
    adapter_path = expand_path(adapter_path) if adapter_path else None
    tokenizer_path = adapter_path if adapter_path and Path(adapter_path).exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=Path(base_model).exists())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=False,
        local_files_only=Path(base_model).exists(),
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    do_sample = temperature > 0
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    return strip_eos(decoded[len(prompt_text):], eos_token=tokenizer.eos_token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot / zero-shot event relation inference with local Llama2, optionally with LoRA adapter.")
    parser.add_argument("--base_model", default="${LLAMA2_7B_PATH}", help="Local Llama2 path or HF repo id.")
    parser.add_argument("--adapter_path", default=None, help="Optional LoRA adapter path.")
    parser.add_argument("--train_path", required=True, help="Raw train JSON/JSONL used as few-shot pool.")
    parser.add_argument("--test_path", required=True, help="Raw test JSON/JSONL to evaluate.")
    parser.add_argument("--output_file", default="outputs/event_fewshot_predictions.jsonl")
    parser.add_argument("--k", type=int, default=4, help="Number of few-shot examples. Use 0 for zero-shot.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    train_raw = get_mention_schema(load_json_or_jsonl(args.train_path), seed=args.seed)
    test_raw = get_mention_schema(load_json_or_jsonl(args.test_path), seed=args.seed + 1)
    train_rows = trans_llm(train_raw, eos_token="", single=bool(args.single), include_meta=True)
    test_rows = trans_llm(test_raw, eos_token="", single=bool(args.single), include_meta=True)
    test_rows = add_few_shot_examples(test_rows, train_rows, args.k, seed=args.seed) if args.k > 0 else test_rows

    model, tokenizer = load_model(args.base_model, args.adapter_path, load_in_4bit=not args.no_4bit)

    results: List[Dict[str, Any]] = []
    for row in tqdm(test_rows, desc="Generating"):
        pred = generate_one(model, tokenizer, row["prompt"], args.max_new_tokens, args.temperature, args.top_p)
        parsed = parse_candidate_label(pred, row.get("candidates", []))
        out = dict(row)
        out["prediction"] = pred
        out["parsed_prediction"] = parsed
        out["is_correct"] = parsed == row.get("gold_label")
        results.append(out)

    write_jsonl(args.output_file, results)
    acc = sum(int(r["is_correct"]) for r in results) / max(len(results), 1)
    print(json.dumps({"n": len(results), "accuracy": acc, "output_file": args.output_file}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
