from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

from prompt_utils import build_alpaca_prompt, strip_eos
try:
    from event_relation_prompt import parse_candidate_label
except Exception:
    parse_candidate_label = None


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def expand_path(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(str(path)))


def load_model(base_model: str, adapter_path: str, load_in_4bit: bool = True):
    base_model = expand_path(base_model)
    adapter_path = expand_path(adapter_path)
    is_local_model = Path(base_model).exists()
    tokenizer_path = adapter_path if Path(adapter_path).exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=Path(tokenizer_path).exists())
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
        local_files_only=is_local_model,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)) :]
    return strip_eos(response, eos_token=tokenizer.eos_token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with base Llama-2 model + LoRA adapter.")
    parser.add_argument("--base_model", default="${LLAMA2_7B_PATH}", help="Local Llama2 path or HF repo id.")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--input", default="")
    parser.add_argument("--input_file", default=None, help="JSONL file with instruction/input or prompt fields.")
    parser.add_argument("--output_file", default="outputs/predictions.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model, args.adapter_path, load_in_4bit=not args.no_4bit)

    if args.input_file:
        results: List[Dict[str, Any]] = []
        for row in tqdm(list(read_jsonl(args.input_file)), desc="Generating"):
            if "prompt" in row:
                prompt = row["prompt"]
            else:
                prompt = build_alpaca_prompt(row.get("instruction") or row.get("question") or "", row.get("input") or "")
            pred = generate_one(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p)
            row["prediction"] = pred
            if parse_candidate_label is not None and row.get("candidates"):
                row["parsed_prediction"] = parse_candidate_label(pred, row["candidates"])
                row["is_correct"] = row["parsed_prediction"] == row.get("gold_label")
            results.append(row)
        write_jsonl(args.output_file, results)
        print(f"Saved predictions to {args.output_file}")
    else:
        if not args.instruction:
            raise ValueError("Please provide --instruction or --input_file")
        prompt = build_alpaca_prompt(args.instruction, args.input)
        print(generate_one(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p))


if __name__ == "__main__":
    main()
