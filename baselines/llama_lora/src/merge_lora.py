from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model for standalone inference/deployment.")
    parser.add_argument("--base_model", default="${LLAMA2_7B_PATH}")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    base_model = os.path.expandvars(os.path.expanduser(args.base_model))
    adapter_path = os.path.expandvars(os.path.expanduser(args.adapter_path))
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer_path = adapter_path if Path(adapter_path).exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=Path(tokenizer_path).exists())
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=False,
        local_files_only=Path(base_model).exists(),
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
