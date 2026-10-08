from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_dtype(name: str | None) -> torch.dtype:
    name = (name or "bfloat16").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def file_exists(path: Optional[str]) -> bool:
    if not path:
        return False
    path = os.path.expandvars(os.path.expanduser(str(path)))
    return Path(path).exists() and Path(path).stat().st_size > 0


def expand_path(path: str) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    if "$" in expanded:
        raise ValueError(
            f"Unresolved environment variable in path: {path}. "
            "Set it first, e.g. export LLAMA2_7B_PATH=/path/to/Llama-2-7b-hf"
        )
    return expanded


def build_bnb_config(cfg: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    if not cfg.get("use_qlora", True):
        return None
    if cfg.get("load_in_4bit", True):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=as_dtype(cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT for Llama-2-7B.")
    parser.add_argument("--config", default="configs/llama2_7b_lora.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)  # for torchrun/accelerate compatibility
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)

    if cfg.get("tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    model_name = expand_path(cfg["model_name_or_path"])
    train_file = expand_path(cfg["train_file"])
    eval_file = expand_path(cfg.get("eval_file")) if cfg.get("eval_file") else None

    data_files = {"train": train_file}
    if file_exists(eval_file):
        data_files["validation"] = eval_file
    dataset = load_dataset("json", data_files=data_files)

    is_local_model = Path(model_name).exists()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=is_local_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = build_bnb_config(cfg)
    torch_dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=False,
        local_files_only=is_local_model,
    )
    model.config.use_cache = False

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        )

    peft_config = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias=cfg.get("lora_bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=cfg.get("target_modules"),
    )

    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        max_length=int(cfg.get("max_seq_length", 2048)),
        packing=bool(cfg.get("packing", False)),
        completion_only_loss=True,
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 16)),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.3)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        bf16=bool(cfg.get("bf16", True)),
        fp16=bool(cfg.get("fp16", False)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        eval_strategy=cfg.get("eval_strategy", "steps") if "validation" in dataset else "no",
        eval_steps=int(cfg.get("eval_steps", 100)),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=int(cfg.get("save_steps", 100)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        report_to=cfg.get("report_to", "none"),
        seed=seed,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    resume_from_checkpoint = cfg.get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    print(f"Training finished. LoRA adapter saved to: {cfg['output_dir']}")


if __name__ == "__main__":
    main()
