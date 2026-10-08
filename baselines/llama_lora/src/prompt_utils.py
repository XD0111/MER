from __future__ import annotations


def build_alpaca_prompt(instruction: str, input_text: str | None = None) -> str:
    """Build a stable instruction prompt for base Llama-2 style SFT.

    This deliberately avoids relying on tokenizer.chat_template, because
    meta-llama/Llama-2-7b-hf is a base model rather than a chat model.
    """
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()

    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task.\n"
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )


def strip_eos(text: str, eos_token: str = "</s>") -> str:
    text = text or ""
    if eos_token and text.endswith(eos_token):
        text = text[: -len(eos_token)]
    return text.strip()
