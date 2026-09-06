"""Merge a trained LoRA adapter into its base model for standalone serving."""

from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

__all__ = ["merge_adapter"]


def merge_adapter(*, adapter_dir: Path, output_dir: Path) -> Path:
    # Merge in fp32 and round once at the end. The adapter delta is small next to the
    # base weight; computing W + BA in bf16 rounds the delta before the sum and loses part
    # of the update that training kept in fp32 for exactly this reason.
    model = AutoPeftModelForCausalLM.from_pretrained(str(adapter_dir), dtype=torch.float32)
    merged = model.merge_and_unload()
    merged = merged.to(torch.bfloat16)
    merged.save_pretrained(str(output_dir))
    # The tokenizer saved next to the adapter carries the chat template training rendered with.
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
