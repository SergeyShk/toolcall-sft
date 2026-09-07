"""Merge a trained LoRA adapter into its base model for standalone serving."""

from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

__all__ = ["merge_adapter"]


def merge_adapter(*, adapter_dir: Path, output_dir: Path) -> Path:
    # Merge in fp32 and round once: adding a small delta to a bf16 weight in bf16 loses part of it.
    model = AutoPeftModelForCausalLM.from_pretrained(str(adapter_dir), dtype=torch.float32)
    merged = model.merge_and_unload()
    merged = merged.to(torch.bfloat16)
    merged.save_pretrained(str(output_dir))
    # The adapter's tokenizer carries the chat template training rendered with.
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
