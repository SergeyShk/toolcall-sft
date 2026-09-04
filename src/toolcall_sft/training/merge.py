"""Merge a trained LoRA adapter into its base model for standalone serving."""

from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

__all__ = ["merge_adapter"]


def merge_adapter(*, adapter_dir: Path, output_dir: Path) -> Path:
    model = AutoPeftModelForCausalLM.from_pretrained(str(adapter_dir), dtype=torch.bfloat16)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
