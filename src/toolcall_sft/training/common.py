"""What training and prediction share: device and dtype resolution, per-turn tokenization
under the dataset's token budget, and the provenance both write next to their output."""

import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

import peft
import torch
import transformers
from transformers import PreTrainedTokenizerBase

from ..config import ConfigError
from ..masking import MaskedExample, tokenize_dialogue
from ..schema import DatasetError, Dialogue

__all__ = [
    "ensure_pad_token",
    "git_state",
    "resolve_device",
    "resolve_dtype",
    "tokenize_dialogues",
    "versions",
]


def resolve_device(requested: str) -> str:
    """The requested device, or the best available for ``auto``. A missing device is an error."""
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if requested == "cuda" and not cuda:
        raise ConfigError("training.device is 'cuda' but no CUDA device is available; use 'auto' or another device")
    if requested == "mps" and not mps:
        raise ConfigError("training.device is 'mps' but the MPS backend is not available; use 'auto' or 'cpu'")
    return requested


def ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> None:
    """Pad with the end-of-turn token when the tokenizer has no pad token of its own."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def versions() -> dict[str, str]:
    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}


def git_state() -> dict[str, Any]:
    """The commit the run was produced at, and whether the tree was dirty."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit.strip(), "dirty": bool(status.strip())}


def resolve_dtype(precision: str, device: str) -> torch.dtype:
    if precision == "auto":
        # CPU bf16 matmuls fall back to slow kernels.
        precision = "fp32" if device == "cpu" else "bf16"
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def tokenize_dialogues(
    tokenizer: PreTrainedTokenizerBase,
    dialogues: Sequence[Dialogue],
    *,
    max_seq_length: int,
    chat_template_kwargs: Mapping[str, Any],
    name: str,
) -> tuple[tuple[Dialogue, tuple[MaskedExample, ...]], ...]:
    """Per-turn examples for every dialogue, kept with the dialogue they came from.

    A dialogue over ``max_seq_length`` is an error, not a drop: run ``tcsft filter``
    so that what trains (and what is scored) is what you reviewed.
    """
    tokenized: list[tuple[Dialogue, tuple[MaskedExample, ...]]] = []
    too_long: list[tuple[str, int]] = []
    for dialogue in dialogues:
        turns = tokenize_dialogue(tokenizer, dialogue, chat_template_kwargs=chat_template_kwargs)
        longest = max(len(example.input_ids) for example in turns)
        if longest > max_seq_length:
            too_long.append((dialogue.dialogue_id, longest))
            continue
        tokenized.append((dialogue, turns))
    if too_long:
        shown = ", ".join(f"{dialogue_id} ({tokens} tokens)" for dialogue_id, tokens in too_long[:5])
        more = f" and {len(too_long) - 5} more" if len(too_long) > 5 else ""
        raise DatasetError(
            f"{name}: {len(too_long)} of {len(dialogues)} dialogues exceed max_seq_length={max_seq_length}: "
            f"{shown}{more}. Run `tcsft filter --config <this config>` or raise dataset.max_seq_length."
        )
    return tuple(tokenized)
