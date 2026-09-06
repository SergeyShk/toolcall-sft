"""Token-length statistics for a dataset under a specific base model's tokenizer.

Answers the sizing questions before a training run: how long the examples are in
the base model's tokens, what share of them actually carries loss, and how many
dialogues survive a given ``max_seq_length``. Uses the same per-turn tokenization
path as training, so the numbers match what the trainer will see.

Three sizes matter per dialogue and they are different numbers:

- ``longest_example`` — the last assistant turn's example, which carries the whole
  conversation as its prompt. This is what has to fit ``max_seq_length``.
- ``epoch_tokens`` — every example's length added up. Prefixes repeat across a
  dialogue's turns, so this is what a pass over the data costs.
- ``trained_tokens`` — target tokens across all examples. This is the signal.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .masking import TemplateCompatibilityError, tokenize_dialogue
from .schema import Dialogue

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = [
    "DEFAULT_MAX_SEQ_LENGTH",
    "DEFAULT_THRESHOLDS",
    "HISTOGRAM_EDGES",
    "DialogueTokenCount",
    "LengthFilterResult",
    "TokenStatsReport",
    "compute_token_stats",
    "filter_by_length",
    "histogram",
    "percentile",
    "threshold_fits",
]

# The example scenario's longest examples land around 500-900 tokens: a ~150-token
# system prompt, ~250 tokens of tool schemas, the rest conversation. 2048 leaves room
# for longer branches while keeping activations small enough to train on a laptop.
# Retarget this together with dataset.max_seq_length when you bring your own data.
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_THRESHOLDS = (1024, DEFAULT_MAX_SEQ_LENGTH, 4096, 8192)
HISTOGRAM_EDGES = (256, 512, 1024, 2048, 4096)


@dataclass(frozen=True, slots=True, kw_only=True)
class DialogueTokenCount:
    dialogue_id: str
    examples: int
    longest_example: int
    epoch_tokens: int
    trained_tokens: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenStatsReport:
    counts: tuple[DialogueTokenCount, ...]
    failures: tuple[tuple[str, str], ...]


def _count_tokens(
    tokenizer: "PreTrainedTokenizerBase",
    dialogue: Dialogue,
    chat_template_kwargs: Mapping[str, Any] | None,
) -> DialogueTokenCount:
    """Token accounting for one dialogue; raises TemplateCompatibilityError like training would."""
    examples = tokenize_dialogue(tokenizer, dialogue, chat_template_kwargs=chat_template_kwargs)
    return DialogueTokenCount(
        dialogue_id=dialogue.dialogue_id,
        examples=len(examples),
        longest_example=max(len(example.input_ids) for example in examples),
        epoch_tokens=sum(len(example.input_ids) for example in examples),
        trained_tokens=sum(example.target_tokens for example in examples),
    )


def compute_token_stats(
    tokenizer: "PreTrainedTokenizerBase",
    dialogues: Sequence[Dialogue],
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> TokenStatsReport:
    counts: list[DialogueTokenCount] = []
    failures: list[tuple[str, str]] = []
    for dialogue in dialogues:
        try:
            counts.append(_count_tokens(tokenizer, dialogue, chat_template_kwargs))
        except TemplateCompatibilityError as error:
            failures.append((dialogue.dialogue_id, str(error)))
    return TokenStatsReport(counts=tuple(counts), failures=tuple(failures))


@dataclass(frozen=True, slots=True, kw_only=True)
class LengthFilterResult:
    kept: tuple[Dialogue, ...]
    dropped: tuple[DialogueTokenCount, ...]
    failures: tuple[tuple[str, str], ...]


def filter_by_length(
    tokenizer: "PreTrainedTokenizerBase",
    dialogues: Sequence[Dialogue],
    *,
    max_seq_length: int,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> LengthFilterResult:
    """Partition dialogues by whether their longest example fits ``max_seq_length``.

    Uses the same tokenization-with-masks path as training, so a kept dialogue is
    exactly one the trainer will accept. Dialogues that fail to tokenize land in
    ``failures`` — they cannot train either.
    """
    if max_seq_length < 1:
        raise ValueError(f"max_seq_length must be >= 1, got {max_seq_length}")
    kept: list[Dialogue] = []
    dropped: list[DialogueTokenCount] = []
    failures: list[tuple[str, str]] = []
    for dialogue in dialogues:
        try:
            count = _count_tokens(tokenizer, dialogue, chat_template_kwargs)
        except TemplateCompatibilityError as error:
            failures.append((dialogue.dialogue_id, str(error)))
            continue
        if count.longest_example <= max_seq_length:
            kept.append(dialogue)
        else:
            dropped.append(count)
    return LengthFilterResult(kept=tuple(kept), dropped=tuple(dropped), failures=tuple(failures))


def percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def threshold_fits(values: Sequence[int], thresholds: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """For each threshold: how many values fit under it."""
    return tuple((threshold, sum(1 for value in values if value <= threshold)) for threshold in thresholds)


def histogram(values: Sequence[int], edges: Sequence[int] = HISTOGRAM_EDGES) -> tuple[tuple[str, int], ...]:
    """Bucket counts with human-readable labels; the last bucket is open-ended."""
    bounds = [0, *edges]
    buckets: list[tuple[str, int]] = []
    for lower, upper in zip(bounds, bounds[1:], strict=False):
        label = f"{lower}-{upper - 1}"
        buckets.append((label, sum(1 for value in values if lower <= value < upper)))
    top = bounds[-1]
    buckets.append((f"{top}+", sum(1 for value in values if value >= top)))
    return tuple(buckets)
