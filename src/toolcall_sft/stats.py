"""Token-length statistics under a base model's tokenizer, via the same per-turn
tokenization as training.

Per dialogue: ``longest_example`` (what must fit max_seq_length), ``epoch_tokens``
(all examples added up; prefixes repeat) and ``trained_tokens`` (target tokens).
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

# The example scenario's longest examples are ~730-920 tokens; 2048 leaves room for
# longer branches while staying trainable on a laptop.
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
    """Split dialogues by whether their longest example fits ``max_seq_length``.

    Dialogues that fail to tokenize go to ``failures``; they cannot train either.
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
