"""Chat-template-exact tokenization with assistant-only loss masks.

Training text must match byte-for-byte what the inference server renders for
the same messages, so serialization always goes through the tokenizer's own
chat template. Assistant spans are recovered with sentinel probes: for each
assistant message the conversation is re-rendered with that message replaced
by a sentinel-content message, and the region where the probe render diverges
from the true render is that message's trained span — content (including any
serialized tool calls) plus the turn terminator. Probing the full conversation
stays correct for templates whose rendering depends on message position:
Qwen3, for example, inserts an empty think block only for the final assistant
turn; the inserted block lands in the masked prompt, so serve the tuned model
in the matching non-thinking mode (``enable_thinking=false``).

A template that does not render assistant content verbatim (so the sentinel
cannot be located) fails loudly instead of producing silently misaligned
masks.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from transformers import PreTrainedTokenizerBase

from .schema import Dialogue, Role, chat_messages

__all__ = ["LABEL_IGNORE_INDEX", "MaskedExample", "TemplateCompatibilityError", "tokenize_dialogue"]

LABEL_IGNORE_INDEX = -100

_SENTINEL_BRACKET = "␀"


class TemplateCompatibilityError(Exception):
    """The tokenizer's chat template cannot express assistant-span probes."""


@dataclass(frozen=True, slots=True, kw_only=True)
class MaskedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


def tokenize_dialogue(tokenizer: PreTrainedTokenizerBase, dialogue: Dialogue) -> MaskedExample:
    if not tokenizer.is_fast:
        raise TemplateCompatibilityError("a fast tokenizer is required for offset-based loss masking")
    messages = chat_messages(dialogue)
    tools = [dict(tool) for tool in dialogue.tools] or None
    full_text = _render(tokenizer, messages, tools)
    terminator = _turn_terminator(tokenizer)
    spans: list[tuple[int, int]] = []
    for index, message in enumerate(dialogue.messages):
        if message.role is not Role.ASSISTANT:
            continue
        sentinel = f"{_SENTINEL_BRACKET}tcsft:{index}{_SENTINEL_BRACKET}"
        probe_messages = [*messages[:index], {"role": "assistant", "content": sentinel}, *messages[index + 1 :]]
        probe_text = _render(tokenizer, probe_messages, tools)
        spans.append(_span_from_probe(dialogue.dialogue_id, index, full_text, probe_text, sentinel, terminator))
    encoding = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    raw_input_ids = encoding["input_ids"]
    raw_offsets = encoding["offset_mapping"]
    if not isinstance(raw_input_ids, list) or not isinstance(raw_offsets, list):
        raise TemplateCompatibilityError("unexpected tokenizer output shape for a single text")
    input_ids: list[int] = raw_input_ids
    offsets: list[tuple[int, int]] = raw_offsets
    labels = [
        token if _starts_in_spans(start, spans) else LABEL_IGNORE_INDEX
        for token, (start, _end) in zip(input_ids, offsets, strict=True)
    ]
    if all(label == LABEL_IGNORE_INDEX for label in labels):
        raise TemplateCompatibilityError(f"dialogue {dialogue.dialogue_id!r}: no assistant tokens to train on")
    return MaskedExample(input_ids=tuple(input_ids), labels=tuple(labels))


def _render(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> str:
    # transformers annotates the conversation as list[dict[str, str]], which rules out valid
    # tool_calls entries and tool schemas; the casts widen to what the template accepts at runtime.
    rendered = tokenizer.apply_chat_template(
        cast("list[Any]", [dict(message) for message in messages]),
        tools=cast("list[Any] | None", tools),
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise TemplateCompatibilityError("chat template did not render to text")
    return rendered


def _turn_terminator(tokenizer: PreTrainedTokenizerBase) -> str:
    """Text the template emits after a final assistant message's content (e.g. ``<|im_end|>\\n``)."""
    sentinel = f"{_SENTINEL_BRACKET}tcsft:terminator{_SENTINEL_BRACKET}"
    rendered = _render(tokenizer, [{"role": "user", "content": "x"}, {"role": "assistant", "content": sentinel}], None)
    position = rendered.find(sentinel)
    if position < 0:
        raise TemplateCompatibilityError("chat template does not render assistant content verbatim")
    return rendered[position + len(sentinel) :]


def _span_from_probe(
    dialogue_id: str,
    index: int,
    base: str,
    probe: str,
    sentinel: str,
    terminator: str,
) -> tuple[int, int]:
    limit = min(len(base), len(probe))
    start = 0
    while start < limit and base[start] == probe[start]:
        start += 1
    tail = 0
    max_tail = limit - start
    while tail < max_tail and base[len(base) - 1 - tail] == probe[len(probe) - 1 - tail]:
        tail += 1
    if sentinel not in probe[start : len(probe) - tail]:
        raise TemplateCompatibilityError(
            f"dialogue {dialogue_id!r}, message {index}: chat template does not render assistant content verbatim"
        )
    end = len(base) - tail
    if terminator:
        if not base.startswith(terminator, end):
            raise TemplateCompatibilityError(
                f"dialogue {dialogue_id!r}, message {index}: expected the turn terminator after the assistant span"
            )
        end += len(terminator)
    if start >= end:
        raise TemplateCompatibilityError(f"dialogue {dialogue_id!r}, message {index}: assistant span is empty")
    return start, end


def _starts_in_spans(position: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)
