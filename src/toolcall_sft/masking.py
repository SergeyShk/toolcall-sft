"""Chat-template-exact tokenization with loss on assistant tokens only.

One example per assistant turn. The prompt is the conversation so far, rendered
with ``add_generation_prompt=True`` and the template kwargs the server uses; the
target is what the template appends for the turn itself (content, tool calls,
terminator).

Per turn rather than per dialogue because templates render history and the
generation prompt differently. Qwen3 in non-thinking mode puts an empty <think>
block before the generation prompt but keeps it only on the last assistant turn
of the history, so a whole-dialogue render trains every tool-call turn on a
prompt the server never builds. Repeating the prefix costs about 3x tokens per
epoch on the example scenario; the trained tokens are the same.

Raises TemplateCompatibilityError instead of misaligning silently: the rendered
turn must extend the prompt, content and tool names must appear verbatim in the
target, and the prompt's tokens must be a prefix of the example's tokens.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from .schema import Dialogue, Role, chat_messages

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = ["LABEL_IGNORE_INDEX", "MaskedExample", "TemplateCompatibilityError", "render_example", "tokenize_dialogue"]

LABEL_IGNORE_INDEX = -100

TARGET_MARKERS = ("⟦", "⟧")


class TemplateCompatibilityError(Exception):
    """The chat template cannot produce training text that agrees with inference."""


@dataclass(frozen=True, slots=True, kw_only=True)
class MaskedExample:
    """One assistant turn: prompt tokens (masked) followed by target tokens (loss)."""

    turn_index: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]

    @property
    def prompt_tokens(self) -> int:
        return sum(1 for label in self.labels if label == LABEL_IGNORE_INDEX)

    @property
    def target_tokens(self) -> int:
        return len(self.labels) - self.prompt_tokens


def tokenize_dialogue(
    tokenizer: "PreTrainedTokenizerBase",
    dialogue: Dialogue,
    *,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[MaskedExample, ...]:
    """One masked example per assistant turn, in dialogue order."""
    kwargs = dict(chat_template_kwargs or {})
    messages = chat_messages(dialogue)
    tools = [dict(tool) for tool in dialogue.tools] or None
    examples: list[MaskedExample] = []
    for index, message in enumerate(dialogue.messages):
        if message.role is not Role.ASSISTANT:
            continue
        where = f"dialogue {dialogue.dialogue_id!r}, message {index}"
        prompt = _render(tokenizer, messages[:index], tools, add_generation_prompt=True, kwargs=kwargs)
        with_turn = _render(tokenizer, messages[: index + 1], tools, add_generation_prompt=False, kwargs=kwargs)
        if not with_turn.startswith(prompt):
            raise TemplateCompatibilityError(
                f"{where}: rendering the assistant turn does not extend the generation prompt, so no "
                "training text can agree with inference for this chat template"
            )
        target = with_turn[len(prompt) :]
        if message.content and message.content not in target:
            raise TemplateCompatibilityError(f"{where}: chat template does not render assistant content verbatim")
        for call in message.tool_calls:
            if call.name not in target:
                raise TemplateCompatibilityError(f"{where}: chat template does not render tool call {call.name!r}")
        prompt_ids = _encode(tokenizer, prompt)
        input_ids = _encode(tokenizer, with_turn)
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise TemplateCompatibilityError(
                f"{where}: the prompt/target boundary tokenizes differently in isolation and in context; "
                "the template needs a token boundary (a special token or newline) before assistant content"
            )
        if len(input_ids) == len(prompt_ids):
            raise TemplateCompatibilityError(f"{where}: assistant turn rendered to an empty target")
        labels = [LABEL_IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids) :]
        examples.append(MaskedExample(turn_index=index, input_ids=tuple(input_ids), labels=tuple(labels)))
    if not examples:
        raise TemplateCompatibilityError(f"dialogue {dialogue.dialogue_id!r}: no assistant turn to train on")
    return tuple(examples)


def render_example(
    tokenizer: "PreTrainedTokenizerBase",
    example: MaskedExample,
    *,
    markers: tuple[str, str] = TARGET_MARKERS,
) -> str:
    """The example decoded from its token ids, with the tokens that carry loss wrapped in ``markers``."""
    open_marker, close_marker = markers
    prompt = tokenizer.decode(list(example.input_ids[: example.prompt_tokens]))
    target = tokenizer.decode(list(example.input_ids[example.prompt_tokens :]))
    return f"{prompt}{open_marker}{target}{close_marker}"


def _render(
    tokenizer: "PreTrainedTokenizerBase",
    messages: Sequence[Mapping[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    add_generation_prompt: bool,
    kwargs: Mapping[str, Any],
) -> str:
    # transformers types the conversation as list[dict[str, str]]; tool_calls and tool schemas are wider.
    rendered = tokenizer.apply_chat_template(
        cast("list[Any]", [dict(message) for message in messages]),
        tools=cast("list[Any] | None", tools),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    if not isinstance(rendered, str):
        raise TemplateCompatibilityError("chat template did not render to text")
    return rendered


def _encode(tokenizer: "PreTrainedTokenizerBase", text: str) -> list[int]:
    # The template already placed every special token.
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not isinstance(ids, list):
        raise TemplateCompatibilityError("unexpected tokenizer output shape for a single text")
    return cast("list[int]", ids)
