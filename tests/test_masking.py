"""Masking is tested against small purpose-built tokenizers so the suite runs offline in
seconds. The Qwen3 tests use the real Qwen3 chat template (tests/templates/qwen3.jinja,
Apache-2.0, copied verbatim from the Qwen/Qwen3-0.6B tokenizer) on a character-level
tokenizer: what matters is the text the template produces, not how a BPE splits it."""

from pathlib import Path
from typing import Any, cast

import pytest
from tokenizers import Regex, Tokenizer, decoders
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split, WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from toolcall_sft import (
    LABEL_IGNORE_INDEX,
    Dialogue,
    MaskedExample,
    Message,
    Role,
    TemplateCompatibilityError,
    ToolCall,
    chat_messages,
    render_example,
    tokenize_dialogue,
)

QWEN3_TEMPLATE = (Path(__file__).parent / "templates" / "qwen3.jinja").read_text(encoding="utf-8")

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message.role }}|>{{ message.content }}<|end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)

# Rendering depends on global conversation state (message count): the prompt the server builds
# for turn k and the same turn rendered as history can never agree, so no training text exists.
COUNTING_TEMPLATE = "{{ messages | length }}:" + CHAT_TEMPLATE

# Transforms assistant content, so the target cannot contain it verbatim.
NONVERBATIM_TEMPLATE = (
    "{% for message in messages %}<|{{ message.role }}|>{{ message.content | upper }}<|end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)

TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "get_payees",
            "description": "Look up saved payees by name.",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        },
    },
)


def _char_tokenizer(template: str) -> PreTrainedTokenizerFast:
    vocab = {chr(code): index for index, code in enumerate(range(32, 127))}
    vocab["\n"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Split(Regex(r"[\s\S]"), behavior="isolated")
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = template
    return tokenizer


def _dialogue() -> Dialogue:
    return Dialogue(
        dialogue_id="d1",
        messages=(
            Message(role=Role.SYSTEM, content="You are a payment assistant."),
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there"),
            Message(role=Role.USER, content="Pay rent"),
            Message(role=Role.ASSISTANT, content="Sure"),
        ),
    )


def _tool_dialogue() -> Dialogue:
    """Four assistant turns in the shape of the example scenario: call, ask, call, report."""
    return Dialogue(
        dialogue_id="d2",
        messages=(
            Message(role=Role.SYSTEM, content="You are a payment assistant."),
            Message(role=Role.USER, content="Send $250 to Noah Strand"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=(ToolCall(name="get_payees", arguments={"name": "Noah Strand"}),),
            ),
            Message(role=Role.TOOL, content='[{"payee_id": "pay_1", "name": "Noah Strand"}]'),
            Message(role=Role.ASSISTANT, content="Noah Strand - send $250?"),
            Message(role=Role.USER, content="Yes"),
            Message(
                role=Role.ASSISTANT, content="", tool_calls=(ToolCall(name="get_payees", arguments={"name": "again"}),)
            ),
            Message(role=Role.TOOL, content='{"status": "sent"}'),
            Message(role=Role.ASSISTANT, content="Done."),
        ),
        tools=TOOLS,
    )


def _prompt_and_target(tokenizer: PreTrainedTokenizerFast, example: MaskedExample) -> tuple[str, str]:
    prompt = tokenizer.decode(list(example.input_ids[: example.prompt_tokens]))
    target = tokenizer.decode(list(example.input_ids[example.prompt_tokens :]))
    return prompt, target


def test_one_example_per_assistant_turn_trains_exactly_on_that_turn() -> None:
    tokenizer = _char_tokenizer(CHAT_TEMPLATE)

    examples = tokenize_dialogue(tokenizer, _dialogue())

    assert [example.turn_index for example in examples] == [2, 4]
    assert [_prompt_and_target(tokenizer, example)[1] for example in examples] == ["Hi there<|end|>\n", "Sure<|end|>\n"]


def test_prompt_is_the_generation_prompt_the_server_would_build() -> None:
    tokenizer = _char_tokenizer(CHAT_TEMPLATE)
    dialogue = _dialogue()
    messages = chat_messages(dialogue)

    examples = tokenize_dialogue(tokenizer, dialogue)

    for example in examples:
        prompt, _target = _prompt_and_target(tokenizer, example)
        expected = tokenizer.apply_chat_template(
            messages[: example.turn_index], tokenize=False, add_generation_prompt=True
        )
        assert prompt == expected


def test_labels_are_ignore_index_on_the_prompt_and_the_token_ids_on_the_target() -> None:
    tokenizer = _char_tokenizer(CHAT_TEMPLATE)

    example = tokenize_dialogue(tokenizer, _dialogue())[0]

    assert len(example.input_ids) == len(example.labels)
    assert set(example.labels[: example.prompt_tokens]) == {LABEL_IGNORE_INDEX}
    assert example.labels[example.prompt_tokens :] == example.input_ids[example.prompt_tokens :]
    assert "You are a payment assistant." in tokenizer.decode(list(example.input_ids[: example.prompt_tokens]))


def test_render_example_marks_the_target() -> None:
    tokenizer = _char_tokenizer(CHAT_TEMPLATE)

    example = tokenize_dialogue(tokenizer, _dialogue())[1]

    assert render_example(tokenizer, example).endswith("<|assistant|>⟦Sure<|end|>\n⟧")


def test_qwen3_every_turn_is_prompted_with_the_empty_think_block() -> None:
    """The regression the per-turn design exists for: a whole-dialogue render gives only the last
    assistant turn the think block, while the server prefixes every turn with it."""
    tokenizer = _char_tokenizer(QWEN3_TEMPLATE)

    examples = tokenize_dialogue(tokenizer, _tool_dialogue(), chat_template_kwargs={"enable_thinking": False})

    assert len(examples) == 4
    for example in examples:
        prompt, _target = _prompt_and_target(tokenizer, example)
        assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_qwen3_targets_are_the_tool_call_or_the_reply_plus_terminator() -> None:
    tokenizer = _char_tokenizer(QWEN3_TEMPLATE)

    examples = tokenize_dialogue(tokenizer, _tool_dialogue(), chat_template_kwargs={"enable_thinking": False})

    targets = [_prompt_and_target(tokenizer, example)[1] for example in examples]
    assert (
        targets[0]
        == '<tool_call>\n{"name": "get_payees", "arguments": {"name": "Noah Strand"}}\n</tool_call><|im_end|>\n'
    )
    assert targets[1] == "Noah Strand - send $250?<|im_end|>\n"
    assert targets[2] == '<tool_call>\n{"name": "get_payees", "arguments": {"name": "again"}}\n</tool_call><|im_end|>\n'
    assert targets[3] == "Done.<|im_end|>\n"


def test_qwen3_prompt_matches_the_server_side_render_byte_for_byte() -> None:
    tokenizer = _char_tokenizer(QWEN3_TEMPLATE)
    dialogue = _tool_dialogue()
    messages = chat_messages(dialogue)
    tools = [dict(tool) for tool in dialogue.tools]

    examples = tokenize_dialogue(tokenizer, dialogue, chat_template_kwargs={"enable_thinking": False})

    for example in examples:
        prompt, _target = _prompt_and_target(tokenizer, example)
        expected = tokenizer.apply_chat_template(
            messages[: example.turn_index],
            tools=cast("list[Any]", tools),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert prompt == expected


def test_qwen3_history_carries_no_think_blocks_only_the_generation_prompt_does() -> None:
    tokenizer = _char_tokenizer(QWEN3_TEMPLATE)

    last = tokenize_dialogue(tokenizer, _tool_dialogue(), chat_template_kwargs={"enable_thinking": False})[-1]

    prompt, _target = _prompt_and_target(tokenizer, last)
    assert prompt.count("<think>") == 1
    assert prompt.count("<|im_start|>assistant") == 4


def test_qwen3_tools_are_rendered_into_the_prompt() -> None:
    tokenizer = _char_tokenizer(QWEN3_TEMPLATE)

    first = tokenize_dialogue(tokenizer, _tool_dialogue(), chat_template_kwargs={"enable_thinking": False})[0]

    prompt, _target = _prompt_and_target(tokenizer, first)
    assert "<tools>" in prompt
    assert '"name": "get_payees"' in prompt
    assert "Look up saved payees by name." in prompt


def test_qwen3_without_the_kwarg_the_model_would_have_to_emit_the_think_block_itself() -> None:
    """The other self-consistent regime: no enable_thinking flag at serving, empty block in the
    target. Whichever is chosen, training and serving must pick the same one."""
    tokenizer = _char_tokenizer(QWEN3_TEMPLATE)

    first = tokenize_dialogue(tokenizer, _tool_dialogue(), chat_template_kwargs={})[0]

    prompt, target = _prompt_and_target(tokenizer, first)
    assert prompt.endswith("<|im_start|>assistant\n")
    assert target.startswith("<think>\n\n</think>\n\n<tool_call>")


def test_state_dependent_template_fails_loudly_instead_of_training_on_an_impossible_prompt() -> None:
    tokenizer = _char_tokenizer(COUNTING_TEMPLATE)

    with pytest.raises(TemplateCompatibilityError, match="does not extend the generation prompt"):
        tokenize_dialogue(tokenizer, _dialogue())


def test_non_verbatim_template_fails_loudly() -> None:
    tokenizer = _char_tokenizer(NONVERBATIM_TEMPLATE)

    with pytest.raises(TemplateCompatibilityError, match="verbatim"):
        tokenize_dialogue(tokenizer, _dialogue())


def test_boundary_that_tokenizes_differently_in_context_fails_loudly() -> None:
    """A whitespace-split tokenizer glues '<|assistant|>' to the first word of the reply, so the
    prompt alone and the prompt inside the example tokenize differently — the model would be
    trained on a token the server never feeds it."""
    words = ["<|system|>You", "are", "a", "payment", "assistant.<|end|>", "<|user|>Hello<|end|>", "<|assistant|>"]
    words += ["<|assistant|>Hi", "there<|end|>", "<|user|>Pay", "rent<|end|>", "<|assistant|>Sure<|end|>"]
    vocab = {word: index for index, word in enumerate(words)}
    vocab["[UNK]"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = CHAT_TEMPLATE

    with pytest.raises(TemplateCompatibilityError, match="boundary"):
        tokenize_dialogue(tokenizer, _dialogue())


def test_dialogue_without_assistant_turn_fails_loudly() -> None:
    tokenizer = _char_tokenizer(CHAT_TEMPLATE)
    dialogue = Dialogue(dialogue_id="d3", messages=(Message(role=Role.USER, content="Hello"),))

    with pytest.raises(TemplateCompatibilityError, match="no assistant turn"):
        tokenize_dialogue(tokenizer, dialogue)
