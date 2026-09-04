import pytest
from tokenizers import Regex, Tokenizer, decoders
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

from toolcall_sft import (
    LABEL_IGNORE_INDEX,
    Dialogue,
    MaskedExample,
    Message,
    Role,
    TemplateCompatibilityError,
    tokenize_dialogue,
)

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message.role }}|>{{ message.content }}<|end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)

# Rendering depends on global conversation state (message count) — probes must still align.
COUNTING_TEMPLATE = "{{ messages | length }}:" + CHAT_TEMPLATE

# Transforms assistant content, so a sentinel probe cannot be located verbatim.
NONVERBATIM_TEMPLATE = (
    "{% for message in messages %}<|{{ message.role }}|>{{ message.content | upper }}<|end|>\n{% endfor %}"
)

# Position-dependent rendering in the Qwen3 style: only the final assistant turn gets a think block.
THINK_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message.role }}|>"
    "{% if message.role == 'assistant' and loop.last %}<think></think>{% endif %}"
    "{{ message.content }}<|end|>\n"
    "{% endfor %}"
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


def _split_by_mask(example: MaskedExample) -> tuple[list[int], list[int]]:
    pairs = list(zip(example.input_ids, example.labels, strict=True))
    trained = [token for token, label in pairs if label != LABEL_IGNORE_INDEX]
    masked = [token for token, label in pairs if label == LABEL_IGNORE_INDEX]
    return trained, masked


def test_tokenize_dialogue_trains_exactly_on_assistant_turns() -> None:
    tokenizer = _char_tokenizer(CHAT_TEMPLATE)

    example = tokenize_dialogue(tokenizer, _dialogue())

    trained, _masked = _split_by_mask(example)
    assert tokenizer.decode(trained) == "Hi there<|end|>\nSure<|end|>\n"


def test_tokenize_dialogue_masks_the_prompt() -> None:
    tokenizer = _char_tokenizer(CHAT_TEMPLATE)

    example = tokenize_dialogue(tokenizer, _dialogue())

    assert len(example.input_ids) == len(example.labels)
    _trained, masked = _split_by_mask(example)
    assert "You are a payment assistant." in tokenizer.decode(masked)
    kept_labels = [label for label in example.labels if label != LABEL_IGNORE_INDEX]
    trained, _ = _split_by_mask(example)
    assert kept_labels == trained


def test_tokenize_dialogue_matches_full_template_render() -> None:
    tokenizer = _char_tokenizer(CHAT_TEMPLATE)
    dialogue = _dialogue()

    example = tokenize_dialogue(tokenizer, dialogue)

    full_text = (
        "<|system|>You are a payment assistant.<|end|>\n"
        "<|user|>Hello<|end|>\n"
        "<|assistant|>Hi there<|end|>\n"
        "<|user|>Pay rent<|end|>\n"
        "<|assistant|>Sure<|end|>\n"
    )
    assert tokenizer.decode(list(example.input_ids)) == full_text


def test_tokenize_dialogue_handles_state_dependent_template() -> None:
    tokenizer = _char_tokenizer(COUNTING_TEMPLATE)

    example = tokenize_dialogue(tokenizer, _dialogue())

    trained, _masked = _split_by_mask(example)
    assert tokenizer.decode(trained) == "Hi there<|end|>\nSure<|end|>\n"


def test_tokenize_dialogue_position_dependent_think_block_stays_masked() -> None:
    tokenizer = _char_tokenizer(THINK_TEMPLATE)

    example = tokenize_dialogue(tokenizer, _dialogue())

    trained, masked = _split_by_mask(example)
    assert tokenizer.decode(trained) == "Hi there<|end|>\nSure<|end|>\n"
    assert "<think></think>" in tokenizer.decode(masked)


def test_tokenize_dialogue_non_verbatim_template_fails_loudly() -> None:
    tokenizer = _char_tokenizer(NONVERBATIM_TEMPLATE)

    with pytest.raises(TemplateCompatibilityError):
        tokenize_dialogue(tokenizer, _dialogue())
