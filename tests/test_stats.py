import pytest
from tokenizers import Regex, Tokenizer, decoders
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

from toolcall_sft import (
    LABEL_IGNORE_INDEX,
    Dialogue,
    DialogueTokenCount,
    Message,
    Role,
    compute_token_stats,
    filter_by_length,
    histogram,
    percentile,
    threshold_fits,
    tokenize_dialogue,
)

CHAT_TEMPLATE = "{% for message in messages %}<|{{ message.role }}|>{{ message.content }}<|end|>\n{% endfor %}"

NONVERBATIM_TEMPLATE = (
    "{% for message in messages %}<|{{ message.role }}|>{{ message.content | upper }}<|end|>\n{% endfor %}"
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


def _dialogue(dialogue_id: str, reply: str) -> Dialogue:
    return Dialogue(
        dialogue_id=dialogue_id,
        messages=(
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content=reply),
        ),
    )


def test_compute_token_stats_counts_total_and_trained_tokens() -> None:
    tokenizer = _char_tokenizer(CHAT_TEMPLATE)
    dialogue = _dialogue("d1", "Hi")

    report = compute_token_stats(tokenizer, (dialogue,))

    assert len(report.counts) == 1
    count = report.counts[0]
    # Char-level tokenizer: token counts equal rendered character counts.
    assert count.total_tokens == len("<|user|>Hello<|end|>\n<|assistant|>Hi<|end|>\n")
    assert count.trained_tokens == len("Hi<|end|>\n")
    assert report.failures == ()


def test_compute_token_stats_records_failures_and_continues() -> None:
    tokenizer = _char_tokenizer(NONVERBATIM_TEMPLATE)

    report = compute_token_stats(tokenizer, (_dialogue("d1", "Hi"), _dialogue("d2", "Yo")))

    assert report.counts == ()
    assert [dialogue_id for dialogue_id, _reason in report.failures] == ["d1", "d2"]


def test_filter_by_length_keeps_all_when_everything_fits() -> None:
    tokenizer = _char_tokenizer(CHAT_TEMPLATE)
    dialogues = (_dialogue("d1", "Hi"), _dialogue("d2", "Yo"))

    result = filter_by_length(tokenizer, dialogues, max_seq_length=1000)

    assert result.kept == dialogues
    assert all(kept is original for kept, original in zip(result.kept, dialogues, strict=True))
    assert result.dropped == ()
    assert result.failures == ()


def test_filter_by_length_drops_long_dialogues_preserving_kept_order() -> None:
    tokenizer = _char_tokenizer(CHAT_TEMPLATE)
    short_one = _dialogue("d1", "Hi")
    long_one = _dialogue("d2", "A" * 100)
    short_two = _dialogue("d3", "Yo")
    long_example = tokenize_dialogue(tokenizer, long_one)

    result = filter_by_length(tokenizer, (short_one, long_one, short_two), max_seq_length=50)

    assert result.kept == (short_one, short_two)
    assert result.kept[0] is short_one
    assert result.kept[1] is short_two
    assert result.dropped == (
        DialogueTokenCount(
            dialogue_id="d2",
            total_tokens=len(long_example.input_ids),
            trained_tokens=sum(1 for label in long_example.labels if label != LABEL_IGNORE_INDEX),
        ),
    )
    assert result.failures == ()


def test_filter_by_length_boundary_is_inclusive() -> None:
    tokenizer = _char_tokenizer(CHAT_TEMPLATE)
    dialogue = _dialogue("d1", "Hi")
    exact = len(tokenize_dialogue(tokenizer, dialogue).input_ids)

    kept_result = filter_by_length(tokenizer, (dialogue,), max_seq_length=exact)
    dropped_result = filter_by_length(tokenizer, (dialogue,), max_seq_length=exact - 1)

    assert kept_result.kept == (dialogue,)
    assert kept_result.dropped == ()
    assert dropped_result.kept == ()
    assert [count.dialogue_id for count in dropped_result.dropped] == ["d1"]


def test_filter_by_length_sends_template_failures_to_failures_only() -> None:
    tokenizer = _char_tokenizer(CHAT_TEMPLATE)
    good = _dialogue("d1", "Hi")
    # No assistant turn means nothing to train on: tokenize_dialogue raises
    # TemplateCompatibilityError for this dialogue while its sibling still tokenizes.
    bad = Dialogue(dialogue_id="d2", messages=(Message(role=Role.USER, content="Hello"),))

    result = filter_by_length(tokenizer, (good, bad), max_seq_length=1000)

    assert result.kept == (good,)
    assert result.dropped == ()
    assert [dialogue_id for dialogue_id, _reason in result.failures] == ["d2"]
    assert result.failures[0][1] != ""


def test_filter_by_length_non_verbatim_template_puts_everything_in_failures() -> None:
    tokenizer = _char_tokenizer(NONVERBATIM_TEMPLATE)

    result = filter_by_length(tokenizer, (_dialogue("d1", "Hi"),), max_seq_length=1000)

    assert result.kept == ()
    assert result.dropped == ()
    assert [dialogue_id for dialogue_id, _reason in result.failures] == ["d1"]


@pytest.mark.parametrize("max_seq_length", [0, -1])
def test_filter_by_length_rejects_non_positive_max_seq_length(max_seq_length: int) -> None:
    tokenizer = _char_tokenizer(CHAT_TEMPLATE)

    with pytest.raises(ValueError, match="max_seq_length"):
        filter_by_length(tokenizer, (_dialogue("d1", "Hi"),), max_seq_length=max_seq_length)


def test_percentile_nearest_rank() -> None:
    values = [10, 20, 30, 40]

    assert percentile(values, 0.0) == 10
    assert percentile(values, 0.5) == 30
    assert percentile(values, 1.0) == 40
    assert percentile([7], 0.9) == 7


def test_percentile_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="fraction"):
        percentile([1], 1.5)


def test_threshold_fits_counts_values_under_each_threshold() -> None:
    assert threshold_fits([100, 200, 300], (150, 250, 350)) == ((150, 1), (250, 2), (350, 3))


def test_histogram_buckets_values_with_open_end() -> None:
    buckets = dict(histogram([100, 5000, 70000], edges=(4096, 8192, 16384, 32768, 65536)))

    assert buckets["0-4095"] == 1
    assert buckets["4096-8191"] == 1
    assert buckets["65536+"] == 1
    assert buckets["8192-16383"] == 0
