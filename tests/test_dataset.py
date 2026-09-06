import json
from pathlib import Path

import pytest

from toolcall_sft import (
    DatasetError,
    Dialogue,
    Message,
    Role,
    ToolCall,
    dataset_stats,
    dedup_dialogues,
    load_dialogues,
    split_dialogues,
    write_dialogues,
)


def _dialogue(dialogue_id: str, question: str, answer: str = "Done") -> Dialogue:
    return Dialogue(
        dialogue_id=dialogue_id,
        messages=(
            Message(role=Role.USER, content=question),
            Message(role=Role.ASSISTANT, content=answer),
        ),
    )


def _dialogues(count: int) -> tuple[Dialogue, ...]:
    return tuple(_dialogue(f"d{index}", f"question {index}", f"answer {index}") for index in range(count))


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _valid_line(dialogue_id: str = "d1") -> str:
    return json.dumps(
        {
            "dialogue_id": dialogue_id,
            "messages": [
                {"role": "user", "content": "Pay rent"},
                {"role": "assistant", "content": "Done"},
            ],
        }
    )


def test_load_dialogues_reports_line_number_on_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    _write_lines(path, [_valid_line(), "{broken"])

    with pytest.raises(DatasetError, match=":2:"):
        load_dialogues(path)


def test_load_dialogues_reports_line_number_on_schema_error(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    _write_lines(path, [_valid_line(), json.dumps({"dialogue_id": "d2", "messages": []})])

    with pytest.raises(DatasetError, match=":2:"):
        load_dialogues(path)


def test_load_dialogues_duplicate_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    _write_lines(path, [_valid_line("d1"), _valid_line("d1")])

    with pytest.raises(DatasetError, match="duplicate dialogue_id"):
        load_dialogues(path)


def test_load_dialogues_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="no dialogues"):
        load_dialogues(path)


def test_write_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    dialogues = _dialogues(5)

    write_dialogues(path, dialogues)

    assert load_dialogues(path) == dialogues


def test_dedup_dialogues_keeps_first_occurrence() -> None:
    original = _dialogue("d1", "Pay rent")
    duplicate = _dialogue("d2", "Pay  rent")
    other = _dialogue("d3", "Pay salary")

    unique = dedup_dialogues((original, duplicate, other))

    assert unique == (original, other)


def test_dedup_dialogues_keeps_same_messages_with_different_tools() -> None:
    with_tools = Dialogue(
        dialogue_id="d1",
        messages=_dialogue("x", "Pay rent").messages,
        tools=({"type": "function", "function": {"name": "get_payees"}},),
    )
    without_tools = _dialogue("d2", "Pay rent")

    assert dedup_dialogues((with_tools, without_tools)) == (with_tools, without_tools)


def test_split_dialogues_is_deterministic_and_disjoint() -> None:
    dialogues = _dialogues(200)

    first = split_dialogues(dialogues, eval_fraction=0.2, salt="v1")
    second = split_dialogues(dialogues, eval_fraction=0.2, salt="v1")

    assert first == second
    assert len(first.train) + len(first.evaluation) == len(dialogues)
    train_ids = {dialogue.dialogue_id for dialogue in first.train}
    eval_ids = {dialogue.dialogue_id for dialogue in first.evaluation}
    assert not train_ids & eval_ids


def test_split_dialogues_salt_reshuffles() -> None:
    dialogues = _dialogues(200)

    first = split_dialogues(dialogues, eval_fraction=0.2, salt="v1")
    second = split_dialogues(dialogues, eval_fraction=0.2, salt="v2")

    assert first != second


def test_split_dialogues_bad_fraction_raises() -> None:
    with pytest.raises(DatasetError, match="eval_fraction"):
        split_dialogues(_dialogues(10), eval_fraction=1.5, salt="v1")


def test_dataset_stats_counts_tool_calls() -> None:
    dialogue = Dialogue(
        dialogue_id="d1",
        messages=(
            Message(role=Role.USER, content="Pay $100 to John"),
            Message(role=Role.ASSISTANT, content="", tool_calls=(ToolCall(name="get_payees", arguments={}),)),
            Message(role=Role.TOOL, content="[]"),
            Message(role=Role.ASSISTANT, content="No payee found"),
        ),
    )

    stats = dataset_stats((dialogue,))

    assert stats.dialogues == 1
    assert stats.messages == 4
    assert stats.assistant_messages == 2
    assert stats.tool_call_messages == 1
    assert stats.tool_call_counts == (("get_payees", 1),)
