import copy
from typing import Any

import pytest

from toolcall_sft import (
    DatasetError,
    Dialogue,
    Message,
    Role,
    ToolCall,
    content_fingerprint,
    dialogue_from_json,
    dialogue_to_json,
    to_chat_record,
)

RAW_DIALOGUE: dict[str, Any] = {
    "dialogue_id": "d1",
    "messages": [
        {"role": "system", "content": "You are a payment assistant."},
        {"role": "user", "content": "Pay $100 to John"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "get_payees", "arguments": {"query": "John"}}]},
        {"role": "tool", "content": '[{"payee_id": "p1", "name": "John Smith"}]'},
        {"role": "assistant", "content": "Found John Smith - confirm $100?"},
    ],
    "tools": [{"type": "function", "function": {"name": "get_payees"}}],
}


def test_dialogue_from_json_valid_tool_flow_parses() -> None:
    dialogue = dialogue_from_json(RAW_DIALOGUE)

    assert dialogue.dialogue_id == "d1"
    assert len(dialogue.messages) == 5
    assert dialogue.messages[2].tool_calls == (ToolCall(name="get_payees", arguments={"query": "John"}),)
    assert dialogue.tools == ({"type": "function", "function": {"name": "get_payees"}},)


def test_dialogue_from_json_unknown_role_raises() -> None:
    raw = copy.deepcopy(RAW_DIALOGUE)
    raw["messages"][1]["role"] = "customer"

    with pytest.raises(DatasetError, match="unknown role"):
        dialogue_from_json(raw)


def test_dialogue_from_json_tool_calls_on_user_raises() -> None:
    raw = copy.deepcopy(RAW_DIALOGUE)
    raw["messages"][1]["tool_calls"] = [{"name": "get_payees", "arguments": {}}]

    with pytest.raises(DatasetError, match="only assistant messages"):
        dialogue_from_json(raw)


def test_dialogue_from_json_tool_result_without_call_raises() -> None:
    raw = copy.deepcopy(RAW_DIALOGUE)
    del raw["messages"][2]

    with pytest.raises(DatasetError, match="without a pending assistant tool call"):
        dialogue_from_json(raw)


def test_dialogue_from_json_not_ending_with_assistant_raises() -> None:
    raw = copy.deepcopy(RAW_DIALOGUE)
    raw["messages"].append({"role": "user", "content": "thanks"})

    with pytest.raises(DatasetError, match="must end with an assistant message"):
        dialogue_from_json(raw)


def test_dialogue_from_json_assistant_without_content_or_calls_raises() -> None:
    raw = copy.deepcopy(RAW_DIALOGUE)
    raw["messages"][4] = {"role": "assistant", "content": ""}

    with pytest.raises(DatasetError, match="needs content or tool calls"):
        dialogue_from_json(raw)


def test_content_fingerprint_ignores_id_and_whitespace() -> None:
    first = dialogue_from_json(RAW_DIALOGUE)
    raw = copy.deepcopy(RAW_DIALOGUE)
    raw["dialogue_id"] = "d2"
    raw["messages"][1]["content"] = "Pay  $100   to John"
    second = dialogue_from_json(raw)

    assert content_fingerprint(first) == content_fingerprint(second)


def test_content_fingerprint_differs_on_tools_change() -> None:
    first = dialogue_from_json(RAW_DIALOGUE)
    raw = copy.deepcopy(RAW_DIALOGUE)
    raw["tools"] = []
    second = dialogue_from_json(raw)

    assert content_fingerprint(first) != content_fingerprint(second)


def test_content_fingerprint_differs_on_content_change() -> None:
    first = dialogue_from_json(RAW_DIALOGUE)
    raw = copy.deepcopy(RAW_DIALOGUE)
    raw["messages"][1]["content"] = "Pay $200 to John"
    second = dialogue_from_json(raw)

    assert content_fingerprint(first) != content_fingerprint(second)


def test_to_chat_record_nests_tool_calls_in_openai_form() -> None:
    dialogue = dialogue_from_json(RAW_DIALOGUE)

    record = to_chat_record(dialogue)

    assert record["messages"][2]["tool_calls"] == [
        {"type": "function", "function": {"name": "get_payees", "arguments": {"query": "John"}}}
    ]
    assert record["tools"] == [{"type": "function", "function": {"name": "get_payees"}}]


def test_dialogue_to_json_round_trips() -> None:
    dialogue = dialogue_from_json(RAW_DIALOGUE)

    assert dialogue_from_json(dialogue_to_json(dialogue)) == dialogue


def test_dialogue_ending_with_tool_call_is_valid() -> None:
    dialogue = Dialogue(
        dialogue_id="d1",
        messages=(
            Message(role=Role.USER, content="Pay rent"),
            Message(role=Role.ASSISTANT, content="", tool_calls=(ToolCall(name="get_payees", arguments={}),)),
        ),
    )

    assert dialogue.messages[-1].tool_calls
