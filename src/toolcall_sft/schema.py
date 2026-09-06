"""Dataset schema: a dialogue is the unit of training, deduplication and splitting.

Storage format (one JSON object per JSONL line) keeps tool calls flat
(``{"name": ..., "arguments": {...}}``); ``chat_messages`` converts them to the
OpenAI-style nested form that HF chat templates expect.

Tool-call ids are optional. Qwen and Llama templates ignore them; Mistral's
requires a nine-character id on every call and a matching ``tool_call_id`` on
every result. When a dialogue carries them they are stored, validated against
the pending calls and passed through to the template; when it does not, nothing
is invented — a template that needs ids has to be fed data that has them.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Any

__all__ = [
    "DatasetError",
    "Dialogue",
    "Message",
    "Role",
    "ToolCall",
    "chat_messages",
    "content_fingerprint",
    "dialogue_from_json",
    "dialogue_to_json",
    "to_chat_record",
]


class DatasetError(Exception):
    """A dataset file or record violates the expected structure."""


@unique
class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Message:
    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Dialogue:
    dialogue_id: str
    messages: tuple[Message, ...]
    tools: tuple[dict[str, Any], ...] = ()


def dialogue_from_json(raw: Mapping[str, Any]) -> Dialogue:
    dialogue_id = raw.get("dialogue_id")
    if not isinstance(dialogue_id, str) or not dialogue_id:
        raise DatasetError("'dialogue_id' must be a non-empty string")
    raw_messages = raw.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise DatasetError(f"dialogue {dialogue_id!r}: 'messages' must be a non-empty list")
    messages = tuple(_message_from_json(dialogue_id, index, item) for index, item in enumerate(raw_messages))
    raw_tools = raw.get("tools", [])
    if not isinstance(raw_tools, list) or not all(isinstance(tool, dict) for tool in raw_tools):
        raise DatasetError(f"dialogue {dialogue_id!r}: 'tools' must be a list of JSON objects")
    _validate_structure(dialogue_id, messages)
    return Dialogue(dialogue_id=dialogue_id, messages=messages, tools=tuple(raw_tools))


def dialogue_to_json(dialogue: Dialogue) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for message in dialogue.messages:
        entry: dict[str, Any] = {"role": message.role.value, "content": message.content}
        if message.tool_calls:
            entry["tool_calls"] = [_tool_call_to_json(call) for call in message.tool_calls]
        if message.tool_call_id is not None:
            entry["tool_call_id"] = message.tool_call_id
        messages.append(entry)
    result: dict[str, Any] = {"dialogue_id": dialogue.dialogue_id, "messages": messages}
    if dialogue.tools:
        result["tools"] = list(dialogue.tools)
    return result


def chat_messages(dialogue: Dialogue) -> list[dict[str, Any]]:
    """Messages in the OpenAI-style form understood by HF chat templates."""
    messages: list[dict[str, Any]] = []
    for message in dialogue.messages:
        entry: dict[str, Any] = {"role": message.role.value, "content": message.content}
        if message.tool_calls:
            calls: list[dict[str, Any]] = []
            for call in message.tool_calls:
                nested: dict[str, Any] = {
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                if call.id is not None:
                    nested["id"] = call.id
                calls.append(nested)
            entry["tool_calls"] = calls
        if message.tool_call_id is not None:
            entry["tool_call_id"] = message.tool_call_id
        messages.append(entry)
    return messages


def to_chat_record(dialogue: Dialogue) -> dict[str, Any]:
    record: dict[str, Any] = {"messages": chat_messages(dialogue)}
    if dialogue.tools:
        record["tools"] = [dict(tool) for tool in dialogue.tools]
    return record


def content_fingerprint(dialogue: Dialogue) -> str:
    """Content-based identity for deduplication and split assignment: ignores ``dialogue_id``
    and tool-call ids, normalises whitespace, and is stable across argument key order. Tools
    are part of the identity — they render into the training prompt, so a different tool
    catalogue is a different example."""
    payload = {
        "messages": [
            [
                message.role.value,
                " ".join(message.content.split()),
                [[call.name, json.dumps(call.arguments, sort_keys=True)] for call in message.tool_calls],
            ]
            for message in dialogue.messages
        ],
        "tools": [json.dumps(tool, sort_keys=True) for tool in dialogue.tools],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def _tool_call_to_json(call: ToolCall) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": call.name, "arguments": call.arguments}
    if call.id is not None:
        entry["id"] = call.id
    return entry


def _message_from_json(dialogue_id: str, index: int, raw: object) -> Message:
    where = f"dialogue {dialogue_id!r}, message {index}"
    if not isinstance(raw, Mapping):
        raise DatasetError(f"{where}: must be a JSON object")
    try:
        role = Role(raw.get("role"))
    except ValueError:
        raise DatasetError(f"{where}: unknown role {raw.get('role')!r}") from None
    content = raw.get("content", "")
    if not isinstance(content, str):
        raise DatasetError(f"{where}: 'content' must be a string")
    raw_calls = raw.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise DatasetError(f"{where}: 'tool_calls' must be a list")
    if raw_calls and role is not Role.ASSISTANT:
        raise DatasetError(f"{where}: only assistant messages may carry 'tool_calls'")
    tool_calls = tuple(_tool_call_from_json(where, item) for item in raw_calls)
    tool_call_id = _opt_id(where, raw.get("tool_call_id"), "tool_call_id")
    if tool_call_id is not None and role is not Role.TOOL:
        raise DatasetError(f"{where}: only tool messages may carry 'tool_call_id'")
    if role is Role.ASSISTANT and not content and not tool_calls:
        raise DatasetError(f"{where}: assistant message needs content or tool calls")
    if role is not Role.ASSISTANT and not content:
        raise DatasetError(f"{where}: '{role.value}' message needs non-empty content")
    return Message(role=role, content=content, tool_calls=tool_calls, tool_call_id=tool_call_id)


def _tool_call_from_json(where: str, raw: object) -> ToolCall:
    if not isinstance(raw, Mapping):
        raise DatasetError(f"{where}: each tool call must be a JSON object")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise DatasetError(f"{where}: tool call 'name' must be a non-empty string")
    arguments = raw.get("arguments", {})
    if not isinstance(arguments, dict):
        raise DatasetError(f"{where}: tool call 'arguments' must be a JSON object")
    return ToolCall(name=name, arguments=arguments, id=_opt_id(where, raw.get("id"), "tool call 'id'"))


def _opt_id(where: str, value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise DatasetError(f"{where}: {label} must be a non-empty string when present")
    return value


def _validate_structure(dialogue_id: str, messages: tuple[Message, ...]) -> None:
    if messages[0].role not in (Role.SYSTEM, Role.USER):
        raise DatasetError(f"dialogue {dialogue_id!r}: must open with a system or user message")
    if any(message.role is Role.SYSTEM for message in messages[1:]):
        raise DatasetError(f"dialogue {dialogue_id!r}: only the first message may be 'system'")
    if messages[-1].role is not Role.ASSISTANT:
        raise DatasetError(f"dialogue {dialogue_id!r}: must end with an assistant message (the training target)")
    # Ids of the tool calls still waiting for a result (None when the call carries no id).
    pending: list[str | None] = []
    for index, message in enumerate(messages):
        where = f"dialogue {dialogue_id!r}, message {index}"
        if message.role is Role.TOOL:
            if not pending:
                raise DatasetError(f"{where}: tool result without a pending assistant tool call")
            if message.tool_call_id is None:
                pending.pop(0)
            elif message.tool_call_id in pending:
                pending.remove(message.tool_call_id)
            else:
                raise DatasetError(f"{where}: tool_call_id {message.tool_call_id!r} matches no pending tool call")
            continue
        if pending:
            # A dialogue may end on a tool call (the call itself is the target), but a user or
            # assistant turn must not arrive while results are outstanding: the template would
            # render a call the model never saw answered.
            raise DatasetError(
                f"{where}: {len(pending)} tool call(s) from the previous assistant turn have no tool result"
            )
        if message.role is Role.ASSISTANT:
            pending = [call.id for call in message.tool_calls]
