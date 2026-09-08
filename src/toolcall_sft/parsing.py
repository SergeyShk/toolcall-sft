"""What the model actually produced: tool calls parsed out of a generated turn, and
whether each one fits the schema of the tool it names.

Hermes-style tool calls, one JSON object per ``<tool_call>``…``</tool_call>`` block
(Qwen2.5/Qwen3, Hermes; vLLM's ``hermes`` parser reads the same thing). A block that
is not a JSON object with a ``name`` is kept as ``malformed`` rather than dropped:
the model tried to act, and a scorer needs to know that. ``arguments`` may be an
object or the JSON string the OpenAI format serialises it to.

The schema check covers what a tool router would reject before running anything:
unknown tool, missing required argument, undeclared argument, wrong primitive type,
value outside an ``enum``. It is not a JSON Schema validator; nested schemas are not
descended into. A dialogue that declares no tools is checked against nothing at all.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .schema import ToolCall

__all__ = [
    "TOOL_CALL_CLOSE",
    "TOOL_CALL_OPEN",
    "ParsedTurn",
    "check_tool_call",
    "check_tool_calls",
    "parse_assistant_output",
]

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

_BLOCK = re.compile(re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE), re.DOTALL)
_JSON_TYPES: Mapping[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ParsedTurn:
    """A generated assistant turn split into free text and the calls it carries."""

    content: str
    tool_calls: tuple[ToolCall, ...]
    malformed: tuple[str, ...]


def parse_assistant_output(text: str) -> ParsedTurn:
    """Split generated text into content, parsed tool calls and blocks that did not parse.

    An opening tag without its closing tag swallows the rest of the text as one
    malformed block: the model ran out of tokens or lost the format mid-call.
    """
    calls: list[ToolCall] = []
    malformed: list[str] = []
    for match in _BLOCK.finditer(text):
        body = match.group(1).strip()
        call = _parse_call(body)
        if call is None:
            malformed.append(body)
        else:
            calls.append(call)
    remainder = _BLOCK.sub("", text)
    unclosed = remainder.find(TOOL_CALL_OPEN)
    if unclosed != -1:
        malformed.append(remainder[unclosed + len(TOOL_CALL_OPEN) :].strip())
        remainder = remainder[:unclosed]
    return ParsedTurn(content=remainder.strip(), tool_calls=tuple(calls), malformed=tuple(malformed))


def check_tool_call(call: ToolCall, tools: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Problems a tool router would raise for ``call`` against ``tools``; empty means it would run."""
    schema = _parameters_for(call.name, tools)
    if schema is None:
        return (f"unknown tool {call.name!r}",)
    problems: list[str] = []
    required = schema.get("required", [])
    if isinstance(required, list):
        problems.extend(f"missing required argument {name!r}" for name in required if name not in call.arguments)
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return tuple(problems)
    for name, value in call.arguments.items():
        declared = properties.get(name)
        if not isinstance(declared, Mapping):
            problems.append(f"undeclared argument {name!r}")
            continue
        problems.extend(_check_value(name, value, declared))
    return tuple(problems)


def check_tool_calls(calls: Sequence[ToolCall], tools: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, ...], ...]:
    """``check_tool_call`` per call, or no problems at all when the dialogue declares no tools."""
    if not tools:
        return tuple(() for _ in calls)
    return tuple(check_tool_call(call, tools) for call in calls)


def _parse_call(body: str) -> ToolCall | None:
    try:
        raw = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    arguments = _arguments(raw.get("arguments", {}))
    if not isinstance(name, str) or not name or arguments is None:
        return None
    return ToolCall(name=name, arguments=arguments)


def _arguments(value: object) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _parameters_for(name: str, tools: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for tool in tools:
        function = tool.get("function", tool)
        if not isinstance(function, Mapping) or function.get("name") != name:
            continue
        parameters = function.get("parameters")
        return parameters if isinstance(parameters, Mapping) else {}
    return None


def _check_value(name: str, value: object, declared: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    expected = declared.get("type")
    types = [expected] if isinstance(expected, str) else expected if isinstance(expected, list) else []
    known = [item for item in types if isinstance(item, str) and item in _JSON_TYPES]
    if known and not any(_is_type(value, item) for item in known):
        problems.append(f"argument {name!r} should be {' or '.join(known)}, got {type(value).__name__}")
    choices = declared.get("enum")
    if isinstance(choices, list) and not _in_enum(value, choices):
        problems.append(f"argument {name!r} must be one of {', '.join(map(str, choices))}, got {value!r}")
    return problems


def _in_enum(value: object, choices: list[Any]) -> bool:
    # bool is an int in Python, so True must not match a 1 among the choices.
    return any(isinstance(value, bool) == isinstance(choice, bool) and value == choice for choice in choices)


def _is_type(value: object, json_type: str) -> bool:
    # bool is an int in Python but not a number in JSON Schema.
    if isinstance(value, bool):
        return json_type == "boolean"
    if json_type == "integer":
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    return isinstance(value, _JSON_TYPES[json_type])
