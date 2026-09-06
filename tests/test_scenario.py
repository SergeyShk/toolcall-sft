import json
from importlib.resources import files

from toolcall_sft import SYSTEM_PROMPT, TOOLS, tool_names


def test_system_prompt_is_loaded_from_its_own_file() -> None:
    """The prompt is the file you edit and diff between runs, not a string literal."""
    raw = (files("toolcall_sft") / "system_prompt.txt").read_text(encoding="utf-8")

    assert SYSTEM_PROMPT == raw.strip()
    assert SYSTEM_PROMPT


def test_the_toolset_is_exactly_the_three_documented_tools() -> None:
    assert tool_names() == ("get_payees", "create_payment", "escalate")


def test_every_tool_and_every_parameter_carries_a_description() -> None:
    """Descriptions are what the model reads to choose a tool — they are not decoration,
    and they are rendered into the training prompt by the chat template."""
    for tool in TOOLS:
        function = tool["function"]
        assert function["description"].strip(), f"{function['name']}: no description"
        for name, parameter in function["parameters"]["properties"].items():
            assert parameter.get("description", "").strip(), f"{function['name']}.{name}: no description"


def test_tools_are_json_serializable_in_the_openai_shape() -> None:
    for tool in TOOLS:
        assert tool["type"] == "function"
        assert json.loads(json.dumps(tool)) == tool


def test_the_prompt_tells_the_model_about_the_escape_hatch() -> None:
    assert "escalate" in SYSTEM_PROMPT
