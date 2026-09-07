from toolcall_sft import TOOLS, ToolCall, check_tool_call, parse_assistant_output

CALL = '<tool_call>\n{"name": "get_payees", "arguments": {"name": "Tom Bright"}}\n</tool_call>'


def test_parse_single_call() -> None:
    turn = parse_assistant_output(CALL)

    assert turn.tool_calls == (ToolCall(name="get_payees", arguments={"name": "Tom Bright"}),)
    assert turn.content == ""
    assert turn.malformed == ()


def test_parse_keeps_text_around_calls_as_content() -> None:
    second = '<tool_call>\n{"name": "escalate", "arguments": {"reason": "why"}}\n</tool_call>'

    turn = parse_assistant_output(f"Let me check.\n{CALL}\n{second}\nOne moment.")

    assert [call.name for call in turn.tool_calls] == ["get_payees", "escalate"]
    assert turn.content == "Let me check.\n\n\nOne moment."


def test_parse_plain_reply_has_no_calls() -> None:
    turn = parse_assistant_output("Ready to send $40.00 to Tom Bright. Confirm?\n")

    assert turn.tool_calls == ()
    assert turn.content == "Ready to send $40.00 to Tom Bright. Confirm?"


def test_parse_keeps_a_block_that_is_not_json_as_malformed() -> None:
    turn = parse_assistant_output('<tool_call>\n{"name": "get_payees", "arguments": {"name": }\n</tool_call>')

    assert turn.tool_calls == ()
    assert turn.malformed == ('{"name": "get_payees", "arguments": {"name": }',)


def test_parse_block_without_a_name_or_with_non_object_arguments_is_malformed() -> None:
    turn = parse_assistant_output(
        '<tool_call>\n{"arguments": {}}\n</tool_call><tool_call>\n{"name": "x", "arguments": []}\n</tool_call>'
    )

    assert turn.tool_calls == ()
    assert len(turn.malformed) == 2


def test_parse_arguments_default_to_empty() -> None:
    turn = parse_assistant_output('<tool_call>{"name": "escalate"}</tool_call>')

    assert turn.tool_calls == (ToolCall(name="escalate", arguments={}),)


def test_parse_unclosed_block_swallows_the_rest_as_malformed() -> None:
    turn = parse_assistant_output('Sure.\n<tool_call>\n{"name": "get_payees", "arguments": {"name": "To')

    assert turn.tool_calls == ()
    assert turn.malformed == ('{"name": "get_payees", "arguments": {"name": "To',)
    assert turn.content == "Sure."


def _call(name: str, /, **arguments: object) -> ToolCall:
    return ToolCall(name=name, arguments=dict(arguments))


def test_check_accepts_a_call_that_fits_its_schema() -> None:
    call = _call("create_payment", payee_id="pay_1", amount=40.0, currency="USD", reference="Invoice 1")

    assert check_tool_call(call, TOOLS) == ()


def test_check_unknown_tool() -> None:
    assert check_tool_call(_call("send_money", amount=1), TOOLS) == ("unknown tool 'send_money'",)


def test_check_missing_required_argument() -> None:
    problems = check_tool_call(_call("create_payment", payee_id="pay_1", amount=40.0), TOOLS)

    assert problems == ("missing required argument 'currency'",)


def test_check_undeclared_argument() -> None:
    problems = check_tool_call(_call("get_payees", name="Tom", limit=5), TOOLS)

    assert problems == ("undeclared argument 'limit'",)


def test_check_wrong_type_and_enum() -> None:
    problems = check_tool_call(_call("create_payment", payee_id="pay_1", amount="40", currency="GBP"), TOOLS)

    assert problems == (
        "argument 'amount' should be number, got str",
        "argument 'currency' must be one of USD, EUR, CAD, got 'GBP'",
    )


def test_check_bool_is_not_a_number_and_a_whole_float_is_an_integer() -> None:
    tools = (
        {
            "type": "function",
            "function": {
                "name": "t",
                "parameters": {"type": "object", "properties": {"n": {"type": "integer"}, "f": {"type": "boolean"}}},
            },
        },
    )

    assert check_tool_call(_call("t", n=2.0, f=True), tools) == ()
    assert check_tool_call(_call("t", n=True, f=1), tools) == (
        "argument 'n' should be integer, got bool",
        "argument 'f' should be boolean, got int",
    )


def test_check_tool_without_declared_properties_accepts_anything() -> None:
    tools = ({"type": "function", "function": {"name": "ping"}},)

    assert check_tool_call(_call("ping", anything="goes"), tools) == ()
