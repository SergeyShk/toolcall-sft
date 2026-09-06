from toolcall_sft import ToolCall, aggregate_comparisons, compare_tool_calls


def _call(name: str, **arguments: str) -> ToolCall:
    return ToolCall(name=name, arguments=dict(arguments))


def test_compare_tool_calls_exact_match() -> None:
    expected = [_call("get_payees", query="John")]
    predicted = [_call("get_payees", query="John")]

    result = compare_tool_calls(expected, predicted)

    assert result.exact_matches == 1
    assert result.name_matches == 1


def test_compare_tool_calls_name_only_when_arguments_differ() -> None:
    expected = [_call("get_payees", query="John")]
    predicted = [_call("get_payees", query="Jon")]

    result = compare_tool_calls(expected, predicted)

    assert result.exact_matches == 0
    assert result.name_matches == 1


def test_compare_tool_calls_exact_pairing_wins_over_name_only() -> None:
    expected = [_call("get_payees", query="John")]
    predicted = [_call("get_payees", query="Jon"), _call("get_payees", query="John")]

    result = compare_tool_calls(expected, predicted)

    assert result.exact_matches == 1
    assert result.predicted_count == 2


def test_compare_tool_calls_missing_and_extra() -> None:
    expected = [_call("get_payees", query="John"), _call("create_payment", amount="100")]
    predicted = [_call("get_payees", query="John"), _call("escalate")]

    result = compare_tool_calls(expected, predicted)

    assert result.exact_matches == 1
    assert result.name_matches == 1
    assert result.expected_count == 2
    assert result.predicted_count == 2


def test_aggregate_comparisons_rates() -> None:
    comparisons = [
        compare_tool_calls([_call("a", x="1")], [_call("a", x="1")]),
        compare_tool_calls([_call("b", x="1")], [_call("b", x="2")]),
        compare_tool_calls([_call("c")], []),
    ]

    report = aggregate_comparisons(comparisons)

    assert report.turns == 3
    assert report.expected_calls == 3
    assert report.predicted_calls == 2
    assert report.name_matches == 2
    assert report.exact_matches == 1
    assert report.name_recall == 2 / 3
    assert report.name_precision == 1.0
    assert report.exact_recall == 1 / 3


def test_aggregate_comparisons_empty_turn_is_perfect() -> None:
    report = aggregate_comparisons([compare_tool_calls([], [])])

    assert report.name_recall == 1.0
    assert report.name_precision == 1.0
    assert report.exact_recall == 1.0
    assert report.exact_precision == 1.0
