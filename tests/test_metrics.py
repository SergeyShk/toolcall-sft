import json

from toolcall_sft import ToolCall, TurnRecord, branch_of, compare_tool_calls, evaluate_turns, format_report


def _call(name: str, /, **arguments: object) -> ToolCall:
    return ToolCall(name=name, arguments=dict(arguments))


def _turn(
    expected: list[ToolCall], predicted: list[ToolCall], *, dialogue_id: str | None = None, **rest: int
) -> TurnRecord:
    return TurnRecord(expected=tuple(expected), predicted=tuple(predicted), dialogue_id=dialogue_id, **rest)


def test_compare_tool_calls_exact_match() -> None:
    result = compare_tool_calls([_call("get_payees", query="John")], [_call("get_payees", query="John")])

    assert result.exact_matches == 1
    assert result.name_matches == 1
    assert result.correct


def test_compare_tool_calls_name_only_when_arguments_differ() -> None:
    result = compare_tool_calls([_call("get_payees", query="John")], [_call("get_payees", query="Jon")])

    assert result.exact_matches == 0
    assert result.name_matches == 1
    assert not result.correct


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


def test_compare_tool_calls_counts_arguments_over_name_matched_pairs() -> None:
    expected = [_call("create_payment", payee_id="p1", amount=40.0, currency="USD"), _call("escalate", reason="r")]
    predicted = [_call("create_payment", payee_id="p1", amount=40, currency="GBP")]

    result = compare_tool_calls(expected, predicted)

    assert result.expected_arguments == 3  # the unmatched escalate contributes nothing
    assert result.matched_arguments == 2  # 40 == 40.0; GBP != USD


def test_evaluate_turns_headline_and_decisions() -> None:
    report = evaluate_turns(
        [
            _turn([_call("a", x="1")], [_call("a", x="1")]),
            _turn([_call("b", x="1")], [_call("b", x="2")]),
            _turn([_call("c")], []),
            _turn([], []),
            _turn([], [_call("a")]),
        ]
    )

    assert report.turns == 5
    assert report.correct_turns == 2
    assert report.turn_accuracy == 0.4
    assert report.expected_call_turns == 3
    assert report.expected_text_turns == 2
    assert report.missed_calls == 1
    assert report.missed_call_rate == 1 / 3
    assert report.false_fires == 1
    assert report.false_fire_rate == 0.5
    assert report.calls.expected_calls == 3
    assert report.calls.predicted_calls == 3
    assert report.calls.name_matches == 2
    assert report.calls.exact_matches == 1
    assert report.calls.name_recall == 2 / 3
    assert report.calls.name_precision == 2 / 3
    assert report.calls.exact_recall == 1 / 3


def test_evaluate_turns_malformed_block_counts_as_acting_and_spoils_the_turn() -> None:
    report = evaluate_turns(
        [
            _turn([], [], malformed=1),
            _turn([_call("a")], [_call("a")], malformed=1),
            _turn([_call("a")], [], malformed=1),
        ]
    )

    assert report.correct_turns == 0
    assert report.false_fires == 1
    assert report.missed_calls == 0  # it tried to call, badly
    assert report.malformed_calls == 3
    assert report.attempted_calls == 4
    assert report.valid_rate == 0.25


def test_evaluate_turns_invalid_calls_lower_the_valid_rate_only() -> None:
    report = evaluate_turns([_turn([_call("a", x=1)], [_call("a", x=1)], invalid=1)])

    assert report.correct_turns == 1
    assert report.invalid_calls == 1
    assert report.valid_rate == 0.0


def test_evaluate_turns_by_tool_restricts_the_pairing_to_one_name() -> None:
    report = evaluate_turns(
        [
            _turn([_call("get_payees", name="Tom")], [_call("get_payees", name="Tom")]),
            _turn([_call("create_payment", amount=1)], [_call("create_payment", amount=2)]),
            _turn([], [_call("create_payment", amount=3)]),
            _turn([_call("escalate", reason="x")], []),
        ]
    )

    by_tool = {tool.name: tool.calls for tool in report.by_tool}
    assert list(by_tool) == ["create_payment", "escalate", "get_payees"]
    assert by_tool["get_payees"].exact_matches == 1
    assert by_tool["create_payment"].expected_calls == 1
    assert by_tool["create_payment"].predicted_calls == 2
    assert by_tool["create_payment"].name_matches == 1
    assert by_tool["create_payment"].exact_matches == 0
    assert by_tool["create_payment"].exact_precision == 0.0
    assert by_tool["create_payment"].argument_accuracy == 0.0
    assert by_tool["escalate"].expected_calls == 1
    assert by_tool["escalate"].predicted_calls == 0
    assert by_tool["escalate"].name_recall == 0.0


def test_evaluate_turns_groups_by_dialogue_branch() -> None:
    report = evaluate_turns(
        [
            _turn([_call("a")], [_call("a")], dialogue_id="happy_path-00001"),
            _turn([], [_call("a")], dialogue_id="happy_path-00002"),
            _turn([_call("a")], [], dialogue_id="out_of_scope-00003"),
            _turn([], [], dialogue_id=None),
        ]
    )

    groups = {group.name: group for group in report.by_group}
    assert list(groups) == ["happy_path", "out_of_scope"]
    assert groups["happy_path"].turns == 2
    assert groups["happy_path"].correct_turns == 1
    assert groups["happy_path"].turn_accuracy == 0.5
    assert groups["happy_path"].false_fires == 1
    assert groups["out_of_scope"].missed_calls == 1
    assert report.turns == 4  # the ungrouped turn still counts overall


def test_evaluate_turns_rate_without_a_denominator_is_none_not_perfect() -> None:
    report = evaluate_turns([_turn([], [])])

    assert report.turn_accuracy == 1.0
    assert report.false_fire_rate == 0.0  # one reply turn, no false fire
    assert report.calls.name_precision is None
    assert report.calls.exact_recall is None
    assert report.calls.argument_accuracy is None
    assert report.missed_call_rate is None
    assert report.valid_rate is None


def test_report_dashes_a_tool_the_model_never_called() -> None:
    report = evaluate_turns(
        [
            _turn([_call("escalate", reason="x")], [], dialogue_id="out_of_scope-1"),
            _turn([_call("get_payees", name="Tom")], [_call("get_payees", name="Tom")], dialogue_id="happy_path-1"),
        ]
    )

    escalate = next(tool.calls for tool in report.by_tool if tool.name == "escalate")
    assert escalate.exact_recall == 0.0
    assert escalate.exact_precision is None and escalate.argument_accuracy is None
    assert json.loads(json.dumps(report.as_dict()))["by_tool"]["escalate"]["exact_precision"] is None

    row = next(line for line in format_report(report).splitlines() if line.strip().startswith("escalate"))
    assert row.split() == ["escalate", "1", "0", "0", "—", "0.000", "—"]


def test_turn_record_correct_needs_exact_calls_and_nothing_malformed() -> None:
    assert _turn([_call("a", x=1)], [_call("a", x=1)]).correct
    assert not _turn([_call("a", x=1)], [_call("a", x=1)], malformed=1).correct
    assert not _turn([_call("a", x=1)], [_call("a", x=1), _call("a", x=1)]).correct


def test_branch_of_is_the_id_up_to_the_last_dash() -> None:
    assert branch_of("out_of_scope-00012") == "out_of_scope"
    assert branch_of("custom") == "custom"


def test_report_serializes_and_formats() -> None:
    report = evaluate_turns(
        [
            _turn([_call("get_payees", name="Tom")], [_call("get_payees", name="Tom")], dialogue_id="happy_path-1"),
            _turn([], [_call("escalate", reason="x")], dialogue_id="cancelled-2", invalid=1),
        ]
    )

    data = json.loads(json.dumps(report.as_dict()))
    assert data["turn_accuracy"] == 0.5
    assert data["decisions"]["false_fires"] == 1
    assert data["calls"]["valid_rate"] == 0.5
    assert set(data["by_tool"]) == {"get_payees", "escalate"}
    assert data["by_group"]["cancelled"]["false_fires"] == 1

    text = format_report(report)
    assert text.startswith("turns: 2, correct: 1 (50.0%)")
    assert "false fires 1 (100.0% of reply turns)" in text
    assert "by tool:" in text and "by branch:" in text
    assert "  escalate " in text and "  cancelled " in text
