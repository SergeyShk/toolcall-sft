"""Tool-call quality metrics: how closely predicted calls match the reference calls.

Per turn, matching is order-insensitive and two-phase: exact pairs (name and
arguments equal) are taken first, then name-only pairs among the leftovers, so
an exact match is never consumed by a weaker one. Matching never crosses tool
names, so the per-tool breakdown is the same pairing restricted to one name.

The report answers, in this order: did the model make the right decision on
each turn (call or reply), did it pick the right tool, did it get the arguments
right, and would its calls have run at all (parsed, schema-valid). Free text is
not scored. A turn is correct when its calls are exactly the reference calls and
nothing failed to parse.

A rate with nothing to divide by is None. Argument accuracy covers only the calls
that found a partner by name.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .schema import ToolCall

__all__ = [
    "CallTotals",
    "GroupReport",
    "ToolCallComparison",
    "ToolCallReport",
    "ToolReport",
    "TurnRecord",
    "branch_of",
    "compare_tool_calls",
    "evaluate_turns",
    "format_report",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnRecord:
    """One assistant turn as scored: the reference calls and what the model produced."""

    expected: tuple[ToolCall, ...]
    predicted: tuple[ToolCall, ...]
    dialogue_id: str | None = None
    turn_index: int | None = None
    malformed: int = 0
    """Tool-call blocks in the output that did not parse into a call."""
    invalid: int = 0
    """Predicted calls that parsed but fail their tool's schema."""
    truncated: bool = False
    """Generation stopped at the token budget rather than on an end-of-turn token."""

    @property
    def correct(self) -> bool:
        """Exactly the reference calls, and nothing that failed to parse. Content is not compared."""
        return self.malformed == 0 and compare_tool_calls(self.expected, self.predicted).correct


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallComparison:
    expected_count: int
    predicted_count: int
    name_matches: int
    exact_matches: int
    expected_arguments: int
    """Arguments of the expected calls that found a partner by name."""
    matched_arguments: int
    """Of those, present in the partner with an equal value."""

    @property
    def correct(self) -> bool:
        return self.expected_count == self.predicted_count == self.exact_matches


@dataclass(frozen=True, slots=True, kw_only=True)
class CallTotals:
    expected_calls: int
    predicted_calls: int
    name_matches: int
    exact_matches: int
    expected_arguments: int
    matched_arguments: int

    @property
    def name_recall(self) -> float | None:
        return _rate(self.name_matches, self.expected_calls)

    @property
    def name_precision(self) -> float | None:
        return _rate(self.name_matches, self.predicted_calls)

    @property
    def exact_recall(self) -> float | None:
        return _rate(self.exact_matches, self.expected_calls)

    @property
    def exact_precision(self) -> float | None:
        return _rate(self.exact_matches, self.predicted_calls)

    @property
    def argument_accuracy(self) -> float | None:
        """Over the name-matched pairs only."""
        return _rate(self.matched_arguments, self.expected_arguments)

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "expected": self.expected_calls,
            "predicted": self.predicted_calls,
            "name_matches": self.name_matches,
            "exact_matches": self.exact_matches,
            "name_precision": _round(self.name_precision),
            "name_recall": _round(self.name_recall),
            "exact_precision": _round(self.exact_precision),
            "exact_recall": _round(self.exact_recall),
            "argument_accuracy": _round(self.argument_accuracy),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolReport:
    name: str
    calls: CallTotals


@dataclass(frozen=True, slots=True, kw_only=True)
class GroupReport:
    """Turn-level outcomes for the dialogues of one branch."""

    name: str
    turns: int
    correct_turns: int
    false_fires: int
    missed_calls: int

    @property
    def turn_accuracy(self) -> float | None:
        return _rate(self.correct_turns, self.turns)

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "turns": self.turns,
            "correct_turns": self.correct_turns,
            "turn_accuracy": _round(self.turn_accuracy),
            "false_fires": self.false_fires,
            "missed_calls": self.missed_calls,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallReport:
    turns: int
    correct_turns: int
    expected_call_turns: int
    """Turns whose reference makes at least one call; the rest expect a reply."""
    false_fires: int
    """Reply expected, the model called a tool (or emitted a block that did not parse)."""
    missed_calls: int
    """Call expected, the model replied with text only."""
    calls: CallTotals
    malformed_calls: int
    invalid_calls: int
    truncated_turns: int
    """Turns cut off at the token budget; to the rest of the report they look like a plain reply."""
    by_tool: tuple[ToolReport, ...]
    by_group: tuple[GroupReport, ...]

    @property
    def turn_accuracy(self) -> float | None:
        return _rate(self.correct_turns, self.turns)

    @property
    def expected_text_turns(self) -> int:
        return self.turns - self.expected_call_turns

    @property
    def false_fire_rate(self) -> float | None:
        return _rate(self.false_fires, self.expected_text_turns)

    @property
    def missed_call_rate(self) -> float | None:
        return _rate(self.missed_calls, self.expected_call_turns)

    @property
    def attempted_calls(self) -> int:
        return self.calls.predicted_calls + self.malformed_calls

    @property
    def valid_rate(self) -> float | None:
        """Share of attempted calls a tool router would have run."""
        return _rate(self.calls.predicted_calls - self.invalid_calls, self.attempted_calls)

    def as_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "correct_turns": self.correct_turns,
            "turn_accuracy": _round(self.turn_accuracy),
            "truncated_turns": self.truncated_turns,
            "decisions": {
                "expected_call_turns": self.expected_call_turns,
                "expected_text_turns": self.expected_text_turns,
                "false_fires": self.false_fires,
                "missed_calls": self.missed_calls,
                "false_fire_rate": _round(self.false_fire_rate),
                "missed_call_rate": _round(self.missed_call_rate),
            },
            "calls": {
                **self.calls.as_dict(),
                "malformed": self.malformed_calls,
                "invalid": self.invalid_calls,
                "valid_rate": _round(self.valid_rate),
            },
            "by_tool": {tool.name: tool.calls.as_dict() for tool in self.by_tool},
            "by_group": {group.name: group.as_dict() for group in self.by_group},
        }


def compare_tool_calls(expected: Sequence[ToolCall], predicted: Sequence[ToolCall]) -> ToolCallComparison:
    remaining = list(predicted)
    unmatched: list[ToolCall] = []
    exact = 0
    expected_arguments = 0
    matched_arguments = 0
    for call in expected:
        index = _find(remaining, call, exact_arguments=True)
        if index is None:
            unmatched.append(call)
            continue
        del remaining[index]
        exact += 1
        expected_arguments += len(call.arguments)
        matched_arguments += len(call.arguments)
    name_only = 0
    for call in unmatched:
        index = _find(remaining, call, exact_arguments=False)
        if index is None:
            continue
        partner = remaining.pop(index)
        name_only += 1
        expected_arguments += len(call.arguments)
        matched_arguments += sum(
            1 for key, value in call.arguments.items() if key in partner.arguments and partner.arguments[key] == value
        )
    return ToolCallComparison(
        expected_count=len(expected),
        predicted_count=len(predicted),
        name_matches=exact + name_only,
        exact_matches=exact,
        expected_arguments=expected_arguments,
        matched_arguments=matched_arguments,
    )


def evaluate_turns(turns: Sequence[TurnRecord]) -> ToolCallReport:
    """Score every turn and aggregate: overall, per tool name, per branch."""
    overall = _CallSums()
    per_tool: dict[str, _CallSums] = {}
    per_group: dict[str, _GroupSums] = {}
    correct_turns = 0
    expected_call_turns = 0
    false_fires = 0
    missed_calls = 0
    malformed_calls = 0
    invalid_calls = 0
    truncated_turns = 0
    for turn in turns:
        comparison = compare_tool_calls(turn.expected, turn.predicted)
        overall.add(comparison)
        for name in sorted({call.name for call in (*turn.expected, *turn.predicted)}):
            restricted = compare_tool_calls(_named(turn.expected, name), _named(turn.predicted, name))
            per_tool.setdefault(name, _CallSums()).add(restricted)
        correct = turn.correct
        acted = bool(turn.predicted) or turn.malformed > 0
        false_fire = not turn.expected and acted
        missed = bool(turn.expected) and not acted
        correct_turns += correct
        expected_call_turns += bool(turn.expected)
        false_fires += false_fire
        missed_calls += missed
        malformed_calls += turn.malformed
        invalid_calls += turn.invalid
        truncated_turns += turn.truncated
        if turn.dialogue_id is not None:
            group = per_group.setdefault(branch_of(turn.dialogue_id), _GroupSums())
            group.turns += 1
            group.correct_turns += correct
            group.false_fires += false_fire
            group.missed_calls += missed
    return ToolCallReport(
        turns=len(turns),
        correct_turns=correct_turns,
        expected_call_turns=expected_call_turns,
        false_fires=false_fires,
        missed_calls=missed_calls,
        calls=overall.freeze(),
        malformed_calls=malformed_calls,
        invalid_calls=invalid_calls,
        truncated_turns=truncated_turns,
        by_tool=tuple(ToolReport(name=name, calls=sums.freeze()) for name, sums in sorted(per_tool.items())),
        by_group=tuple(
            GroupReport(
                name=name,
                turns=sums.turns,
                correct_turns=sums.correct_turns,
                false_fires=sums.false_fires,
                missed_calls=sums.missed_calls,
            )
            for name, sums in sorted(per_group.items())
        ),
    )


def branch_of(dialogue_id: str) -> str:
    """The branch a dialogue belongs to: its id up to the last dash, the generator's convention."""
    return dialogue_id.rsplit("-", 1)[0]


def format_report(report: ToolCallReport) -> str:
    """The report as a few lines of text, headline first."""
    lines = [
        f"turns: {report.turns}, correct: {report.correct_turns} ({_percent(report.turn_accuracy)})",
        f"decisions: {report.expected_call_turns} turns expect a call, {report.expected_text_turns} a reply; "
        f"false fires {report.false_fires} ({_percent(report.false_fire_rate)} of reply turns), "
        f"missed calls {report.missed_calls} ({_percent(report.missed_call_rate)} of call turns)",
        f"calls: {report.calls.expected_calls} expected, {report.calls.predicted_calls} predicted, "
        f"{report.malformed_calls} malformed, {report.invalid_calls} invalid ({_percent(report.valid_rate)} would run)",
        f"  name precision {_ratio(report.calls.name_precision)}  recall {_ratio(report.calls.name_recall)} | "
        f"exact precision {_ratio(report.calls.exact_precision)}  recall {_ratio(report.calls.exact_recall)} | "
        f"argument accuracy {_ratio(report.calls.argument_accuracy)}",
    ]
    if report.truncated_turns:
        lines.append(
            f"truncated: {report.truncated_turns} of {report.turns} turns hit the token budget before finishing"
        )
    if report.by_tool:
        lines.append("by tool:")
        rows = [
            (
                tool.name,
                str(tool.calls.expected_calls),
                str(tool.calls.predicted_calls),
                str(tool.calls.exact_matches),
                _ratio(tool.calls.exact_precision),
                _ratio(tool.calls.exact_recall),
                _ratio(tool.calls.argument_accuracy),
            )
            for tool in report.by_tool
        ]
        lines.extend(_table(("tool", "expected", "predicted", "exact", "precision", "recall", "arguments"), rows))
    if report.by_group:
        lines.append("by branch:")
        rows = [
            (
                group.name,
                str(group.turns),
                str(group.correct_turns),
                _ratio(group.turn_accuracy),
                str(group.false_fires),
                str(group.missed_calls),
            )
            for group in report.by_group
        ]
        lines.extend(_table(("branch", "turns", "correct", "accuracy", "false fires", "missed"), rows))
    return "\n".join(lines)


class _CallSums:
    __slots__ = ("expected", "predicted", "name_matches", "exact_matches", "expected_arguments", "matched_arguments")

    def __init__(self) -> None:
        self.expected = 0
        self.predicted = 0
        self.name_matches = 0
        self.exact_matches = 0
        self.expected_arguments = 0
        self.matched_arguments = 0

    def add(self, comparison: ToolCallComparison) -> None:
        self.expected += comparison.expected_count
        self.predicted += comparison.predicted_count
        self.name_matches += comparison.name_matches
        self.exact_matches += comparison.exact_matches
        self.expected_arguments += comparison.expected_arguments
        self.matched_arguments += comparison.matched_arguments

    def freeze(self) -> CallTotals:
        return CallTotals(
            expected_calls=self.expected,
            predicted_calls=self.predicted,
            name_matches=self.name_matches,
            exact_matches=self.exact_matches,
            expected_arguments=self.expected_arguments,
            matched_arguments=self.matched_arguments,
        )


class _GroupSums:
    __slots__ = ("turns", "correct_turns", "false_fires", "missed_calls")

    def __init__(self) -> None:
        self.turns = 0
        self.correct_turns = 0
        self.false_fires = 0
        self.missed_calls = 0


def _named(calls: Iterable[ToolCall], name: str) -> tuple[ToolCall, ...]:
    return tuple(call for call in calls if call.name == name)


def _find(candidates: list[ToolCall], call: ToolCall, *, exact_arguments: bool) -> int | None:
    for index, candidate in enumerate(candidates):
        if candidate.name != call.name:
            continue
        if exact_arguments and candidate.arguments != call.arguments:
            continue
        return index
    return None


def _rate(matches: int, total: int) -> float | None:
    return matches / total if total else None


def _round(rate: float | None) -> float | None:
    return None if rate is None else round(rate, 4)


_UNDEFINED = "—"


def _percent(rate: float | None) -> str:
    return _UNDEFINED if rate is None else f"{100 * rate:.1f}%"


def _ratio(rate: float | None) -> str:
    return _UNDEFINED if rate is None else f"{rate:.3f}"


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    widths = [max(len(header[column]), *(len(row[column]) for row in rows)) for column in range(len(header))]
    aligned = [header, *rows]
    lines: list[str] = []
    for row in aligned:
        cells = [row[0].ljust(widths[0])] + [cell.rjust(width) for cell, width in zip(row[1:], widths[1:], strict=True)]
        lines.append("  " + "  ".join(cells))
    return lines
