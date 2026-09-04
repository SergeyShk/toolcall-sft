"""Tool-call quality metrics: how closely predicted calls match the reference calls.

Matching is order-insensitive and two-phase per turn: exact pairs (name and
arguments equal) are taken first, then name-only pairs among the leftovers, so
an exact match is never consumed by a weaker one.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .schema import ToolCall

__all__ = ["ToolCallComparison", "ToolCallReport", "aggregate_comparisons", "compare_tool_calls"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallComparison:
    expected_count: int
    predicted_count: int
    name_matches: int
    exact_matches: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCallReport:
    turns: int
    expected_calls: int
    predicted_calls: int
    name_matches: int
    exact_matches: int

    @property
    def name_recall(self) -> float:
        return _rate(self.name_matches, self.expected_calls)

    @property
    def name_precision(self) -> float:
        return _rate(self.name_matches, self.predicted_calls)

    @property
    def exact_recall(self) -> float:
        return _rate(self.exact_matches, self.expected_calls)

    @property
    def exact_precision(self) -> float:
        return _rate(self.exact_matches, self.predicted_calls)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "turns": self.turns,
            "expected_calls": self.expected_calls,
            "predicted_calls": self.predicted_calls,
            "name_matches": self.name_matches,
            "exact_matches": self.exact_matches,
            "name_recall": round(self.name_recall, 4),
            "name_precision": round(self.name_precision, 4),
            "exact_recall": round(self.exact_recall, 4),
            "exact_precision": round(self.exact_precision, 4),
        }


def compare_tool_calls(expected: Sequence[ToolCall], predicted: Sequence[ToolCall]) -> ToolCallComparison:
    remaining = list(predicted)
    unmatched: list[ToolCall] = []
    exact = 0
    for call in expected:
        index = _find(remaining, call, exact_arguments=True)
        if index is None:
            unmatched.append(call)
        else:
            del remaining[index]
            exact += 1
    name_only = 0
    for call in unmatched:
        index = _find(remaining, call, exact_arguments=False)
        if index is not None:
            del remaining[index]
            name_only += 1
    return ToolCallComparison(
        expected_count=len(expected),
        predicted_count=len(predicted),
        name_matches=exact + name_only,
        exact_matches=exact,
    )


def aggregate_comparisons(comparisons: Sequence[ToolCallComparison]) -> ToolCallReport:
    return ToolCallReport(
        turns=len(comparisons),
        expected_calls=sum(item.expected_count for item in comparisons),
        predicted_calls=sum(item.predicted_count for item in comparisons),
        name_matches=sum(item.name_matches for item in comparisons),
        exact_matches=sum(item.exact_matches for item in comparisons),
    )


def _find(candidates: list[ToolCall], call: ToolCall, *, exact_arguments: bool) -> int | None:
    for index, candidate in enumerate(candidates):
        if candidate.name != call.name:
            continue
        if exact_arguments and candidate.arguments != call.arguments:
            continue
        return index
    return None


def _rate(matches: int, total: int) -> float:
    return matches / total if total else 1.0
