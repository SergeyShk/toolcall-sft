from collections import Counter

import pytest

from toolcall_sft import (
    BRANCH_WEIGHTS,
    Dialogue,
    GenerationError,
    Role,
    branch_names,
    content_fingerprint,
    dialogue_from_json,
    dialogue_to_json,
    generate_dialogues,
    tool_names,
)


def _branch_of(dialogue: Dialogue) -> str:
    return dialogue.dialogue_id.rsplit("-", 1)[0]


def _calls(dialogue: Dialogue) -> list[str]:
    return [call.name for message in dialogue.messages for call in message.tool_calls]


def test_generation_is_deterministic_in_the_seed() -> None:
    first = generate_dialogues(50, seed=7)
    second = generate_dialogues(50, seed=7)

    assert [content_fingerprint(d) for d in first] == [content_fingerprint(d) for d in second]


def test_different_seeds_give_different_corpora() -> None:
    first = {content_fingerprint(d) for d in generate_dialogues(50, seed=7)}
    second = {content_fingerprint(d) for d in generate_dialogues(50, seed=8)}

    assert first != second


def test_every_dialogue_survives_a_json_round_trip() -> None:
    """Generated dialogues must pass the same validation as a hand-written dataset."""
    for dialogue in generate_dialogues(200, seed=1):
        assert dialogue_from_json(dialogue_to_json(dialogue)) == dialogue


def test_dialogues_are_distinct() -> None:
    dialogues = generate_dialogues(300, seed=3)

    assert len({content_fingerprint(d) for d in dialogues}) == 300


def test_branches_are_interleaved_not_grouped() -> None:
    """File order must not correlate with the branch: a split by file position would skew."""
    branches = [_branch_of(dialogue) for dialogue in generate_dialogues(200, seed=9)]

    assert len(set(branches[:40])) > 1
    assert branches[:40] != sorted(branches[:40])


def test_all_branches_appear() -> None:
    seen = {_branch_of(dialogue) for dialogue in generate_dialogues(400, seed=5)}

    assert seen == set(branch_names())


def test_half_the_corpus_withholds_the_write_tool() -> None:
    """A model trained only on the happy path calls create_payment on everything."""
    dialogues = generate_dialogues(400, seed=5)

    writes = sum(1 for dialogue in dialogues if "create_payment" in _calls(dialogue))

    assert 0.4 <= writes / len(dialogues) <= 0.6


def test_only_declared_tools_are_called() -> None:
    allowed = set(tool_names())

    for dialogue in generate_dialogues(200, seed=2):
        assert set(_calls(dialogue)) <= allowed


def test_every_dialogue_opens_with_the_system_prompt_and_ends_on_the_assistant() -> None:
    for dialogue in generate_dialogues(100, seed=4):
        assert dialogue.messages[0].role is Role.SYSTEM
        assert dialogue.messages[-1].role is Role.ASSISTANT
        assert dialogue.tools


def test_branch_weights_are_respected_exactly() -> None:
    """Quota filling, not weighted draws — the corpus matches the declared shares."""
    counts = Counter(_branch_of(dialogue) for dialogue in generate_dialogues(600, seed=11))

    assert counts == Counter({branch: 600 * weight // 100 for branch, weight in BRANCH_WEIGHTS.items()})


def test_weights_can_select_a_single_branch() -> None:
    dialogues = generate_dialogues(60, seed=1, weights={"out_of_scope": 1})

    assert {_branch_of(dialogue) for dialogue in dialogues} == {"out_of_scope"}


def test_unknown_branch_is_rejected() -> None:
    with pytest.raises(GenerationError, match="unknown branches: refund"):
        generate_dialogues(10, weights={"refund": 1})


def test_exhausted_templates_raise_instead_of_returning_fewer() -> None:
    """out_of_scope has a fixed template pool — asking for more must fail loudly."""
    with pytest.raises(GenerationError, match="templates are exhausted"):
        generate_dialogues(400, weights={"out_of_scope": 1})


def test_count_must_be_positive() -> None:
    with pytest.raises(GenerationError, match="at least 1"):
        generate_dialogues(0)
