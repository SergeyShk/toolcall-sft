import json
from pathlib import Path

from click.testing import CliRunner

from toolcall_sft.cli import main


def _write(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _records() -> list[dict[str, object]]:
    return [
        # The minimum a hand-written file needs.
        {"expected": [{"name": "get_payees", "arguments": {"name": "Tom"}}], "predicted": []},
        # What predict writes.
        {
            "dialogue_id": "happy_path-00007",
            "turn_index": 6,
            "expected": [
                {"name": "create_payment", "arguments": {"payee_id": "p1", "amount": 40.0, "currency": "USD"}}
            ],
            "expected_content": "",
            "predicted": [
                {
                    "name": "create_payment",
                    "arguments": {"payee_id": "p1", "amount": 40.0, "currency": "GBP"},
                    "problems": ["argument 'currency' must be one of USD, EUR, CAD, got 'GBP'"],
                }
            ],
            "predicted_content": "",
            "malformed": [],
            "raw_output": (
                '<tool_call>\n{"name": "create_payment", "arguments": '
                '{"payee_id": "p1", "amount": 40.0, "currency": "GBP"}}\n</tool_call>'
            ),
        },
        {"dialogue_id": "cancelled-00002", "turn_index": 4, "expected": [], "predicted": [], "malformed": ["{oops"]},
        # Cut off at the token budget: no call, and the report has to say why.
        {
            "dialogue_id": "out_of_scope-00004",
            "turn_index": 2,
            "expected": [{"name": "escalate", "arguments": {"reason": "card dispute"}}],
            "predicted": [],
            "truncated": True,
            "raw_output": "Let me look into that for you, it may take a",
        },
    ]


def test_evaluate_prints_the_text_report(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    _write(path, _records())

    result = CliRunner().invoke(main, ["evaluate", str(path)])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("turns: 4, correct: 0 (0.0%)")
    assert "missed calls 2" in result.output
    assert "false fires 1" in result.output
    assert "1 malformed, 1 invalid" in result.output
    assert "truncated: 1 of 4 turns hit the token budget" in result.output
    assert "  cancelled " in result.output and "  happy_path " in result.output


def test_evaluate_json_and_show(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    _write(path, _records())

    result = CliRunner().invoke(main, ["evaluate", str(path), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["turns"] == 4
    assert data["truncated_turns"] == 1
    assert data["calls"]["invalid"] == 1
    assert data["by_group"]["happy_path"]["turns"] == 1

    result = CliRunner().invoke(main, ["evaluate", str(path), "--show", "2"])
    assert result.exit_code == 0, result.output
    assert "--- ?, assistant message ?" in result.output
    assert 'expected:  get_payees {"name": "Tom"}' in result.output
    assert "predicted: (no call)" in result.output
    assert "--- happy_path-00007, assistant message 6" in result.output
    assert "[argument 'currency' must be one of" in result.output
    assert "raw: '<tool_call>" in result.output
    assert "cancelled-00002" not in result.output  # --show 2 stops after two

    result = CliRunner().invoke(main, ["evaluate", str(path), "--show", "4"])
    assert result.exit_code == 0, result.output
    assert "--- out_of_scope-00004, assistant message 2 (truncated)" in result.output


def test_evaluate_rejects_a_bad_record(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    _write(path, [{"expected": [{"name": ""}], "predicted": []}])

    result = CliRunner().invoke(main, ["evaluate", str(path)])

    assert result.exit_code != 0
    assert "needs a non-empty 'name'" in result.output


def test_evaluate_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["evaluate", str(path)])

    assert result.exit_code != 0
    assert "no records found" in result.output


def _dialogue(dialogue_id: str) -> dict[str, object]:
    return {
        "dialogue_id": dialogue_id,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "create_payment",
                    "parameters": {
                        "type": "object",
                        "properties": {"payee_id": {"type": "string"}, "amount": {"type": "number"}},
                        "required": ["payee_id", "amount"],
                    },
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "pay p1 40"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "create_payment", "arguments": {"payee_id": "p1", "amount": 40.0}}],
            },
        ],
    }


def test_evaluate_rechecks_the_calls_against_the_dialogues(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    # No "problems" in the file: another harness wrote it.
    _write(
        predictions,
        [
            {
                "dialogue_id": "happy_path-1",
                "turn_index": 1,
                "expected": [{"name": "create_payment", "arguments": {"payee_id": "p1", "amount": 40.0}}],
                "predicted": [
                    {"name": "create_payment", "arguments": {"payee_id": "p1", "amount": 40.0, "currency": "USD"}}
                ],
            }
        ],
    )
    data = tmp_path / "eval.jsonl"
    _write(data, [_dialogue("happy_path-1")])

    alone = CliRunner().invoke(main, ["evaluate", str(predictions), "--json"])
    assert json.loads(alone.output)["calls"]["invalid"] == 0

    checked = CliRunner().invoke(main, ["evaluate", str(predictions), "--data", str(data), "--json"])
    assert checked.exit_code == 0, checked.output
    assert json.loads(checked.output)["calls"]["invalid"] == 1

    shown = CliRunner().invoke(main, ["evaluate", str(predictions), "--data", str(data), "--show", "1"])
    assert "undeclared argument 'currency'" in shown.output


def test_evaluate_rejects_data_that_does_not_cover_a_record(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    _write(predictions, [{"dialogue_id": "elsewhere-1", "expected": [], "predicted": []}])
    data = tmp_path / "eval.jsonl"
    _write(data, [_dialogue("happy_path-1")])

    result = CliRunner().invoke(main, ["evaluate", str(predictions), "--data", str(data)])

    assert result.exit_code != 0
    assert "is not in the --data file" in result.output
