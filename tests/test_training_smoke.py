"""End-to-end smoke test of run_sft, merge_adapter and run_predictions on a tiny random Qwen3
built in the test.

CPU, offline, a few seconds. Skipped when torch is not installed.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from click.testing import CliRunner  # noqa: E402
from tokenizers import Regex, Tokenizer, decoders  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Split  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from toolcall_sft import Dialogue, Message, Role, ToolCall, load_experiment_config, write_dialogues  # noqa: E402
from toolcall_sft.cli import main as tcsft  # noqa: E402
from toolcall_sft.training import PredictionSettings, merge_adapter, run_predictions, run_sft  # noqa: E402

QWEN3_TEMPLATE = (Path(__file__).parent / "templates" / "qwen3.jinja").read_text(encoding="utf-8")

pytestmark = pytest.mark.training


def _tiny_model_and_tokenizer(directory: Path) -> None:
    vocab = {chr(code): index for index, code in enumerate(range(32, 127))}
    vocab["\n"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["<|pad|>"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Split(Regex(r"[\s\S]"), behavior="isolated")
    backend.decoder = decoders.Fuse()
    # eos is the end-of-turn token, as in the real Qwen3 tokenizer; generation must be able to stop.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="<|pad|>", eos_token="<|im_end|>"
    )
    tokenizer.chat_template = QWEN3_TEMPLATE
    tokenizer.save_pretrained(str(directory))
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=1024,
    )
    torch.manual_seed(0)  # the untrained weights decide what the replay generates
    Qwen3ForCausalLM(config).save_pretrained(str(directory))


def _dialogues() -> tuple[Dialogue, ...]:
    dialogues: list[Dialogue] = []
    for index in range(6):
        dialogues.append(
            Dialogue(
                dialogue_id=f"smoke-{index}",
                messages=(
                    Message(role=Role.SYSTEM, content="You are a payment assistant."),
                    Message(role=Role.USER, content=f"Send ${index + 1}0 to Noah"),
                    Message(
                        role=Role.ASSISTANT,
                        content="",
                        tool_calls=(ToolCall(name="get_payees", arguments={"name": "Noah"}),),
                    ),
                    Message(role=Role.TOOL, content='[{"payee_id": "p1", "name": "Noah"}]'),
                    Message(role=Role.ASSISTANT, content=f"Noah - send ${index + 1}0?"),
                ),
            )
        )
    return tuple(dialogues)


def test_run_sft_trains_saves_and_records_the_run(tmp_path: Path) -> None:
    model_dir = tmp_path / "base"
    _tiny_model_and_tokenizer(model_dir)
    dialogues = _dialogues()
    write_dialogues(tmp_path / "train.jsonl", dialogues[:4])
    write_dialogues(tmp_path / "eval.jsonl", dialogues[4:])
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        f"""
run_name: smoke
base_model: {model_dir}
output_dir: {tmp_path / "out"}
dataset:
  train_path: {tmp_path / "train.jsonl"}
  eval_path: {tmp_path / "eval.jsonl"}
  max_seq_length: 512
lora:
  r: 4
  alpha: 8
training:
  epochs: 1
  learning_rate: 1.0e-3
  per_device_batch_size: 2
  gradient_accumulation_steps: 1
  device: cpu
  logging_steps: 1
""",
        encoding="utf-8",
    )
    config = load_experiment_config(config_path)

    adapter_dir = run_sft(config, config_path=config_path)

    assert (adapter_dir / "adapter_model.safetensors").exists()
    assert (adapter_dir / "chat_template.jinja").exists()
    record = json.loads((config.output_dir / "run.json").read_text(encoding="utf-8"))
    assert record["dataset"] == {"train_dialogues": 4, "train_examples": 8, "eval_dialogues": 2, "eval_examples": 4}
    assert record["resolved"] == {"device": "cpu", "dtype": "float32"}
    assert (config.output_dir / "config.yaml").read_text(encoding="utf-8") == config_path.read_text(encoding="utf-8")
    example = (config.output_dir / "example.txt").read_text(encoding="utf-8")
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n⟦" in example
    assert example.endswith("<|im_end|>\n⟧")

    merged_dir = merge_adapter(adapter_dir=adapter_dir, output_dir=tmp_path / "merged")

    assert (merged_dir / "model.safetensors").exists()
    assert (merged_dir / "chat_template.jinja").exists()

    # The adapter replays on top of its base; the merged checkpoint replays on its own.
    for model_dir, name in ((adapter_dir, "adapter"), (merged_dir, "merged")):
        settings = PredictionSettings(
            model=str(model_dir),
            max_seq_length=512,
            chat_template_kwargs={"enable_thinking": False},
            max_new_tokens=8,
            batch_size=3,  # 4 turns: one full batch and one short one
            device="cpu",
        )
        predictions_path = tmp_path / f"{name}.predictions.jsonl"

        run = run_predictions(settings, dialogues[4:], out_path=predictions_path)

        assert run.dialogues == 2
        assert len(run.turns) == 4
        records = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
        # One token per character here; a model may legitimately stop on its first token.
        assert run.generated_tokens == sum(len(record["raw_output"]) for record in records)
        assert run.generated_tokens <= 4 * 8
        assert [(record["dialogue_id"], record["turn_index"]) for record in records] == [
            ("smoke-4", 2),
            ("smoke-4", 4),
            ("smoke-5", 2),
            ("smoke-5", 4),
        ]
        assert records[0]["expected"] == [{"name": "get_payees", "arguments": {"name": "Noah"}}]
        assert records[1]["expected"] == [] and records[1]["expected_content"] == "Noah - send $50?"
        assert all(isinstance(record["raw_output"], str) for record in records)
        assert all(record.keys() >= {"predicted", "predicted_content", "malformed", "truncated"} for record in records)
        meta = json.loads(predictions_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta["model"] == str(model_dir)
        assert meta["decoding"] == {"greedy": True, "max_new_tokens": 8, "batch_size": 3}
        assert meta["resolved"] == {"device": "cpu", "dtype": "float32"}

        scored = CliRunner().invoke(tcsft, ["evaluate", str(predictions_path), "--show", "1"])
        assert scored.exit_code == 0, scored.output
        assert scored.output.startswith("turns: 4, correct: ")
