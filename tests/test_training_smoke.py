"""End-to-end smoke test of the training path on a tiny random Qwen3 built in the test.

Runs on CPU in seconds and fully offline: the model is two layers of width 32 over a
character-level vocabulary, saved to a temporary directory the config points at. It
exercises everything after the tokenizer download — per-turn dataset construction,
LoRA wrapping, the Trainer, evaluation, saving the adapter, and the run record — which
is the part unit tests cannot reach. Skipped when torch is not installed.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tokenizers import Regex, Tokenizer, decoders  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Split  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from toolcall_sft import Dialogue, Message, Role, ToolCall, load_experiment_config, write_dialogues  # noqa: E402
from toolcall_sft.training import merge_adapter, run_sft  # noqa: E402

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
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="<|pad|>")
    tokenizer.chat_template = QWEN3_TEMPLATE
    tokenizer.save_pretrained(str(directory))
    config = Qwen3Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=1024,
    )
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
