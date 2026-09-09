from pathlib import Path

import pytest

from toolcall_sft import ConfigError, load_experiment_config

MINIMAL_CONFIG = """
run_name: test-run
base_model: test/model
output_dir: outputs/test-run
dataset:
  train_path: data/train.jsonl
  max_seq_length: 1024
lora:
  r: 8
  alpha: 16
training:
  epochs: 1
  learning_rate: 1.0e-4
  per_device_batch_size: 1
  gradient_accumulation_steps: 1
"""


def _write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_experiment_config_minimal_applies_defaults(tmp_path: Path) -> None:
    config = load_experiment_config(_write_config(tmp_path, MINIMAL_CONFIG))

    assert config.run_name == "test-run"
    assert config.base_model == "test/model"
    assert config.output_dir == Path("outputs/test-run")
    assert config.dataset.train_path == Path("data/train.jsonl")
    assert config.dataset.eval_path is None
    assert config.lora.dropout == 0.05
    assert config.lora.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    assert config.training.warmup_ratio == 0.03
    assert config.training.lr_scheduler == "cosine"
    assert config.training.device == "auto"
    assert config.training.precision == "auto"
    assert config.training.load_in_4bit is False
    assert config.evaluation.max_new_tokens == 256
    assert config.evaluation.batch_size == 4
    assert config.training.use_liger_kernel is False
    assert config.training.seed == 42
    assert config.tracking.report_to == ()
    assert config.tracking.mlflow_experiment is None


def test_load_experiment_config_missing_key_names_it(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG.replace("base_model: test/model\n", "")

    with pytest.raises(ConfigError, match="base_model"):
        load_experiment_config(_write_config(tmp_path, text))


def test_load_experiment_config_wrong_type_raises(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG.replace("learning_rate: 1.0e-4", "learning_rate: fast")

    with pytest.raises(ConfigError, match="training.learning_rate"):
        load_experiment_config(_write_config(tmp_path, text))


def test_load_experiment_config_out_of_range_raises(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG.replace("learning_rate: 1.0e-4", "learning_rate: 2.0")

    with pytest.raises(ConfigError, match="training.learning_rate"):
        load_experiment_config(_write_config(tmp_path, text))


def test_load_experiment_config_rejects_unknown_device(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG + "  device: tpu\n"

    with pytest.raises(ConfigError, match="training.device"):
        load_experiment_config(_write_config(tmp_path, text))


def test_load_experiment_config_rejects_unknown_precision(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG + "  precision: int8\n"

    with pytest.raises(ConfigError, match="training.precision"):
        load_experiment_config(_write_config(tmp_path, text))


def test_load_experiment_config_mlflow_without_experiment_raises(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG + '\ntracking:\n  report_to: ["mlflow"]\n'

    with pytest.raises(ConfigError, match="mlflow_experiment"):
        load_experiment_config(_write_config(tmp_path, text))


def test_load_experiment_config_keeps_declared_tracking(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG + '\ntracking:\n  report_to: ["mlflow"]\n  mlflow_experiment: demo\n'

    config = load_experiment_config(_write_config(tmp_path, text))

    assert config.tracking.report_to == ("mlflow",)
    assert config.tracking.mlflow_experiment == "demo"


def test_load_experiment_config_rejects_unknown_keys(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG + "  grad_checkpointing: false\n"

    with pytest.raises(ConfigError, match="unknown key.*training.grad_checkpointing"):
        load_experiment_config(_write_config(tmp_path, text))


def test_load_experiment_config_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG + "trainer: {}\n"

    with pytest.raises(ConfigError, match="unknown key.*trainer"):
        load_experiment_config(_write_config(tmp_path, text))


def test_load_experiment_config_rejects_unknown_scheduler(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG + "  lr_scheduler: cosinus\n"

    with pytest.raises(ConfigError, match="training.lr_scheduler"):
        load_experiment_config(_write_config(tmp_path, text))


def test_load_experiment_config_eval_batch_follows_train_batch(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG.replace("per_device_batch_size: 1", "per_device_batch_size: 4")

    config = load_experiment_config(_write_config(tmp_path, text))

    assert config.training.eval_batch_size == 4


def test_load_experiment_config_defaults_template_kwargs_to_non_thinking(tmp_path: Path) -> None:
    config = load_experiment_config(_write_config(tmp_path, MINIMAL_CONFIG))

    assert dict(config.dataset.chat_template_kwargs) == {"enable_thinking": False}


def test_load_experiment_config_template_kwargs_can_be_emptied(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG.replace("  max_seq_length: 1024\n", "  max_seq_length: 1024\n  chat_template_kwargs: {}\n")

    config = load_experiment_config(_write_config(tmp_path, text))

    assert dict(config.dataset.chat_template_kwargs) == {}


def test_load_experiment_config_template_kwargs_must_be_scalars(tmp_path: Path) -> None:
    text = MINIMAL_CONFIG.replace(
        "  max_seq_length: 1024\n", "  max_seq_length: 1024\n  chat_template_kwargs: {tools: [1, 2]}\n"
    )

    with pytest.raises(ConfigError, match="chat_template_kwargs"):
        load_experiment_config(_write_config(tmp_path, text))


def test_evaluation_section_overrides_the_replay_defaults(tmp_path: Path) -> None:
    config = load_experiment_config(
        _write_config(tmp_path, MINIMAL_CONFIG + "evaluation:\n  max_new_tokens: 64\n  batch_size: 8\n")
    )

    assert config.evaluation.max_new_tokens == 64
    assert config.evaluation.batch_size == 8


def test_unknown_evaluation_key_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="evaluation.temperature"):
        load_experiment_config(_write_config(tmp_path, MINIMAL_CONFIG + "evaluation:\n  temperature: 0.7\n"))
