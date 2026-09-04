"""Typed experiment configuration loaded from YAML."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ConfigError",
    "DatasetSettings",
    "ExperimentConfig",
    "LoraSettings",
    "TrackingSettings",
    "TrainingSettings",
    "flatten_for_logging",
    "load_experiment_config",
]

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DEVICES = ("auto", "cuda", "mps", "cpu")
PRECISIONS = ("auto", "bf16", "fp16", "fp32")


class ConfigError(Exception):
    """The experiment YAML is missing a key or holds an out-of-range value."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetSettings:
    train_path: Path
    eval_path: Path | None
    max_seq_length: int


@dataclass(frozen=True, slots=True, kw_only=True)
class LoraSettings:
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingSettings:
    epochs: float
    learning_rate: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    lr_scheduler: str
    logging_steps: int
    device: str
    precision: str
    gradient_checkpointing: bool
    load_in_4bit: bool
    use_liger_kernel: bool
    seed: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackingSettings:
    report_to: tuple[str, ...]
    mlflow_experiment: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentConfig:
    run_name: str
    base_model: str
    output_dir: Path
    dataset: DatasetSettings
    lora: LoraSettings
    training: TrainingSettings
    tracking: TrackingSettings


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: config root must be a mapping")
    try:
        return _experiment_from_mapping(raw)
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from error


def flatten_for_logging(config: ExperimentConfig) -> dict[str, str]:
    return {
        "run_name": config.run_name,
        "base_model": config.base_model,
        "output_dir": str(config.output_dir),
        "dataset.train_path": str(config.dataset.train_path),
        "dataset.eval_path": str(config.dataset.eval_path),
        "dataset.max_seq_length": str(config.dataset.max_seq_length),
        "lora.r": str(config.lora.r),
        "lora.alpha": str(config.lora.alpha),
        "lora.dropout": str(config.lora.dropout),
        "lora.target_modules": ",".join(config.lora.target_modules),
        "training.epochs": str(config.training.epochs),
        "training.learning_rate": str(config.training.learning_rate),
        "training.per_device_batch_size": str(config.training.per_device_batch_size),
        "training.gradient_accumulation_steps": str(config.training.gradient_accumulation_steps),
        "training.warmup_ratio": str(config.training.warmup_ratio),
        "training.lr_scheduler": config.training.lr_scheduler,
        "training.logging_steps": str(config.training.logging_steps),
        "training.device": config.training.device,
        "training.precision": config.training.precision,
        "training.gradient_checkpointing": str(config.training.gradient_checkpointing),
        "training.load_in_4bit": str(config.training.load_in_4bit),
        "training.use_liger_kernel": str(config.training.use_liger_kernel),
        "training.seed": str(config.training.seed),
    }


def _experiment_from_mapping(raw: Mapping[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        run_name=_str(raw, "run_name", ""),
        base_model=_str(raw, "base_model", ""),
        output_dir=Path(_str(raw, "output_dir", "")),
        dataset=_dataset_settings(_section(raw, "dataset")),
        lora=_lora_settings(_section(raw, "lora")),
        training=_training_settings(_section(raw, "training")),
        tracking=_tracking_settings(raw.get("tracking", {})),
    )


def _dataset_settings(section: Mapping[str, Any]) -> DatasetSettings:
    eval_path = _opt_str(section, "eval_path", "dataset")
    return DatasetSettings(
        train_path=Path(_str(section, "train_path", "dataset")),
        eval_path=Path(eval_path) if eval_path is not None else None,
        max_seq_length=_int(section, "max_seq_length", "dataset", minimum=1),
    )


def _lora_settings(section: Mapping[str, Any]) -> LoraSettings:
    raw_modules = section.get("target_modules", list(DEFAULT_TARGET_MODULES))
    if not isinstance(raw_modules, list) or not raw_modules or not all(_non_empty_str(m) for m in raw_modules):
        raise ConfigError("'lora.target_modules' must be a non-empty list of module names")
    return LoraSettings(
        r=_int(section, "r", "lora", minimum=1),
        alpha=_int(section, "alpha", "lora", minimum=1),
        dropout=_float(section, "dropout", "lora", default=0.05, minimum=0.0, maximum=0.5),
        target_modules=tuple(raw_modules),
    )


def _training_settings(section: Mapping[str, Any]) -> TrainingSettings:
    return TrainingSettings(
        epochs=_float(section, "epochs", "training", minimum=0.01),
        learning_rate=_float(section, "learning_rate", "training", minimum=1e-8, maximum=1.0),
        per_device_batch_size=_int(section, "per_device_batch_size", "training", minimum=1),
        gradient_accumulation_steps=_int(section, "gradient_accumulation_steps", "training", minimum=1),
        warmup_ratio=_float(section, "warmup_ratio", "training", default=0.03, minimum=0.0, maximum=0.5),
        lr_scheduler=_str(section, "lr_scheduler", "training", default="cosine"),
        logging_steps=_int(section, "logging_steps", "training", default=10, minimum=1),
        device=_choice(section, "device", "training", default="auto", allowed=DEVICES),
        precision=_choice(section, "precision", "training", default="auto", allowed=PRECISIONS),
        gradient_checkpointing=_bool(section, "gradient_checkpointing", "training", default=True),
        load_in_4bit=_bool(section, "load_in_4bit", "training", default=False),
        use_liger_kernel=_bool(section, "use_liger_kernel", "training", default=False),
        seed=_int(section, "seed", "training", default=42, minimum=0),
    )


def _tracking_settings(section: Any) -> TrackingSettings:
    """Tracking is off by default: a fresh clone should train without a tracking server."""
    if not isinstance(section, Mapping):
        raise ConfigError("'tracking' section must be a mapping")
    raw = section.get("report_to", [])
    if not isinstance(raw, list) or not all(_non_empty_str(item) for item in raw):
        raise ConfigError("'tracking.report_to' must be a list of integration names")
    experiment = _opt_str(section, "mlflow_experiment", "tracking")
    if "mlflow" in raw and experiment is None:
        raise ConfigError("'tracking.mlflow_experiment' is required when 'report_to' includes 'mlflow'")
    return TrackingSettings(report_to=tuple(raw), mlflow_experiment=experiment)


def _choice(section: Mapping[str, Any], key: str, context: str, *, default: str, allowed: tuple[str, ...]) -> str:
    value = _str(section, key, context, default=default)
    if value not in allowed:
        raise ConfigError(f"'{_label(key, context)}' must be one of {', '.join(allowed)}, got {value!r}")
    return value


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{key}' section must be a mapping")
    return value


def _label(key: str, context: str) -> str:
    return f"{context}.{key}" if context else key


def _non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _str(section: Mapping[str, Any], key: str, context: str, *, default: str | None = None) -> str:
    value = section.get(key, default)
    if not _non_empty_str(value):
        raise ConfigError(f"'{_label(key, context)}' must be a non-empty string")
    assert isinstance(value, str)
    return value


def _opt_str(section: Mapping[str, Any], key: str, context: str) -> str | None:
    value = section.get(key)
    if value is None:
        return None
    if not _non_empty_str(value):
        raise ConfigError(f"'{_label(key, context)}' must be a non-empty string when present")
    assert isinstance(value, str)
    return value


def _int(section: Mapping[str, Any], key: str, context: str, *, default: int | None = None, minimum: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{_label(key, context)}' must be an integer")
    if value < minimum:
        raise ConfigError(f"'{_label(key, context)}' must be >= {minimum}, got {value}")
    return value


def _float(
    section: Mapping[str, Any],
    key: str,
    context: str,
    *,
    default: float | None = None,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{_label(key, context)}' must be a number")
    number = float(value)
    if number < minimum or (maximum is not None and number > maximum):
        bound = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ConfigError(f"'{_label(key, context)}' must be {bound}, got {number}")
    return number


def _bool(section: Mapping[str, Any], key: str, context: str, *, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"'{_label(key, context)}' must be a boolean")
    return value
