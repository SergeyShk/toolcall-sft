"""Training CLI: run SFT experiments, merge adapters, replay dialogues through a model.

Experiment tracking is opt-in through ``tracking.report_to`` in the config, so a
fresh clone trains without a tracking server running anywhere. With ``mlflow``
listed, runs go to MLFLOW_TRACKING_URI when set and a local ./mlruns otherwise.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import click

from .config import (
    DEFAULT_CHAT_TEMPLATE_KWARGS,
    DEVICES,
    PRECISIONS,
    ConfigError,
    ExperimentConfig,
    flatten_for_logging,
    load_experiment_config,
)
from .dataset import load_dialogues
from .masking import TemplateCompatibilityError
from .metrics import evaluate_turns, format_report
from .schema import DatasetError
from .stats import DEFAULT_MAX_SEQ_LENGTH
from .training import PredictionError, PredictionSettings, merge_adapter, run_predictions, run_sft

__all__ = ["main"]


@click.group()
def main() -> None:
    """Fine-tuning commands for toolcall-sft."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
def train(config_path: Path) -> None:
    """Run an SFT experiment described by a YAML config."""
    try:
        experiment = load_experiment_config(config_path)
    except ConfigError as error:
        raise click.ClickException(str(error)) from error
    try:
        with _tracking(experiment, config_path):
            adapter_dir = run_sft(experiment, config_path=config_path)
    except (ConfigError, DatasetError, TemplateCompatibilityError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"adapter saved to {adapter_dir}")


@contextmanager
def _tracking(experiment: ExperimentConfig, config_path: Path) -> Generator[None]:
    """Open an MLflow run around the training call, or nothing at all when tracking is off."""
    if "mlflow" not in experiment.tracking.report_to:
        yield
        return
    import mlflow  # imported lazily: tracking is opt-in, the dependency should be too

    assert experiment.tracking.mlflow_experiment is not None  # enforced by config validation
    mlflow.set_experiment(experiment.tracking.mlflow_experiment)
    with mlflow.start_run(run_name=experiment.run_name):
        mlflow.log_params(flatten_for_logging(experiment))
        mlflow.log_artifact(str(config_path))
        yield


@main.command()
@click.option("--adapter", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--output", type=click.Path(file_okay=False, path_type=Path), required=True)
def merge(adapter: Path, output: Path) -> None:
    """Merge a LoRA adapter into its base model for standalone serving.

    The merged model inherits the base generation_config; for a Qwen3 tune set
    non-thinking sampling (temperature 0.7, top_p 0.8, top_k 20) at the server.
    """
    try:
        merged = merge_adapter(adapter_dir=adapter, output_dir=output)
    except OSError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"merged model saved to {merged}")


@main.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Experiment YAML: supplies the adapter, eval set, template kwargs, max_seq_length, device and precision.",
)
@click.option("--model", default=None, help="HF id, merged checkpoint or adapter directory; needs --out.")
@click.option(
    "--data",
    "data_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Dialogues to replay. Defaults to dataset.eval_path from --config.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Predictions JSONL. Defaults to <output_dir>/predictions.jsonl when the model comes from --config.",
)
@click.option("--max-new-tokens", type=click.IntRange(min=1), default=256, show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--limit", type=click.IntRange(min=1), default=None, help="Replay only the first N dialogues.")
@click.option("--device", type=click.Choice(DEVICES), default=None, help="Overrides --config; default auto.")
@click.option("--precision", type=click.Choice(PRECISIONS), default=None, help="Overrides --config; default auto.")
def predict(
    config_path: Path | None,
    model: str | None,
    data_path: Path | None,
    out_path: Path | None,
    max_new_tokens: int,
    batch_size: int,
    limit: int | None,
    device: str | None,
    precision: str | None,
) -> None:
    """Replay dialogues through a model, one assistant turn at a time, and score what it produced.

    Teacher-forced and greedy: each turn is generated from the reference history, so
    every decision is scored on its own. The JSONL it writes is what `tcsft evaluate`
    reads; run the same command with `--model <base>` for the untuned baseline.
    """
    experiment: ExperimentConfig | None = None
    if config_path is not None:
        try:
            experiment = load_experiment_config(config_path)
        except ConfigError as error:
            raise click.ClickException(str(error)) from error
    if model is None:
        if experiment is None:
            raise click.ClickException("--model is required without --config")
        model = str(experiment.output_dir / "adapter")
        out_path = out_path or experiment.output_dir / "predictions.jsonl"
    elif out_path is None:
        raise click.ClickException("--out is required when --model is given, so a baseline cannot overwrite a run")
    if data_path is None:
        if experiment is None or experiment.dataset.eval_path is None:
            raise click.ClickException("--data is required without a --config that sets dataset.eval_path")
        data_path = experiment.dataset.eval_path
    settings = PredictionSettings(
        model=model,
        max_seq_length=experiment.dataset.max_seq_length if experiment else DEFAULT_MAX_SEQ_LENGTH,
        chat_template_kwargs=experiment.dataset.chat_template_kwargs if experiment else DEFAULT_CHAT_TEMPLATE_KWARGS,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device or (experiment.training.device if experiment else "auto"),
        precision=precision or (experiment.training.precision if experiment else "auto"),
    )
    try:
        dialogues = load_dialogues(data_path)
        if limit is not None:
            dialogues = dialogues[:limit]
        run = run_predictions(settings, dialogues, out_path=out_path)
    except (ConfigError, DatasetError, TemplateCompatibilityError, PredictionError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        f"{len(run.turns)} turns from {run.dialogues} dialogues -> {out_path}  "
        f"({run.seconds:.0f} s, {run.generated_tokens} tokens generated)"
    )
    click.echo(format_report(evaluate_turns(run.turns)))
    click.echo(f"wrong turns with the model's raw output: tcsft evaluate {out_path} --show 10")
