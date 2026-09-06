"""Training CLI: run SFT experiments and merge adapters.

Experiment tracking is opt-in through ``tracking.report_to`` in the config, so a
fresh clone trains without a tracking server running anywhere. With ``mlflow``
listed, runs go to MLFLOW_TRACKING_URI when set and a local ./mlruns otherwise.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import click

from .config import ConfigError, ExperimentConfig, flatten_for_logging, load_experiment_config
from .masking import TemplateCompatibilityError
from .schema import DatasetError
from .training import merge_adapter, run_sft

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

    The merged model inherits the base model's generation_config. For a Qwen3 tune,
    which is only valid in non-thinking mode, set the sampling parameters Qwen
    recommends for that mode at the server (temperature 0.7, top_p 0.8, top_k 20).
    """
    try:
        merged = merge_adapter(adapter_dir=adapter, output_dir=output)
    except OSError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"merged model saved to {merged}")
