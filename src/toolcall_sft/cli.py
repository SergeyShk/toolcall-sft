"""Data-side CLI: generate, anonymize, validate, split and score datasets.

Deliberately free of training dependencies — everything here runs after a bare
``uv sync --no-group train``, which keeps the data loop fast to iterate on.
"""

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click
from transformers import AutoTokenizer

from .anonymizer import Anonymizer
from .dataset import dataset_stats, dedup_dialogues, load_dialogues, split_dialogues, write_dialogues
from .generate import GenerationError, branch_names, generate_dialogues
from .metrics import ToolCallComparison, aggregate_comparisons, compare_tool_calls
from .schema import DatasetError, Dialogue, ToolCall
from .stats import (
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_THRESHOLDS,
    compute_token_stats,
    filter_by_length,
    histogram,
    percentile,
    threshold_fits,
)

_base_model_option = click.option(
    "--base-model", default="Qwen/Qwen3-1.7B", show_default=True, help="Tokenizer to measure with."
)

__all__ = ["main"]


@click.group()
def main() -> None:
    """Dataset utilities for toolcall-sft."""


@main.command()
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--count", type=click.IntRange(min=1), default=1000, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True, help="Same seed, same corpus.")
def generate(out_path: Path, count: int, seed: int) -> None:
    """Write synthetic dialogues of the example payment scenario.

    A stand-in so the pipeline is runnable on a fresh clone; replace it with
    your own dialogues in the same schema as soon as you have them.
    """
    try:
        dialogues = generate_dialogues(count, seed=seed)
    except GenerationError as error:
        raise click.ClickException(str(error)) from error
    write_dialogues(out_path, dialogues)
    counts = Counter(dialogue.dialogue_id.rsplit("-", 1)[0] for dialogue in dialogues)
    click.echo(f"{len(dialogues)} dialogues -> {out_path}")
    for branch in branch_names():
        click.echo(f"  {branch}: {counts.get(branch, 0)}")


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--salt", default="v1", show_default=True, help="Changing the salt changes every replacement.")
@click.option("--term", "extra_terms", multiple=True, help="Extra term to redact; repeat for several.")
def anonymize(path: Path, out_path: Path, salt: str, extra_terms: tuple[str, ...]) -> None:
    """Pseudonymize dialogues deterministically and report what was replaced.

    For dialogues captured from a real system. Training bakes data into weights,
    so run this before anything else touches them — and read the caveats in
    ``anonymizer.py``: the pass is mechanical, not a guarantee.
    """
    try:
        dialogues = load_dialogues(path)
    except DatasetError as error:
        raise click.ClickException(str(error)) from error
    anonymizer = Anonymizer(salt=salt, extra_terms=extra_terms)
    totals: Counter[str] = Counter()
    anonymized: list[Dialogue] = []
    for dialogue in dialogues:
        clean, report = anonymizer.anonymize_dialogue(dialogue)
        anonymized.append(clean)
        totals.update(dict(report.replacements))
    write_dialogues(out_path, anonymized)
    click.echo(f"{len(anonymized)} dialogues -> {out_path}")
    for kind, count in sorted(totals.items()):
        click.echo(f"  {kind}: {count}")
    click.echo("Anonymization is heuristic — review a sample before training.")


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate(path: Path) -> None:
    """Check a JSONL dataset against the dialogue schema and print its stats."""
    try:
        dialogues = load_dialogues(path)
    except DatasetError as error:
        raise click.ClickException(str(error)) from error
    stats = dataset_stats(dialogues)
    click.echo(
        f"{path}: {stats.dialogues} dialogues, {stats.messages} messages, "
        f"{stats.assistant_messages} assistant turns, {stats.tool_call_messages} tool-call turns"
    )
    for name, count in stats.tool_call_counts:
        click.echo(f"  {name}: {count}")


@main.command(name="filter")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option(
    "--max-seq-length",
    type=click.IntRange(min=1),
    default=DEFAULT_MAX_SEQ_LENGTH,
    show_default=True,
    help="Token budget. Keep in sync with dataset.max_seq_length of the experiment config.",
)
@_base_model_option
def filter_dataset(path: Path, out_path: Path, max_seq_length: int, base_model: str) -> None:
    """Drop dialogues that do not fit max_seq_length under the base model's tokenizer.

    Uses the same tokenization-with-masks path as training, so what passes here
    is exactly what the trainer will accept instead of silently dropping.
    """
    try:
        dialogues = load_dialogues(path)
    except DatasetError as error:
        raise click.ClickException(str(error)) from error
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    result = filter_by_length(tokenizer, dialogues, max_seq_length=max_seq_length)
    for count in result.dropped:
        click.echo(f"dropped {count.dialogue_id}: {count.total_tokens} tokens > {max_seq_length}")
    for dialogue_id, reason in result.failures:
        click.echo(f"dropped {dialogue_id}: failed to tokenize: {reason}")
    if not result.kept:
        raise click.ClickException(
            f"nothing to write: {len(result.dropped)} dialogues over max_seq_length={max_seq_length}, "
            f"{len(result.failures)} failed to tokenize"
        )
    write_dialogues(out_path, result.kept)
    click.echo(f"{len(result.kept)} of {len(dialogues)} dialogues fit -> {out_path}")


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--train-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--eval-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--eval-fraction", type=float, default=0.1, show_default=True)
@click.option("--salt", default="v1", show_default=True, help="Changing the salt reshuffles the split.")
def split(path: Path, train_out: Path, eval_out: Path, eval_fraction: float, salt: str) -> None:
    """Deduplicate a dataset and split it into train/eval by content hash."""
    try:
        dialogues = load_dialogues(path)
        unique = dedup_dialogues(dialogues)
        result = split_dialogues(unique, eval_fraction=eval_fraction, salt=salt)
    except DatasetError as error:
        raise click.ClickException(str(error)) from error
    write_dialogues(train_out, result.train)
    write_dialogues(eval_out, result.evaluation)
    click.echo(f"{len(dialogues)} dialogues read, {len(dialogues) - len(unique)} duplicates dropped")
    click.echo(f"train: {len(result.train)} -> {train_out}")
    click.echo(f"eval:  {len(result.evaluation)} -> {eval_out}")


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_base_model_option
@click.option("--top", default=5, show_default=True, help="How many longest dialogues to list.")
def stats(path: Path, base_model: str, top: int) -> None:
    """Token-length statistics of a dataset under the base model's tokenizer."""
    try:
        dialogues = load_dialogues(path)
    except DatasetError as error:
        raise click.ClickException(str(error)) from error
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    report = compute_token_stats(tokenizer, dialogues)
    if not report.counts:
        details = "; ".join(f"{dialogue_id}: {reason}" for dialogue_id, reason in report.failures[:3])
        raise click.ClickException(f"no dialogue tokenized successfully: {details}")
    totals = [count.total_tokens for count in report.counts]
    trained = [count.trained_tokens for count in report.counts]
    click.echo(f"{path} @ {base_model}")
    click.echo(f"dialogues: {len(report.counts)} ({len(report.failures)} failed to tokenize)")
    click.echo(f"total tokens:   {_spread(totals)}")
    click.echo(f"trained tokens: {_spread(trained)}  ({100 * sum(trained) / sum(totals):.1f}% of total)")
    fits = "   ".join(
        f"<={threshold}: {fitting} ({100 * fitting / len(totals):.0f}%)"
        for threshold, fitting in threshold_fits(totals, DEFAULT_THRESHOLDS)
    )
    click.echo(f"fits: {fits}")
    click.echo("histogram:")
    buckets = histogram(totals)
    scale = max(count for _label, count in buckets) or 1
    width = max(len(label) for label, _count in buckets)
    for label, count in buckets:
        bar = "█" * round(24 * count / scale)
        click.echo(f"  {label:>{width}} | {bar}{' ' if bar else ''}{count if count else ''}")
    click.echo("longest:")
    for count in sorted(report.counts, key=lambda item: -item.total_tokens)[:top]:
        click.echo(f"  {count.total_tokens:>7}  {count.dialogue_id}")
    for dialogue_id, reason in report.failures:
        click.echo(f"failed: {dialogue_id}: {reason}")


def _spread(values: list[int]) -> str:
    return f"min {min(values)}  p50 {percentile(values, 0.5)}  p90 {percentile(values, 0.9)}  max {max(values)}"


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def evaluate(path: Path) -> None:
    """Score predicted tool calls against expected ones.

    Input: JSONL, each line {"expected": [{"name", "arguments"}], "predicted": [...]}.
    """
    comparisons: list[ToolCallComparison] = []
    for line_number, raw in _read_jsonl(path):
        expected = _parse_calls(raw.get("expected"), path, line_number, "expected")
        predicted = _parse_calls(raw.get("predicted"), path, line_number, "predicted")
        comparisons.append(compare_tool_calls(expected, predicted))
    if not comparisons:
        raise click.ClickException(f"{path}: no records found")
    click.echo(json.dumps(aggregate_comparisons(comparisons).as_dict(), indent=2))


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise click.ClickException(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(raw, dict):
                raise click.ClickException(f"{path}:{line_number}: each line must be a JSON object")
            yield line_number, raw


def _parse_calls(value: object, path: Path, line_number: int, field: str) -> tuple[ToolCall, ...]:
    if not isinstance(value, list):
        raise click.ClickException(f"{path}:{line_number}: '{field}' must be a list")
    calls: list[ToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            raise click.ClickException(f"{path}:{line_number}: each '{field}' entry must be a JSON object")
        name = item.get("name")
        arguments = item.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise click.ClickException(f"{path}:{line_number}: '{field}' entry needs a non-empty 'name'")
        if not isinstance(arguments, dict):
            raise click.ClickException(f"{path}:{line_number}: '{field}' entry 'arguments' must be a JSON object")
        calls.append(ToolCall(name=name, arguments=arguments))
    return tuple(calls)
