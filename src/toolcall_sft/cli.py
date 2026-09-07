"""Data-side CLI: generate, anonymize, validate, stats, render, filter, split, evaluate.

Commands that need a tokenizer import transformers when they run; the rest work
on a bare install.
"""

import json
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from .anonymizer import Anonymizer
from .config import DEFAULT_CHAT_TEMPLATE_KWARGS, ConfigError, load_experiment_config
from .dataset import dataset_stats, dedup_dialogues, load_dialogues, split_dialogues, write_dialogues
from .generate import GenerationError, branch_names, generate_dialogues
from .masking import TemplateCompatibilityError, render_example, tokenize_dialogue
from .metrics import TurnRecord, branch_of, evaluate_turns, format_report
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

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = ["main"]

DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"

_config_option = click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Experiment YAML to take base_model, max_seq_length and chat_template_kwargs from.",
)
_base_model_option = click.option(
    "--base-model",
    default=None,
    help=f"Tokenizer to measure with. Overrides --config; defaults to {DEFAULT_BASE_MODEL}.",
)


@click.group()
def main() -> None:
    """Dataset utilities for toolcall-sft."""


@main.command()
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--count", type=click.IntRange(min=1), default=1000, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True, help="Same seed, same corpus.")
def generate(out_path: Path, count: int, seed: int) -> None:
    """Write synthetic dialogues for the example payment scenario."""
    try:
        dialogues = generate_dialogues(count, seed=seed)
    except GenerationError as error:
        raise click.ClickException(str(error)) from error
    write_dialogues(out_path, dialogues)
    counts = _branch_counts(dialogues)
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

    Heuristic, not a guarantee; see anonymizer.py and review a sample.
    """
    dialogues = _load(path)
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
    dialogues = _load(path)
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
    default=None,
    help=f"Token budget per example. Overrides --config; defaults to {DEFAULT_MAX_SEQ_LENGTH}.",
)
@_base_model_option
@_config_option
def filter_dataset(
    path: Path, out_path: Path, max_seq_length: int | None, base_model: str | None, config_path: Path | None
) -> None:
    """Drop dialogues whose longest example does not fit max_seq_length.

    Same per-turn tokenization as training, so what passes is what the trainer accepts.
    """
    dialogues = _load(path)
    settings = _TokenizerSettings.resolve(config_path, base_model=base_model, max_seq_length=max_seq_length)
    result = filter_by_length(
        settings.tokenizer(),
        dialogues,
        max_seq_length=settings.max_seq_length,
        chat_template_kwargs=settings.chat_template_kwargs,
    )
    for count in result.dropped:
        click.echo(f"dropped {count.dialogue_id}: longest example {count.longest_example} > {settings.max_seq_length}")
    for dialogue_id, reason in result.failures:
        click.echo(f"dropped {dialogue_id}: failed to tokenize: {reason}")
    if not result.kept:
        raise click.ClickException(
            f"nothing to write: {len(result.dropped)} dialogues over max_seq_length={settings.max_seq_length}, "
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
    """Deduplicate a dataset and split it into train/eval by content hash.

    Not stratified: check the per-branch breakdown it prints.
    """
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
    train_counts = _branch_counts(result.train)
    eval_counts = _branch_counts(result.evaluation)
    if len(train_counts) > 1 or len(eval_counts) > 1:
        click.echo("per branch (dialogue_id prefix): train / eval")
        for branch in sorted(set(train_counts) | set(eval_counts)):
            click.echo(f"  {branch}: {train_counts.get(branch, 0)} / {eval_counts.get(branch, 0)}")


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_base_model_option
@_config_option
@click.option("--top", default=5, show_default=True, help="How many longest dialogues to list.")
def stats(path: Path, base_model: str | None, config_path: Path | None, top: int) -> None:
    """Token-length statistics under the base model's tokenizer, per training example."""
    dialogues = _load(path)
    settings = _TokenizerSettings.resolve(config_path, base_model=base_model, max_seq_length=None)
    report = compute_token_stats(settings.tokenizer(), dialogues, chat_template_kwargs=settings.chat_template_kwargs)
    if not report.counts:
        details = "; ".join(f"{dialogue_id}: {reason}" for dialogue_id, reason in report.failures[:3])
        raise click.ClickException(f"no dialogue tokenized successfully: {details}")
    longest = [count.longest_example for count in report.counts]
    epoch = [count.epoch_tokens for count in report.counts]
    trained = [count.trained_tokens for count in report.counts]
    examples = sum(count.examples for count in report.counts)
    click.echo(f"{path} @ {settings.base_model}  ({settings.describe_kwargs()})")
    click.echo(
        f"dialogues: {len(report.counts)} ({len(report.failures)} failed to tokenize), "
        f"examples: {examples} (one per assistant turn)"
    )
    click.echo(f"longest example per dialogue: {_spread(longest)}   <- must fit max_seq_length")
    click.echo(f"tokens per epoch per dialogue: {_spread(epoch)}   ({sum(epoch)} total)")
    click.echo(f"trained tokens per dialogue:   {_spread(trained)}   ({100 * sum(trained) / sum(epoch):.1f}% of epoch)")
    fits = "   ".join(
        f"<={threshold}: {fitting} ({100 * fitting / len(longest):.0f}%)"
        for threshold, fitting in threshold_fits(longest, DEFAULT_THRESHOLDS)
    )
    click.echo(f"fits: {fits}")
    click.echo("histogram of longest example:")
    buckets = histogram(longest)
    scale = max(count for _label, count in buckets) or 1
    width = max(len(label) for label, _count in buckets)
    for label, count in buckets:
        bar = "█" * round(24 * count / scale)
        click.echo(f"  {label:>{width}} | {bar}{' ' if bar else ''}{count if count else ''}")
    click.echo("longest:")
    for count in sorted(report.counts, key=lambda item: -item.longest_example)[:top]:
        click.echo(f"  {count.longest_example:>7}  {count.dialogue_id}")
    for dialogue_id, reason in report.failures:
        click.echo(f"failed: {dialogue_id}: {reason}")


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dialogue-id", default=None, help="Which dialogue to show; defaults to the first in the file.")
@click.option(
    "--turn",
    type=click.IntRange(min=1),
    default=None,
    help="Show only the N-th assistant turn's example (1-based); defaults to every turn.",
)
@_base_model_option
@_config_option
def render(
    path: Path, dialogue_id: str | None, turn: int | None, base_model: str | None, config_path: Path | None
) -> None:
    """Print a dialogue as the model trains on it: one example per assistant turn, loss tokens in ⟦ ⟧.

    Everything outside the brackets is the prompt the server builds at that turn.
    """
    dialogues = _load(path)
    if dialogue_id is None:
        dialogue = dialogues[0]
    else:
        matches = [candidate for candidate in dialogues if candidate.dialogue_id == dialogue_id]
        if not matches:
            raise click.ClickException(f"{path}: no dialogue with id {dialogue_id!r}")
        dialogue = matches[0]
    settings = _TokenizerSettings.resolve(config_path, base_model=base_model, max_seq_length=None)
    tokenizer = settings.tokenizer()
    try:
        examples = tokenize_dialogue(tokenizer, dialogue, chat_template_kwargs=settings.chat_template_kwargs)
    except TemplateCompatibilityError as error:
        raise click.ClickException(str(error)) from error
    if turn is not None:
        if turn > len(examples):
            raise click.ClickException(f"{dialogue.dialogue_id} has {len(examples)} assistant turns, not {turn}")
        examples = (examples[turn - 1],)
    click.echo(f"{dialogue.dialogue_id} @ {settings.base_model}  ({settings.describe_kwargs()})")
    for number, example in enumerate(examples, start=1):
        click.echo(
            f"\n=== example {number}/{len(examples)}: assistant message {example.turn_index}, "
            f"{len(example.input_ids)} tokens, {example.target_tokens} with loss ==="
        )
        click.echo(render_example(tokenizer, example))


def _spread(values: list[int]) -> str:
    return f"min {min(values)}  p50 {percentile(values, 0.5)}  p90 {percentile(values, 0.9)}  max {max(values)}"


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Print the full report as JSON instead of text.")
@click.option(
    "--show",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Also print the first N turns the model got wrong, with its raw output when the file has it.",
)
def evaluate(path: Path, as_json: bool, show: int) -> None:
    """Score predicted tool calls against expected ones, one assistant turn per line.

    Input: the JSONL `tcsft-train predict` writes. The minimum is
    {"expected": [{"name", "arguments"}], "predicted": [...]}; "dialogue_id" adds the
    per-branch breakdown, "malformed" and per-call "problems" feed the validity rate,
    "truncated" marks a turn cut off at the token budget.
    """
    turns: list[TurnRecord] = []
    raw_turns: list[dict[str, Any]] = []
    for line_number, raw in _read_jsonl(path):
        where = f"{path}:{line_number}"
        expected, _ = _parse_calls(raw.get("expected"), where, "expected")
        predicted, invalid = _parse_calls(raw.get("predicted"), where, "predicted")
        turns.append(
            TurnRecord(
                expected=expected,
                predicted=predicted,
                dialogue_id=_opt_str(raw, "dialogue_id", where),
                turn_index=_opt_int(raw, "turn_index", where),
                malformed=_malformed_count(raw.get("malformed"), where),
                invalid=invalid,
                truncated=_opt_bool(raw, "truncated", where),
            )
        )
        raw_turns.append(raw)
    if not turns:
        raise click.ClickException(f"{path}: no records found")
    report = evaluate_turns(turns)
    click.echo(json.dumps(report.as_dict(), indent=2) if as_json else format_report(report))
    wrong = [(turn, raw) for turn, raw in zip(turns, raw_turns, strict=True) if not turn.correct]
    for turn, raw in wrong[:show]:
        where = "?" if turn.turn_index is None else str(turn.turn_index)
        cut = " (truncated)" if turn.truncated else ""
        click.echo(f"--- {turn.dialogue_id or '?'}, assistant message {where}{cut}")
        click.echo(f"expected:  {_describe_calls(raw.get('expected'))}")
        click.echo(f"predicted: {_describe_calls(raw.get('predicted'))}")
        if raw.get("malformed"):
            click.echo(f"malformed: {json.dumps(raw['malformed'], ensure_ascii=False)}")
        if isinstance(raw.get("raw_output"), str):
            click.echo(f"raw: {raw['raw_output']!r}")


class _TokenizerSettings:
    """Settings for the tokenizer-dependent commands: flags, then --config, then the shipped Qwen3 defaults."""

    __slots__ = ("base_model", "max_seq_length", "chat_template_kwargs")

    def __init__(self, base_model: str, max_seq_length: int, chat_template_kwargs: Mapping[str, Any]) -> None:
        self.base_model = base_model
        self.max_seq_length = max_seq_length
        self.chat_template_kwargs = chat_template_kwargs

    @classmethod
    def resolve(
        cls, config_path: Path | None, *, base_model: str | None, max_seq_length: int | None
    ) -> "_TokenizerSettings":
        if config_path is not None:
            try:
                config = load_experiment_config(config_path)
            except ConfigError as error:
                raise click.ClickException(str(error)) from error
            return cls(
                base_model or config.base_model,
                max_seq_length or config.dataset.max_seq_length,
                config.dataset.chat_template_kwargs,
            )
        return cls(
            base_model or DEFAULT_BASE_MODEL, max_seq_length or DEFAULT_MAX_SEQ_LENGTH, DEFAULT_CHAT_TEMPLATE_KWARGS
        )

    def describe_kwargs(self) -> str:
        if not self.chat_template_kwargs:
            return "no chat template kwargs"
        return "chat template kwargs: " + ", ".join(
            f"{key}={value}" for key, value in self.chat_template_kwargs.items()
        )

    def tokenizer(self) -> "PreTrainedTokenizerBase":
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise click.ClickException(
                "this command renders the chat template and needs transformers; install the 'train' or "
                "'test' dependency group (uv sync --group test)"
            ) from error
        return AutoTokenizer.from_pretrained(self.base_model)


def _load(path: Path) -> tuple[Dialogue, ...]:
    try:
        return load_dialogues(path)
    except DatasetError as error:
        raise click.ClickException(str(error)) from error


def _branch_counts(dialogues: tuple[Dialogue, ...]) -> Counter[str]:
    """Dialogues per branch, the branch being the dialogue_id up to its last dash."""
    return Counter(branch_of(dialogue.dialogue_id) for dialogue in dialogues)


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


def _parse_calls(value: object, where: str, field: str) -> tuple[tuple[ToolCall, ...], int]:
    """The calls under ``field`` and how many of them carry schema ``problems``."""
    if not isinstance(value, list):
        raise click.ClickException(f"{where}: '{field}' must be a list")
    calls: list[ToolCall] = []
    invalid = 0
    for item in value:
        if not isinstance(item, dict):
            raise click.ClickException(f"{where}: each '{field}' entry must be a JSON object")
        name = item.get("name")
        arguments = item.get("arguments", {})
        problems = item.get("problems", [])
        if not isinstance(name, str) or not name:
            raise click.ClickException(f"{where}: '{field}' entry needs a non-empty 'name'")
        if not isinstance(arguments, dict):
            raise click.ClickException(f"{where}: '{field}' entry 'arguments' must be a JSON object")
        if not isinstance(problems, list):
            raise click.ClickException(f"{where}: '{field}' entry 'problems' must be a list")
        calls.append(ToolCall(name=name, arguments=arguments))
        invalid += bool(problems)
    return tuple(calls), invalid


def _malformed_count(value: object, where: str) -> int:
    """``malformed`` is the list of blocks that did not parse; a bare count is accepted too."""
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    raise click.ClickException(f"{where}: 'malformed' must be a list of strings or a count")


def _opt_bool(raw: Mapping[str, Any], key: str, where: str) -> bool:
    value = raw.get(key, False)
    if not isinstance(value, bool):
        raise click.ClickException(f"{where}: '{key}' must be a boolean")
    return value


def _opt_str(raw: Mapping[str, Any], key: str, where: str) -> str | None:
    value = raw.get(key)
    if value is not None and not isinstance(value, str):
        raise click.ClickException(f"{where}: '{key}' must be a string")
    return value


def _opt_int(raw: Mapping[str, Any], key: str, where: str) -> int | None:
    value = raw.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise click.ClickException(f"{where}: '{key}' must be an integer")
    return value


def _describe_calls(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "(no call)"
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = f"{item.get('name')} {json.dumps(item.get('arguments', {}), ensure_ascii=False)}"
        problems = item.get("problems")
        if isinstance(problems, list) and problems:
            text += f"  [{'; '.join(map(str, problems))}]"
        parts.append(text)
    return " | ".join(parts)
