"""Replay dialogues through a model, one assistant turn at a time, and record what it produced.

Teacher-forced: every assistant turn is generated from the reference history up
to that turn, rendered by the same per-turn tokenization training used, so each
decision is scored on its own and a wrong call early in a dialogue does not
poison the turns after it. What this does not measure is how the model recovers
from its own mistakes; for that, drive it through a tool simulator.

Greedy decoding: the point is a reproducible score, not a sample of the model's
distribution. Sampling parameters are a serving concern.

An adapter directory is loaded on top of its base model without merging, so what
is scored is exactly what was trained. Point ``model`` at a merged checkpoint to
score the artefact that will be served.

Records are written as each batch finishes, so a replay that dies leaves what it
already scored. That puts the file in generation order — longest prompts first —
rather than dialogue order; the scorer reads it either way.
"""

import json
import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedTokenizerBase

from ..config import DEFAULT_BATCH_SIZE, DEFAULT_MAX_NEW_TOKENS
from ..masking import TemplateCompatibilityError
from ..metrics import TurnRecord
from ..parsing import TOOL_CALL_OPEN, check_tool_calls, parse_assistant_output
from ..schema import Dialogue, ToolCall
from .common import ensure_pad_token, git_state, resolve_device, resolve_dtype, tokenize_dialogues, versions

__all__ = ["PredictionError", "PredictionRun", "PredictionSettings", "run_predictions"]

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """The model or tokenizer cannot be replayed the way the scorer expects."""


class _Generative(Protocol):
    """What replay needs from a model; both a plain causal LM and a peft-wrapped one provide it."""

    generation_config: GenerationConfig | None

    def eval(self) -> object: ...

    def to(self, device: str) -> "_Generative": ...

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: GenerationConfig,
        use_model_defaults: bool,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionSettings:
    model: str
    """HF id, merged checkpoint directory or adapter directory."""
    max_seq_length: int
    chat_template_kwargs: Mapping[str, Any]
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = "auto"
    precision: str = "auto"


@dataclass(frozen=True, slots=True, kw_only=True)
class _Generated:
    text: str
    tokens: int
    """Decoding steps the model took, its stop token included."""
    truncated: bool
    """Generation stopped at max_new_tokens rather than on an end-of-turn token."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionRun:
    turns: tuple[TurnRecord, ...]
    dialogues: int
    generated_tokens: int
    seconds: float


def run_predictions(
    settings: PredictionSettings,
    dialogues: Sequence[Dialogue],
    *,
    out_path: Path,
    data_path: Path | None = None,
    limit: int | None = None,
) -> PredictionRun:
    """Generate every assistant turn of ``dialogues``, write one JSONL record per turn to ``out_path``
    and a ``.meta.json`` next to it, and return the turns in scorable form.

    ``data_path`` and ``limit`` say where the dialogues came from; they are recorded, not read.
    """
    device = resolve_device(settings.device)
    dtype = resolve_dtype(settings.precision, device)
    logger.info("predicting on %s in %s", device, str(dtype).removeprefix("torch."))

    tokenizer = _load_tokenizer(settings.model)
    ensure_pad_token(tokenizer)
    pad_token_id = _token_id(tokenizer.pad_token_id)
    if pad_token_id is None:
        raise PredictionError(f"{settings.model}: tokenizer has neither a pad nor an eos token to pad batches with")
    _check_tool_call_format(tokenizer, settings.chat_template_kwargs)
    tokenized = tokenize_dialogues(
        tokenizer,
        dialogues,
        max_seq_length=settings.max_seq_length,
        chat_template_kwargs=settings.chat_template_kwargs,
        name="predict",
    )
    sources = [(dialogue, example) for dialogue, examples in tokenized for example in examples]
    prompts = [list(example.input_ids[: example.prompt_tokens]) for _dialogue, example in sources]
    logger.info("%d dialogues -> %d turns to generate", len(tokenized), len(prompts))
    unchecked = sum(1 for dialogue, _examples in tokenized if not dialogue.tools)
    if unchecked:
        logger.warning(
            "%d of %d dialogues declare no tool schemas; their calls are recorded unchecked",
            unchecked,
            len(tokenized),
        )

    model = _load_model(settings.model, dtype=dtype, device=device)
    stop_ids = _stop_token_ids(model, tokenizer)
    generation = GenerationConfig(
        max_new_tokens=settings.max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
        eos_token_id=stop_ids,
    )
    turns: list[TurnRecord] = []
    generated_tokens = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as stream:
        for index, item in _generate_all(
            model, tokenizer, prompts, generation, batch_size=settings.batch_size, device=device, stop_ids=stop_ids
        ):
            dialogue, example = sources[index]
            message = dialogue.messages[example.turn_index]
            parsed = parse_assistant_output(item.text)
            problems = check_tool_calls(parsed.tool_calls, dialogue.tools)
            record = {
                "dialogue_id": dialogue.dialogue_id,
                "turn_index": example.turn_index,
                "expected": [_call_json(call) for call in message.tool_calls],
                "expected_content": message.content,
                "predicted": [
                    {**_call_json(call), "problems": list(found)}
                    for call, found in zip(parsed.tool_calls, problems, strict=True)
                ],
                "predicted_content": parsed.content,
                "malformed": list(parsed.malformed),
                "truncated": item.truncated,
                "raw_output": item.text,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            turns.append(
                TurnRecord(
                    expected=tuple(ToolCall(name=call.name, arguments=call.arguments) for call in message.tool_calls),
                    predicted=parsed.tool_calls,
                    dialogue_id=dialogue.dialogue_id,
                    turn_index=example.turn_index,
                    malformed=len(parsed.malformed),
                    invalid=sum(1 for found in problems if found),
                    truncated=item.truncated,
                )
            )
            generated_tokens += item.tokens
    seconds = time.perf_counter() - started

    run = PredictionRun(
        turns=tuple(turns), dialogues=len(tokenized), generated_tokens=generated_tokens, seconds=seconds
    )
    _write_meta(out_path, settings, run, device=device, dtype=dtype, data_path=data_path, limit=limit)
    return run


def _check_tool_call_format(tokenizer: PreTrainedTokenizerBase, chat_template_kwargs: Mapping[str, Any]) -> None:
    """The parser reads Hermes-style blocks; a template that renders calls any other way is refused."""
    probe = [
        {"role": "user", "content": "probe"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "probe_tool", "arguments": {"key": "value"}}}],
        },
    ]
    rendered = tokenizer.apply_chat_template(cast("list[Any]", probe), tokenize=False, **chat_template_kwargs)
    if not isinstance(rendered, str) or TOOL_CALL_OPEN not in rendered:
        raise TemplateCompatibilityError(
            f"chat template does not render tool calls as {TOOL_CALL_OPEN} blocks; predict only parses the "
            "Hermes/Qwen format"
        )


def _load_tokenizer(model: str) -> PreTrainedTokenizerBase:
    try:
        return AutoTokenizer.from_pretrained(model)
    except (OSError, ValueError) as error:
        raise PredictionError(f"{model}: no tokenizer to load ({error})") from error


def _load_model(path: str, *, dtype: torch.dtype, device: str) -> _Generative:
    try:
        if (Path(path) / "adapter_config.json").is_file():
            from peft import AutoPeftModelForCausalLM

            loaded: object = AutoPeftModelForCausalLM.from_pretrained(path, dtype=dtype)
        else:
            loaded = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except (OSError, ValueError) as error:
        raise PredictionError(f"{path}: no model to load ({error})") from error
    model = cast("_Generative", loaded).to(device)
    model.eval()
    return model


def _base_model(path: str) -> str | None:
    """What an adapter sits on; None for a plain checkpoint or an HF id."""
    config = Path(path) / "adapter_config.json"
    if not config.is_file():
        return None
    try:
        raw = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    base = raw.get("base_model_name_or_path") if isinstance(raw, dict) else None
    return base if isinstance(base, str) else None


def _stop_token_ids(model: _Generative, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """The model's own end-of-turn ids (Qwen3 lists two), else the tokenizer's eos."""
    configured = model.generation_config.eos_token_id if model.generation_config is not None else None
    if isinstance(configured, int):
        return [configured]
    if isinstance(configured, list) and configured:
        return [int(item) for item in configured]
    eos_token_id = _token_id(tokenizer.eos_token_id)
    if eos_token_id is None:
        raise PredictionError(f"{tokenizer.name_or_path}: no end-of-turn token, generation could not stop")
    return [eos_token_id]


def _token_id(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _generate_all(
    model: _Generative,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[list[int]],
    generation: GenerationConfig,
    *,
    batch_size: int,
    device: str,
    stop_ids: Sequence[int],
) -> Iterator[tuple[int, _Generated]]:
    """Greedy continuations, each yielded with its prompt index as its batch finishes.

    Longest prompts go first so memory problems show early.
    """
    order = sorted(range(len(prompts)), key=lambda index: -len(prompts[index]))
    batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
    for number, batch in enumerate(batches, start=1):
        started = time.perf_counter()
        rows = [prompts[index] for index in batch]
        items = _generate_batch(model, tokenizer, rows, generation, device=device, stop_ids=stop_ids)
        logger.info("batch %d/%d: %d turns in %.1f s", number, len(batches), len(batch), time.perf_counter() - started)
        yield from zip(batch, items, strict=True)


def _generate_batch(
    model: _Generative,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[list[int]],
    generation: GenerationConfig,
    *,
    device: str,
    stop_ids: Sequence[int],
) -> list[_Generated]:
    pad = _token_id(tokenizer.pad_token_id)
    assert pad is not None  # checked in run_predictions
    width = max(len(prompt) for prompt in prompts)
    # Left padding: generation continues from the last position of every row.
    input_ids = torch.full((len(prompts), width), pad, dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        input_ids[row, width - len(prompt) :] = torch.tensor(prompt, dtype=torch.long)
        attention_mask[row, width - len(prompt) :] = 1
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            generation_config=generation,
            # Otherwise transformers overrides do_sample=False with the checkpoint's sampling defaults.
            use_model_defaults=False,
        )
    continuations = generated[:, width:].tolist()
    stop = set(stop_ids)
    items: list[_Generated] = []
    for ids in continuations:
        # A row ends at its first stop token; whatever follows pads the batch. No stop token means
        # it ran out of budget. The stop token and the padding are special; <tool_call> is not.
        finished = next((position + 1 for position, token in enumerate(ids) if token in stop), None)
        items.append(
            _Generated(
                text=tokenizer.decode(ids, skip_special_tokens=True),
                tokens=len(ids) if finished is None else finished,
                truncated=finished is None,
            )
        )
    return items


def _call_json(call: ToolCall) -> dict[str, Any]:
    return {"name": call.name, "arguments": call.arguments}


def _write_meta(
    out_path: Path,
    settings: PredictionSettings,
    run: PredictionRun,
    *,
    device: str,
    dtype: torch.dtype,
    data_path: Path | None,
    limit: int | None,
) -> None:
    meta: dict[str, Any] = {
        "model": settings.model,
        "base_model": _base_model(settings.model),
        "data": {
            "path": str(data_path) if data_path is not None else None,
            "limit": limit,
            "dialogues": run.dialogues,
            "turns": len(run.turns),
        },
        "decoding": {"greedy": True, "max_new_tokens": settings.max_new_tokens, "batch_size": settings.batch_size},
        "chat_template_kwargs": dict(settings.chat_template_kwargs),
        "max_seq_length": settings.max_seq_length,
        "resolved": {"device": device, "dtype": str(dtype).removeprefix("torch.")},
        "generated_tokens": run.generated_tokens,
        "seconds": round(run.seconds, 1),
        "versions": versions(),
        "git": git_state(),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
