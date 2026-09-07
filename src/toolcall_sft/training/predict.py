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
"""

import json
import logging
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import peft
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedTokenizerBase

from ..masking import TemplateCompatibilityError
from ..metrics import TurnRecord
from ..parsing import TOOL_CALL_OPEN, check_tool_calls, parse_assistant_output
from ..schema import Dialogue, ToolCall
from .common import resolve_device, resolve_dtype, tokenize_dialogues

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
    max_new_tokens: int = 256
    batch_size: int = 4
    device: str = "auto"
    precision: str = "auto"


@dataclass(frozen=True, slots=True, kw_only=True)
class _Generated:
    text: str
    truncated: bool
    """Generation stopped at max_new_tokens rather than on an end-of-turn token."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionRun:
    turns: tuple[TurnRecord, ...]
    dialogues: int
    generated_tokens: int
    seconds: float


def run_predictions(settings: PredictionSettings, dialogues: Sequence[Dialogue], *, out_path: Path) -> PredictionRun:
    """Generate every assistant turn of ``dialogues``, write one JSONL record per turn to ``out_path``
    and a ``.meta.json`` next to it, and return the turns in scorable form."""
    device = resolve_device(settings.device)
    dtype = resolve_dtype(settings.precision, device)
    logger.info("predicting on %s in %s", device, str(dtype).removeprefix("torch."))

    tokenizer = AutoTokenizer.from_pretrained(settings.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = _token_id(tokenizer.pad_token_id)
    if pad_token_id is None:
        raise PredictionError(f"{settings.model}: tokenizer has neither a pad nor an eos token to pad batches with")
    _check_tool_call_format(tokenizer)
    tokenized = tokenize_dialogues(
        tokenizer,
        dialogues,
        max_seq_length=settings.max_seq_length,
        chat_template_kwargs=settings.chat_template_kwargs,
        name="predict",
    )
    prompts = [
        list(example.input_ids[: example.prompt_tokens]) for _dialogue, examples in tokenized for example in examples
    ]
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
    started = time.perf_counter()
    outputs = _generate_all(
        model, tokenizer, prompts, generation, batch_size=settings.batch_size, device=device, stop_ids=stop_ids
    )
    seconds = time.perf_counter() - started

    records: list[dict[str, Any]] = []
    turns: list[TurnRecord] = []
    position = 0
    for dialogue, examples in tokenized:
        for example in examples:
            generated = outputs[position]
            position += 1
            message = dialogue.messages[example.turn_index]
            parsed = parse_assistant_output(generated.text)
            problems = check_tool_calls(parsed.tool_calls, dialogue.tools)
            records.append(
                {
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
                    "truncated": generated.truncated,
                    "raw_output": generated.text,
                }
            )
            turns.append(
                TurnRecord(
                    expected=tuple(ToolCall(name=call.name, arguments=call.arguments) for call in message.tool_calls),
                    predicted=parsed.tool_calls,
                    dialogue_id=dialogue.dialogue_id,
                    turn_index=example.turn_index,
                    malformed=len(parsed.malformed),
                    invalid=sum(1 for found in problems if found),
                    truncated=generated.truncated,
                )
            )
    generated_tokens = sum(len(tokenizer(item.text, add_special_tokens=False)["input_ids"]) for item in outputs)
    run = PredictionRun(
        turns=tuple(turns), dialogues=len(tokenized), generated_tokens=generated_tokens, seconds=seconds
    )
    _write(out_path, records, settings, run, device=device, dtype=dtype)
    return run


def _check_tool_call_format(tokenizer: PreTrainedTokenizerBase) -> None:
    """The parser reads Hermes-style blocks; a template that renders calls any other way is refused."""
    probe = [
        {"role": "user", "content": "probe"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "probe_tool", "arguments": {"key": "value"}}}],
        },
    ]
    rendered = tokenizer.apply_chat_template(cast("list[Any]", probe), tokenize=False)
    if not isinstance(rendered, str) or TOOL_CALL_OPEN not in rendered:
        raise TemplateCompatibilityError(
            f"chat template does not render tool calls as {TOOL_CALL_OPEN} blocks; predict only parses the "
            "Hermes/Qwen format"
        )


def _load_model(path: str, *, dtype: torch.dtype, device: str) -> _Generative:
    if (Path(path) / "adapter_config.json").is_file():
        from peft import AutoPeftModelForCausalLM

        loaded: object = AutoPeftModelForCausalLM.from_pretrained(path, dtype=dtype)
    else:
        loaded = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    model = cast("_Generative", loaded).to(device)
    model.eval()
    return model


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
) -> list[_Generated]:
    """Greedy continuations in prompt order. Longest prompts go first so memory problems show early."""
    order = sorted(range(len(prompts)), key=lambda index: -len(prompts[index]))
    outputs = [_Generated(text="", truncated=False)] * len(prompts)
    batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
    for number, batch in enumerate(batches, start=1):
        started = time.perf_counter()
        rows = [prompts[index] for index in batch]
        items = _generate_batch(model, tokenizer, rows, generation, device=device, stop_ids=stop_ids)
        for index, item in zip(batch, items, strict=True):
            outputs[index] = item
        logger.info("batch %d/%d: %d turns in %.1f s", number, len(batches), len(batch), time.perf_counter() - started)
    return outputs


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
    # The stop token and any padding after it are special tokens; <tool_call> and <think> are not.
    # A row without a stop token in it ran out of budget instead of finishing.
    return [
        _Generated(
            text=tokenizer.decode(ids, skip_special_tokens=True),
            truncated=not any(token in stop for token in ids),
        )
        for ids in continuations
    ]


def _call_json(call: ToolCall) -> dict[str, Any]:
    return {"name": call.name, "arguments": call.arguments}


def _write(
    out_path: Path,
    records: list[dict[str, Any]],
    settings: PredictionSettings,
    run: PredictionRun,
    *,
    device: str,
    dtype: torch.dtype,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta: dict[str, Any] = {
        "model": settings.model,
        "decoding": {"greedy": True, "max_new_tokens": settings.max_new_tokens, "batch_size": settings.batch_size},
        "chat_template_kwargs": dict(settings.chat_template_kwargs),
        "max_seq_length": settings.max_seq_length,
        "resolved": {"device": device, "dtype": str(dtype).removeprefix("torch.")},
        "dialogues": run.dialogues,
        "turns": len(run.turns),
        "generated_tokens": run.generated_tokens,
        "seconds": round(run.seconds, 1),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "git": _git_state(),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit.strip(), "dirty": bool(status.strip())}
