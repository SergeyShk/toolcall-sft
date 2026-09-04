"""Dataset file operations: JSONL loading, deduplication, deterministic splitting."""

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .schema import DatasetError, Dialogue, Role, content_fingerprint, dialogue_from_json, dialogue_to_json

__all__ = [
    "DatasetSplit",
    "DatasetStats",
    "dataset_stats",
    "dedup_dialogues",
    "load_dialogues",
    "split_dialogues",
    "write_dialogues",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetSplit:
    train: tuple[Dialogue, ...]
    evaluation: tuple[Dialogue, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetStats:
    dialogues: int
    messages: int
    assistant_messages: int
    tool_call_messages: int
    tool_call_counts: tuple[tuple[str, int], ...]


def load_dialogues(path: Path) -> tuple[Dialogue, ...]:
    dialogues: list[Dialogue] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise DatasetError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(raw, dict):
                raise DatasetError(f"{path}:{line_number}: each line must be a JSON object")
            try:
                dialogue = dialogue_from_json(raw)
            except DatasetError as error:
                raise DatasetError(f"{path}:{line_number}: {error}") from error
            if dialogue.dialogue_id in seen_ids:
                raise DatasetError(f"{path}:{line_number}: duplicate dialogue_id {dialogue.dialogue_id!r}")
            seen_ids.add(dialogue.dialogue_id)
            dialogues.append(dialogue)
    if not dialogues:
        raise DatasetError(f"{path}: no dialogues found")
    return tuple(dialogues)


def write_dialogues(path: Path, dialogues: Sequence[Dialogue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for dialogue in dialogues:
            stream.write(json.dumps(dialogue_to_json(dialogue), ensure_ascii=False) + "\n")


def dedup_dialogues(dialogues: Sequence[Dialogue]) -> tuple[Dialogue, ...]:
    """Content-level deduplication: dialogues with equal fingerprints collapse to the first one."""
    seen: set[str] = set()
    unique: list[Dialogue] = []
    for dialogue in dialogues:
        fingerprint = content_fingerprint(dialogue)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(dialogue)
    return tuple(unique)


def split_dialogues(dialogues: Sequence[Dialogue], *, eval_fraction: float, salt: str) -> DatasetSplit:
    """Deterministic content-hash split: a dialogue lands on the same side for a given salt
    regardless of file order, so regenerating the data keeps train/eval stable."""
    if not 0.0 < eval_fraction < 1.0:
        raise DatasetError(f"eval_fraction must be in (0, 1), got {eval_fraction}")
    train: list[Dialogue] = []
    evaluation: list[Dialogue] = []
    for dialogue in dialogues:
        digest = hashlib.sha256(f"{salt}:{content_fingerprint(dialogue)}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        (evaluation if bucket < eval_fraction else train).append(dialogue)
    if not train or not evaluation:
        raise DatasetError(
            f"split produced an empty side (train={len(train)}, eval={len(evaluation)}); "
            "adjust eval_fraction or grow the dataset"
        )
    return DatasetSplit(train=tuple(train), evaluation=tuple(evaluation))


def dataset_stats(dialogues: Sequence[Dialogue]) -> DatasetStats:
    tool_counter: Counter[str] = Counter()
    messages = 0
    assistant_messages = 0
    tool_call_messages = 0
    for dialogue in dialogues:
        messages += len(dialogue.messages)
        for message in dialogue.messages:
            if message.role is not Role.ASSISTANT:
                continue
            assistant_messages += 1
            if message.tool_calls:
                tool_call_messages += 1
                tool_counter.update(call.name for call in message.tool_calls)
    ordered = tuple(sorted(tool_counter.items(), key=lambda item: (-item[1], item[0])))
    return DatasetStats(
        dialogues=len(dialogues),
        messages=messages,
        assistant_messages=assistant_messages,
        tool_call_messages=tool_call_messages,
        tool_call_counts=ordered,
    )
