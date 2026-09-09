"""LoRA / QLoRA supervised fine-tuning on per-turn, chat-template-exact examples.

Runs on CUDA, MPS and CPU. Device differences live in ``common.resolve_device``
and ``common.resolve_dtype``: 4-bit and Liger are CUDA-only, and ``bf16=True`` means CUDA
autocast, so on MPS the base model is loaded in bf16 and the LoRA parameters are
kept in fp32.

Each run writes ``config.yaml``, ``run.json`` and ``example.txt`` to output_dir
before training starts.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from ..config import ConfigError, ExperimentConfig, flatten_for_logging
from ..dataset import load_dialogues
from ..masking import LABEL_IGNORE_INDEX, MaskedExample, render_example
from ..schema import Dialogue
from .common import ensure_pad_token, git_state, resolve_device, resolve_dtype, tokenize_dialogues, versions

__all__ = ["run_sft"]

logger = logging.getLogger(__name__)


def run_sft(config: ExperimentConfig, *, config_path: Path | None = None) -> Path:
    """Train a LoRA adapter and return the directory it is saved to."""
    device = resolve_device(config.training.device)
    dtype = resolve_dtype(config.training.precision, device)
    _check_supported(config, device)
    logger.info("training on %s in %s", device, str(dtype).removeprefix("torch."))

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    ensure_pad_token(tokenizer)

    train_dialogues = load_dialogues(config.dataset.train_path)
    train_examples = _tokenize(tokenizer, train_dialogues, config, name="train")
    eval_examples: tuple[MaskedExample, ...] | None = None
    eval_dialogues: tuple[Dialogue, ...] = ()
    if config.dataset.eval_path is not None:
        eval_dialogues = load_dialogues(config.dataset.eval_path)
        eval_examples = _tokenize(tokenizer, eval_dialogues, config, name="eval")

    _write_run_record(
        config,
        config_path,
        device=device,
        dtype=dtype,
        counts={
            "train_dialogues": len(train_dialogues),
            "train_examples": len(train_examples),
            "eval_dialogues": len(eval_dialogues),
            "eval_examples": len(eval_examples) if eval_examples is not None else 0,
        },
        example_text=render_example(tokenizer, train_examples[-1]),
    )

    model = _load_model(config, dtype)
    if config.training.gradient_checkpointing:
        model.enable_input_require_grads()
    peft_model = get_peft_model(model, _lora_config(config))
    assert isinstance(peft_model, PeftModel)  # guaranteed while mixed=False stays the default
    if dtype is not torch.float32:
        # Half-precision adapters lose small updates to rounding; peft casts activations
        # to the adapter dtype and back.
        for parameter in peft_model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(torch.float32)
    peft_model.print_trainable_parameters()

    arguments = TrainingArguments(
        output_dir=str(config.output_dir),
        run_name=config.run_name,
        num_train_epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.per_device_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler,
        logging_steps=config.training.logging_steps,
        bf16=device == "cuda" and dtype is torch.bfloat16,
        fp16=device == "cuda" and dtype is torch.float16,
        use_cpu=device == "cpu",
        dataloader_pin_memory=device == "cuda",
        gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.training.gradient_checkpointing else None,
        eval_strategy="epoch" if eval_examples is not None else "no",
        save_strategy="epoch",
        # Fused linear cross-entropy; CUDA only, see _check_supported.
        use_liger_kernel=config.training.use_liger_kernel,
        seed=config.training.seed,
        report_to=list(config.tracking.report_to),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=peft_model,
        args=arguments,
        train_dataset=_to_dataset(train_examples),
        eval_dataset=_to_dataset(eval_examples) if eval_examples is not None else None,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=LABEL_IGNORE_INDEX),
    )
    trainer.train()

    adapter_dir = config.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("adapter saved to %s", adapter_dir)
    return adapter_dir


def _tokenize(
    tokenizer: PreTrainedTokenizerBase,
    dialogues: tuple[Dialogue, ...],
    config: ExperimentConfig,
    *,
    name: str,
) -> tuple[MaskedExample, ...]:
    tokenized = tokenize_dialogues(
        tokenizer,
        dialogues,
        max_seq_length=config.dataset.max_seq_length,
        chat_template_kwargs=config.dataset.chat_template_kwargs,
        name=name,
    )
    examples = tuple(example for _dialogue, turns in tokenized for example in turns)
    logger.info("%s: %d dialogues -> %d examples (one per assistant turn)", name, len(dialogues), len(examples))
    return examples


def _to_dataset(examples: tuple[MaskedExample, ...]) -> Dataset:
    rows: list[dict[str, list[int]]] = [
        {
            "input_ids": list(example.input_ids),
            "attention_mask": [1] * len(example.input_ids),
            "labels": list(example.labels),
        }
        for example in examples
    ]
    return Dataset.from_list(rows)


def _check_supported(config: ExperimentConfig, device: str) -> None:
    """Fail on a CUDA-only option before the trainer does."""
    if config.training.load_in_4bit and device != "cuda":
        raise ConfigError(
            f"training.load_in_4bit needs bitsandbytes, which is CUDA-only (resolved device: {device}). "
            "Drop it and use a smaller base model instead."
        )
    if config.training.use_liger_kernel and device != "cuda":
        raise ConfigError(f"training.use_liger_kernel is CUDA-only (resolved device: {device}). Set it to false.")


def _load_model(config: ExperimentConfig, dtype: torch.dtype) -> PreTrainedModel:
    quantization_config = None
    if config.training.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        dtype=dtype,
        quantization_config=quantization_config,
    )
    if config.training.load_in_4bit:
        # Keep the config in charge of checkpointing; peft would otherwise enable the reentrant variant.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    return model


def _lora_config(config: ExperimentConfig) -> LoraConfig:
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
    )


def _write_run_record(
    config: ExperimentConfig,
    config_path: Path | None,
    *,
    device: str,
    dtype: torch.dtype,
    counts: dict[str, int],
    example_text: str,
) -> None:
    """Enough to explain the run later, written before anything can crash."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        shutil.copyfile(config_path, config.output_dir / "config.yaml")
    record: dict[str, Any] = {
        "run_name": config.run_name,
        "config": flatten_for_logging(config),
        "resolved": {"device": device, "dtype": str(dtype).removeprefix("torch.")},
        "dataset": counts,
        "versions": versions(),
        "git": git_state(),
    }
    (config.output_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (config.output_dir / "example.txt").write_text(example_text, encoding="utf-8")
