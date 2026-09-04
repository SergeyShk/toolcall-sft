"""LoRA / QLoRA supervised fine-tuning on chat-template-exact examples.

Runs on CUDA, Apple Silicon (MPS) and CPU. The differences between them are all
resolved in one place, ``_resolve_device`` / ``_resolve_dtype``, because getting
them wrong fails deep inside the trainer with an unhelpful message: 4-bit
quantization and the Liger kernels are CUDA-only, and ``TrainingArguments``'
``bf16`` flag asks for CUDA-style autocast that MPS does not provide. On MPS the
base model is instead loaded in bf16 directly and the LoRA parameters are kept in
fp32, which is what keeps a 1.7B tune inside a laptop's memory without training
the adapters in half precision.
"""

import logging
from pathlib import Path

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

from ..config import ConfigError, ExperimentConfig
from ..dataset import load_dialogues
from ..masking import LABEL_IGNORE_INDEX, tokenize_dialogue
from ..schema import DatasetError, Dialogue

__all__ = ["run_sft"]

logger = logging.getLogger(__name__)


def run_sft(config: ExperimentConfig) -> Path:
    """Train a LoRA adapter and return the directory it is saved to."""
    device = _resolve_device(config.training.device)
    dtype = _resolve_dtype(config.training.precision, device)
    _check_supported(config, device)
    logger.info("training on %s in %s", device, str(dtype).removeprefix("torch."))

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = _build_dataset(tokenizer, load_dialogues(config.dataset.train_path), config, name="train")
    eval_dataset = None
    if config.dataset.eval_path is not None:
        eval_dataset = _build_dataset(tokenizer, load_dialogues(config.dataset.eval_path), config, name="eval")

    model = _load_model(config, dtype)
    if config.training.gradient_checkpointing:
        model.enable_input_require_grads()
    peft_model = get_peft_model(model, _lora_config(config))
    assert isinstance(peft_model, PeftModel)  # guaranteed while mixed=False stays the default
    if dtype is not torch.float32:
        # Half-precision adapters lose small updates to rounding. The base weights stay
        # in bf16; peft casts activations into the adapter's dtype and back on its own.
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
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        # Fused linear cross-entropy: the loss never materializes the fp32 logits
        # tensor, which over a large vocabulary dominates memory at long context.
        # CUDA only — see _check_supported.
        use_liger_kernel=config.training.use_liger_kernel,
        seed=config.training.seed,
        report_to=list(config.tracking.report_to),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=peft_model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=LABEL_IGNORE_INDEX),
    )
    trainer.train()

    adapter_dir = config.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("adapter saved to %s", adapter_dir)
    return adapter_dir


def _build_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dialogues: tuple[Dialogue, ...],
    config: ExperimentConfig,
    *,
    name: str,
) -> Dataset:
    max_seq_length = config.dataset.max_seq_length
    rows: list[dict[str, list[int]]] = []
    dropped = 0
    for dialogue in dialogues:
        example = tokenize_dialogue(tokenizer, dialogue)
        if len(example.input_ids) > max_seq_length:
            dropped += 1
            continue
        rows.append(
            {
                "input_ids": list(example.input_ids),
                "attention_mask": [1] * len(example.input_ids),
                "labels": list(example.labels),
            }
        )
    if dropped:
        logger.warning(
            "%s: dropped %d of %d dialogues longer than max_seq_length=%d",
            name,
            dropped,
            len(dialogues),
            max_seq_length,
        )
    if not rows:
        raise DatasetError(f"{name}: every dialogue exceeded max_seq_length={max_seq_length}")
    logger.info("%s: %d dialogues tokenized", name, len(rows))
    return Dataset.from_list(rows)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(precision: str, device: str) -> torch.dtype:
    if precision == "auto":
        # CPU bf16 matmuls fall back to slow kernels; accelerators do bf16 natively.
        precision = "fp32" if device == "cpu" else "bf16"
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def _check_supported(config: ExperimentConfig, device: str) -> None:
    """Fail on a CUDA-only option before the trainer does, with the reason."""
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
        model = prepare_model_for_kbit_training(model)
    return model


def _lora_config(config: ExperimentConfig) -> LoraConfig:
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
    )
