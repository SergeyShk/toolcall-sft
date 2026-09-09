# Hardware notes

Measured numbers for the default config, the memory levers in the order to reach for them, and
why the default model is smaller than the one you would ship.

## Choosing the size

The default is **Qwen3-0.6B**, smaller than you would ship. The first things you debug are the data
pipeline, the loss masking and the metrics, and all three misbehave identically at 0.6B and at 8B;
the small one just tells you sooner.

**This dataset cannot rank models.** It is template-generated and low-entropy, so once a model has
the sentence shapes there is nothing left to learn and loss saturates for everyone — two thirds of
its held-out tool calls appear verbatim in the training set, measured in
[DATASET.md](DATASET.md#keeping-the-split-honest). It is a good benchmark for the pipeline and a
useless one for choosing what to ship; for that you need real dialogues and tool-call metrics.

Iterate at 0.6B, step up to 1.7B or 4B once the harness is trustworthy and the question has changed
from "does this train" to "are the arguments right". A larger model costs less extra wall clock than
its parameter count suggests: model loading, the optimizer, evaluation and MPS launch overhead do
not shrink with the model.

## What the default run costs

Apple M3 Pro, 36 GB. Qwen3-0.6B, LoRA r=16, `max_seq_length: 2048`, micro-batch 1 with 8-step
accumulation, gradient checkpointing on, 10.1M trainable parameters (1.66% of the model).

| | |
|---|---|
| 300 dialogues → per-turn examples | 975 (267 train / 33 eval dialogues) |
| Optimizer steps, 2 epochs | 218 |
| Seconds per step | ~15.8 |
| Training wall clock | 58m48s |
| Held-out loss after epoch 1 / 2 | 0.097 / 0.064 |
| Evaluation time, epoch 1 / 2 | 48 s / 46 s |

A 1000-dialogue run extrapolates to roughly three hours. A CUDA GPU is an order of magnitude
faster. MPS wall-clock numbers vary by 20% between sessions on the same machine; ratios measured
back to back are trustworthy, absolute times are not.

## Replaying the eval split

`tcsft-train predict` on the 33-dialogue split (106 assistant turns), greedy, batch 4, bf16 on MPS,
Qwen3-0.6B:

| Model | Wall clock | Tokens generated |
|---|---|---|
| Adapter on top of the base, unmerged | 79 s | 2702 |
| Merged checkpoint | 63 s | 2708 |
| Untuned base | 63 s | 2455 |

Same score for the adapter and its merge (96.2% turn accuracy, identical wrong turns), so replaying
the adapter straight after training is enough; merge for serving. Most of the time is the prompt:
every turn re-encodes the system prompt and the tool schemas, and the last turn of a dialogue
carries the whole conversation. Batch size trades memory for speed the same way as in training,
and 4 is comfortable for 0.6B on this machine. The prompts are the trainer's prompts token for
token, so a template mismatch shows up here as a number before it shows up in production.

## Memory on MPS

**Leave `gradient_checkpointing: true`.** PyTorch's MPS allocator may reserve well beyond physical
memory (1.7x the recommended maximum by default) and does not release what it has cached. Measured
on a 1.7B run with checkpointing off: the footprint grew to 42 GB on this 36 GB machine, filled
16 GB of swap, and decayed from 20 s/step to over 200 s/step. It never raised an error; it just got
slower until it was unusable.

**Evaluation has its own lever.** Each evaluated position materializes logits over the whole
vocabulary (152k floats for Qwen3), so an eval batch of 8 examples of ~900 tokens is several
gigabytes training never allocates. `training.eval_batch_size` therefore defaults to the train
batch size instead of transformers' 8; at 1, the run above evaluated its 106 examples in 48 s and
46 s with no drift between epochs. Raising it is the first thing to blame if evaluation slows down
from one epoch to the next.

Process memory is a poor guide to what a run needs; the footprint expands to fill what is
available. The levers, in order:

1. `training.gradient_checkpointing: true`. The biggest saving. Costs roughly 30% throughput on
   hardware that has memory to spare.
2. `training.per_device_batch_size: 1` with a higher `gradient_accumulation_steps`; keep
   `eval_batch_size` in step. Already the case in the MPS config.
3. Lower `dataset.max_seq_length` to what `tcsft stats` reports. Activation memory is linear in it,
   and 2048 is generous for the example scenario's ~900-token longest examples.
4. A smaller base model, if you have stepped up from the default.

On CUDA, `configs/cuda_qlora.yaml` adds 4-bit base weights and fused cross-entropy, which is what
makes an 8B model fit a mid-range card. Both are CUDA-only and the trainer refuses them elsewhere.
That config has not been run end to end by the author.
