# toolcall-sft

Fine-tune a small local model to drive a multi-turn **tool-calling** conversation — pick the right
tool, fill its arguments from what the user said and what earlier tools returned, and know when to
ask instead of act.

It runs on a laptop. The default config tunes Qwen3-0.6B with LoRA on Apple Silicon; the same config
picks up a CUDA GPU if there is one, and falls back to CPU if there is not. Step up a size or two
once the pipeline is trustworthy — see [Choosing the size](#choosing-the-size).

There is no data to find first: `tcsft generate` writes a synthetic dataset for the bundled example
scenario (send a payment to a saved payee), so a clone goes from `git clone` to a trained adapter in
one `make demo`.

## Why this rather than a training framework

Nothing here is magic, and that is the point. Two things go wrong in tool-calling fine-tunes and
both are invisible until the model is already bad:

- **Chat-template skew.** Training text is rendered by the base model's *own* chat template — the
  same one vLLM and Ollama use at inference. Tool calls are never serialized by hand.
- **Loss on the wrong tokens.** Loss is masked to assistant tokens only, located with sentinel
  probes rather than by string search, so it stays correct for position-dependent templates (Qwen3
  inserts think-blocks). A template that does not render assistant content verbatim fails loudly
  instead of training on a silently mangled target.

Everything else is `transformers.Trainer` + `peft` — about 1700 lines of pipeline, plus a
synthetic-data generator and an anonymizer you can delete once you have your own data.

## Documentation

- [docs/FINETUNING.md](docs/FINETUNING.md) — why and how to tune a model for tool calling: task
  framing, choosing a base model, method, evaluation, serving, and the traps in the order people hit
  them.
- [docs/DATASET.md](docs/DATASET.md) — how much data, which branches, how to keep the balance and
  the split honest, and how to bring your own dialogues.
- [CONTRIBUTING.md](CONTRIBUTING.md) — setup, conventions, scope.

## Quickstart

```bash
uv sync
make demo
```

`make demo` runs steps 1-4 below. The full loop, step by step:

```bash
# 1. A synthetic dataset for the example scenario (deterministic in --seed).
#    300 keeps the demo short; 1000-2000 is the real target — see docs/DATASET.md
uv run tcsft generate --out data/all.jsonl --count 300

# 2. Check it against the schema; look at token lengths
uv run tcsft validate data/all.jsonl
uv run tcsft stats data/all.jsonl

# 3. Deduplicate and split (content-hash based: a dialogue lands on the same
#    side regardless of file order)
uv run tcsft split data/all.jsonl --train-out data/train.jsonl --eval-out data/eval.jsonl

# 4. Train a LoRA adapter
uv run tcsft-train train --config configs/mac_mps.yaml

# 5. Merge the adapter into the base model for standalone serving
uv run tcsft-train merge \
  --adapter outputs/payments-qwen3-0.6b-lora/adapter \
  --output outputs/payments-qwen3-0.6b-lora/merged

# 6. Serve (OpenAI-compatible)
# vllm serve outputs/payments-qwen3-0.6b-lora/merged \
#   --enable-auto-tool-choice --tool-call-parser hermes
```

Score predicted tool calls against reference ones with `uv run tcsft evaluate data/predictions.jsonl`.

**How long it takes.** Measured on an Apple M3 Pro with the default config: `make demo` (300
dialogues, 2 epochs, 68 optimizer steps) took **24m44s**, ending at a held-out loss of 0.045. A
1000-dialogue run extrapolates to roughly 80 minutes. A CUDA GPU is an order of magnitude faster —
MPS is usable, not quick.

Treat MPS wall-clock numbers as approximate: the same config measured 18.4 and 21.8 seconds per step
in two different sessions, and evaluation inside this very run slowed 5x between the first epoch and
the second as the machine warmed up. Ratios measured back to back are trustworthy; absolute times
are not. Read [Hardware](#hardware) before changing the memory settings.

The dialogues carry no reasoning content, so a Qwen3 tune must be served in non-thinking mode
(`enable_thinking: false`) — the mode it was trained in.

## The example scenario

Three tools, defined in [scenario.py](src/toolcall_sft/scenario.py), and a system prompt of about
150 tokens kept in its own file, [system_prompt.txt](src/toolcall_sft/system_prompt.txt) — it is the
thing you edit most, the thing worth diffing between two runs, and the thing the serving side has to
receive verbatim.

| Tool | Role in the tune |
|---|---|
| `get_payees(name)` | a lookup whose result decides what happens next: no match, one match, or several |
| `create_payment(payee_id, amount, currency, reference?)` | a write that must not fire without confirmation, and that can come back declined |
| `escalate(reason)` | the way out for everything the model cannot do |

`escalate` carries more weight than it looks. Without a way out, a narrow model answers questions it
has no business answering — so the exit has to be a tool call it was trained to make, not a hope
that the prompt will hold.

The generator covers seven branches, and **about half of them never reach `create_payment`**:

| Branch | Share | Calls the write tool |
|---|---:|---|
| `happy_path` — named payee and amount, confirm, send | 30% | yes |
| `out_of_scope` — off-topic request, leaves through `escalate` | 16% | no |
| `missing_amount` — ask for the amount, then send | 13% | yes |
| `ambiguous_payee` — two namesakes, ask which | 12% | no |
| `payee_not_found` — lookup returns nothing | 11% | no |
| `cancelled` — customer changes their mind | 10% | no |
| `payment_declined` — the write is refused, and must be reported as refused | 8% | yes |

That balance is the part people get wrong with real data too. A model trained only on the happy
path learns that the write tool is the answer to everything; one never shown a refusal learns to
report a success it did not get.

## Dataset format

One dialogue per JSONL line. Tool calls are stored flat; the pipeline converts them to the
OpenAI-style nested form that chat templates expect.

```json
{
  "dialogue_id": "happy_path-00001",
  "messages": [
    {"role": "system", "content": "You are a payment assistant..."},
    {"role": "user", "content": "Send $450.00 to James Whitfield"},
    {"role": "assistant", "content": "", "tool_calls": [{"name": "get_payees", "arguments": {"name": "James Whitfield"}}]},
    {"role": "tool", "content": "[{\"payee_id\": \"pay_1c9f\", \"name\": \"James Whitfield\"}]"},
    {"role": "assistant", "content": "James Whitfield — send $450.00?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_payees",
        "description": "Look up the customer's saved payees by name. Returns every match, so an empty list means the payee is not saved and more than one match has to be resolved with the customer before paying.",
        "parameters": {
          "type": "object",
          "properties": {
            "name": {"type": "string", "description": "Full or partial payee name, as the customer said it."}
          },
          "required": ["name"]
        }
      }
    }
  ]
}
```

The `description` fields are not decoration and are not optional: the chat template renders them
into the training prompt, so they are literally what the model reads to decide which tool to call
and what to put in it. A toolset stripped of descriptions trains a different model than the one you
will serve. One tool is shown here in full — the committed sample has all three.

Validation enforces that a dialogue opens with `system`/`user`, ends with an `assistant` message
(the training target), that only `assistant` messages carry `tool_calls`, and that every `tool`
result answers a pending call.

Evaluation input for `tcsft evaluate` is JSONL of
`{"expected": [{"name", "arguments"}], "predicted": [...]}` per assistant turn.

Twenty generated dialogues covering all seven branches are committed at
[examples/dialogues.sample.jsonl](examples/dialogues.sample.jsonl) if you would rather read the real
thing than the excerpt above (`tcsft generate --out ... --count 20 --seed 42` reproduces it).

## Bringing your own data

Replace [scenario.py](src/toolcall_sft/scenario.py) with your prompt and tools, and produce JSONL in
the schema above from wherever your dialogues live. Nothing else in the package knows what a payment
is.

If those dialogues come from a real system, run them through the anonymizer **before** anything
else — training bakes data into weights:

```bash
uv run tcsft anonymize data/raw.jsonl --out data/anon.jsonl --term "Some Name"
```

It is deterministic pseudonymization: regex detectors for structured identifiers (emails,
international phone numbers, IBANs, card numbers, account numbers, UUIDs) plus names harvested from
tool payloads under name-like keys. The same original maps to the same fake everywhere, so an id
passed from a tool result into a later tool call stays consistent. Amounts are kept — they carry the
scenario's meaning.

The detectors are deliberately region-neutral. Locally-shaped identifiers — national ids, tax
numbers, postal codes, domestic bank codes — differ per country and are yours to add; they are
exactly what a generic list misses.

It is a mechanical pass, not a guarantee: a free-text name that appears in no tool payload survives
it. Review a sample, and feed known names in with `--term`. See
[anonymizer.py](src/toolcall_sft/anonymizer.py) for the full set of caveats.

## Experiment config

Two starting points:

- [configs/mac_mps.yaml](configs/mac_mps.yaml) — Qwen3-0.6B + LoRA, `device: auto`. The default.
- [configs/cuda_qlora.yaml](configs/cuda_qlora.yaml) — Qwen3-8B in 4-bit with fused cross-entropy
  and MLflow, for a single CUDA GPU. Needs `uv sync --group cuda`.

Required keys: `run_name`, `base_model`, `output_dir`, `dataset.train_path`,
`dataset.max_seq_length`, `lora.r`, `lora.alpha`, `training.epochs`, `training.learning_rate`,
`training.per_device_batch_size`, `training.gradient_accumulation_steps`. Everything else has a
default — [config.py](src/toolcall_sft/config.py) is the authority.

`training.device` is `auto | cuda | mps | cpu` and `training.precision` is `auto | bf16 | fp16 |
fp32`. `auto` resolves to CUDA, then MPS, then CPU, and to bf16 on an accelerator, fp32 on CPU. On
MPS the base weights are loaded in bf16 while the LoRA parameters are kept in fp32 — half-precision
adapters lose small updates to rounding.

`training.load_in_4bit` (QLoRA, bitsandbytes) and `training.use_liger_kernel` are **CUDA-only**;
asking for them on any other device is refused up front with the reason, rather than failing
somewhere inside the trainer.

Experiment tracking is opt-in. With no `tracking:` section nothing is reported anywhere; add
`report_to: ["mlflow"]` and an `mlflow_experiment` name to log params, the config artifact and
trainer metrics. MLflow is imported only if you ask for it.

## Choosing the size

The default is **Qwen3-0.6B**, which is smaller than you would ship. That is deliberate: the first
things you debug are the data pipeline, the loss masking and the metrics, and all three misbehave
identically at 0.6B and at 8B — the small one just tells you sooner.

Measured on an M3 Pro, same 300-dialogue corpus, one epoch, 34 optimizer steps:

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|
| Trainable parameters | 10.1M (1.66%) | 17.4M (1.00%) |
| Seconds per step | 18.4 | 34.1 |
| Wall clock | 11m02s | 19m31s |
| Held-out loss | 0.105 | 0.136 |

Two things worth reading carefully. The speed-up is **1.85x, not the 2.8x** the parameter counts
suggest — model loading, the optimizer, evaluation and MPS launch overhead do not shrink with the
model. And the smaller model came out *ahead* on loss, which is not evidence that it is better: at
34 steps both are undertrained, the adapter is a larger fraction of the smaller model, and the
held-out set is 26 dialogues, so a gap of 0.03 means nothing.

That last point is the one to take away. **This dataset cannot rank models.** It is template-
generated and low-entropy, so once a model has the sentence shapes there is nothing left to learn,
and loss saturates for everyone. It is a good benchmark for the pipeline and a useless one for
choosing what to ship. For that you need real dialogues and tool-call metrics, not loss.

So: iterate at 0.6B, and step up to 1.7B or 4B once the harness is trustworthy and the question has
changed from "does this train" to "are the arguments right".

## Hardware

The default configuration measured end to end: **Qwen3-0.6B, LoRA r=16, `max_seq_length: 2048`,
micro-batch 1 with 8-step accumulation, gradient checkpointing on, on an Apple M3 Pro with 36 GB** —
10.1M trainable parameters (1.66% of the model).

**On MPS, leave `gradient_checkpointing: true`.** This is the one setting worth stating as a rule
rather than a preference. PyTorch's MPS allocator may reserve well beyond physical memory — 1.7x the
recommended maximum by default — and does not release what it has cached. Measured on a 1.7B run
with checkpointing off: the footprint grew to 42 GB on this 36 GB machine, filled 16 GB of swap, and
decayed from 20 s/step to over 200 s/step. It never raised an error; it just got slower until it was
unusable, which is the worst way for a memory problem to present itself. The default model leaves
more headroom than that, but the failure is silent enough to be worth not courting.

That also makes process memory a poor guide to what a run *needs* — the footprint expands to fill
what is available. Rather than quote invented per-model numbers, the levers, in the order to reach
for them when a run does not fit:

1. `training.gradient_checkpointing: true` — the biggest saving, and the default on MPS for the
   reason above. Costs roughly 30% throughput on hardware that has memory to spare.
2. `training.per_device_batch_size: 1`, raising `gradient_accumulation_steps` to keep the effective
   batch the same — already the case in the MPS config, so this one is for CUDA.
3. Lower `dataset.max_seq_length` to what `tcsft stats` says you need — activation memory is linear
   in it, and 2048 is already generous for the example scenario's ~880-token dialogues.
4. A smaller base model — though the default is already the smallest of the Qwen3 family, so this
   lever only exists if you have stepped up.

On CUDA, `configs/cuda_qlora.yaml` adds 4-bit base weights and fused cross-entropy, which is what
makes an 8B model fit a mid-range card. Both are CUDA-only and the trainer refuses them elsewhere.

## Development

```bash
make test   # pytest
make lint   # ruff format --check, ruff check, pyright (strict)
make fmt    # autoformat + autofix
```

Layout — `src/toolcall_sft/`:

| Module | Role |
|---|---|
| `schema` | the dialogue as the unit of training, dedup and splitting |
| `dataset` | load/write JSONL, dedup, deterministic split |
| `scenario` | the example toolset, and the loader for `system_prompt.txt` — replace these two |
| `generate` | synthetic dialogues, branch-balanced |
| `masking` | chat-template-exact tokenization, assistant-only loss |
| `stats` | token-length statistics and the length filter |
| `metrics` | tool-call precision/recall |
| `anonymizer` | deterministic pseudonymization for captured dialogues |
| `config` | typed experiment YAML |
| `training/` | SFT and adapter merge — the only part that needs torch |

`tcsft` is the data-side CLI; `tcsft-train` is the training one.

## License

Apache 2.0 — see [LICENSE](LICENSE).
