# Fine-tuning a model for tool calling

Why you would do this, what actually decides whether it works, and where the traps are. The
mechanics live in the [README](../README.md); this is the reasoning around them.

## What the task actually is

A tool-calling fine-tune is not text generation. Given a conversation, the model has to pick a tool,
fill its arguments from what the user said *and* from what earlier tools returned, and know when to
ask a question instead of acting. The output is a structured call that either parses and validates,
or does not.

That has two consequences worth internalising before you start:

- **Success is machine-checkable.** Tool name, argument names, argument values — you can score all
  of it without a human or a judge model. Build that scoring before you build the dataset.
- **The narrow domain is the advantage.** A well-tuned 4-8B model matches a frontier model on one
  narrow flow. That is the entire economic case: latency, cost, and data that never leaves your
  infrastructure. It is not a claim about general capability, and it does not survive scope creep.

The usual shape is **distillation**: a large model already handles the flow in production, and the
local model learns to reproduce its behaviour on that flow.

## Data is the whole game

**Format.** Chat messages with tool calls, exactly as your inference server will render them:
`system` (prompt + tool schemas) → `user` → `assistant(tool_call)` → `tool(result)` → … →
`assistant(final)`. Loss on assistant tokens only.

**Where it comes from,** in rough order of value:

- **Production traces** — the real distribution, including the messy turns you would never invent.
  If you have them, start here.
- **Synthetic dialogues** — for the branches production is too clean to supply. This repo's
  `tcsft generate` is a template generator; the more serious version is to script a simulator
  against your real tools, or have a large model role-play the customer side.
- **Negative examples** — refusals, clarifying questions, and turns where the action must *not*
  happen. Non-negotiable, not a nice-to-have. See [DATASET.md](DATASET.md).

**Volume.** For LoRA on a narrow flow: 500-5000 quality dialogues. A first meaningful run is
possible at 300. Past ~5000 you are buying balance and diversity, not volume.

**PII.** If the traces come from a real product, they contain real people. Anonymize with
*consistency inside the dialogue* — the same payee replaced identically in every turn and every tool
result — or the model both memorises real data and learns from an incoherent conversation.
`tcsft anonymize` does the mechanical part; someone still has to read a sample. In a regulated
domain, confirm you are allowed to train on the data at all, anonymized or not.

## Choosing a base model

Judge on: native tool-calling ability (the Berkeley Function Calling Leaderboard is the usual
reference), size against your hardware, and a licence you can actually ship.

| Family | Sizes | Licence | Notes |
|---|---|---|---|
| **Qwen3** | 0.6B / 1.7B / 4B / 8B / 14B / 30B-A3B / 32B | Apache 2.0 | Strongest open tool calling; the default choice |
| Gemma 3 | 1B / 4B / 12B / 27B | Gemma terms | Strong for the size; check the licence against your use |
| Llama 3.1 / 3.3 | 8B / 70B | Llama licence | Huge ecosystem, weaker at tool calling than Qwen |
| Mistral Small 3.x | 24B | Apache 2.0 | Fast, decent function calling |
| Phi-4 | 14B | MIT | Compact, good for experiments |

Start smaller than you think. A 1.7B tune tells you within an hour whether your data pipeline,
masking and metrics are correct — which is what actually goes wrong first. Scale up once the
harness is trustworthy: 4B and 8B are the usual landing spots for a narrow production flow, and a
MoE like Qwen3-30B-A3B is cheap at inference because only ~3B parameters are active.

## Method

- **LoRA / QLoRA (SFT)** is where you start and usually where you stop. The base is frozen (and in
  QLoRA quantized to 4 bits); you train small adapters. A full fine-tune costs more and does more
  damage to general ability for no gain on a narrow flow.
- Typical hyperparameters: `r=16-32`, `alpha=2×r`, lr `1-2e-4`, 2-3 epochs, cosine schedule,
  adapters on all linear projections. Context: whatever your dialogues actually need — measure with
  `tcsft stats` rather than guessing.
- **DPO / KTO** is the second stage, and only if SFT leaves a *systematic* defect — hallucinated
  arguments, acting without confirmation. You collect good/bad pairs and train on the preference.
- **GRPO** (RL with a verifiable reward) is genuinely viable here, precisely because the reward is
  computable: schema-valid call + right tool + right arguments. It is also the most expensive thing
  on this list. Do not start here.

## Evaluation

1. **A held-out set** split at the *dialogue* level, deduplicated before splitting. Near-identical
   dialogues leaking across the split is the most common way to fool yourself.
2. **Programmatic tool-call metrics** — `tcsft evaluate`: fraction of schema-valid calls, tool-name
   exact match, argument accuracy, and the rate of calls that should not have happened at all. That
   last one is what the negative branches exist to measure.
3. **End-to-end simulation.** vLLM and Ollama both expose an OpenAI-compatible endpoint, so the
   tuned model drops into whatever harness already drives your production model. Running the same
   scenario suites against both is the only apples-to-apples answer.
4. **LLM-as-judge** for the tone and completeness of the free-text replies, where exact match says
   nothing.
5. **Guardrails stay on.** Whatever validates the production model's actions should validate the
   tuned model's. A small tuned model is confident, which is not the same as correct.

## Serving

- **vLLM** for production: OpenAI-compatible, high throughput, can load LoRA adapters on top of a
  base model, and ships tool-call parsers (`--enable-auto-tool-choice --tool-call-parser hermes`
  for Qwen).
- **Ollama / llama.cpp** for local development and demos: export to GGUF, quantize to Q4_K_M — on a
  narrow domain the quality cost is close to nothing.
- **SGLang** as an alternative to vLLM, often faster on structured output.

Whatever you pick, it must render the chat template the same way training did. This is the thing to
verify first when a tune "mysteriously" underperforms.

## The traps, in the order people hit them

1. **Chat-template mismatch.** The single most common failure. Tool calls must serialize at training
   time byte-for-byte the way the inference server renders them. Always use the base model's own
   template; never hand-roll the format. This repo enforces it and fails loudly when a template does
   not round-trip.
2. **Training on the prompt.** Without masking, the model learns to generate system prompts and tool
   results. Verify that loss covers assistant tokens only — `tcsft stats` reports the trained-token
   share, and a plausible number is a few percent to a few tens of percent, never ~100%.
3. **Catastrophic forgetting.** Too many epochs or too high a learning rate on narrow data and the
   model forgets everything else. Evaluate on a general benchmark too, not only your flow.
4. **Train/test leakage** through near-duplicate dialogues. Deduplicate before splitting, by content.
5. **PII in the weights.** Not reversible after the fact.
6. **Scenario drift.** Change the prompt or a tool schema and the tuned model degrades harder than a
   general one — it was trained on the old shape. Budget for periodic retraining from the start, and
   keep the prompt and the toolset in one place (here, `scenario.py` and `system_prompt.txt`) so the
   training and serving copies cannot diverge.
