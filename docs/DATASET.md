# Building the dataset

How much data a narrow tool-calling tune needs, which branches it has to cover, and how to keep the
balance honest. The bundled generator (`tcsft generate`) is one worked example of the rules below;
the rules are what transfer to your own data.

## How much

**Target 1000-2000 dialogues after deduplication. A first meaningful run is possible at 300-500.
Returns flatten out around 3000-5000** — past that you are buying coverage and balance, not volume.

Count *behaviours*, not examples. The working rule for a narrow LoRA tune is **50-100 examples per
distinct behaviour that has to work reliably**. Seven branches at 100 each is 700 dialogues, and
that number is a far better planning tool than a total.

### The token-budget correction

What the model actually learns from is assistant tokens — everything else is masked out.
`tcsft stats` reports the share:

```
total tokens:   min 529  p50 707  p90 740  max 759
trained tokens: min 24  p50 106  p90 124  max 136  (12.4% of total)
```

1000 dialogues at ~100 trained tokens is ~100K tokens of signal. That is enough for LoRA at r=16 on
a small model, and it is why the *length* of your system prompt matters to cost but not to learning:
a 6000-token prompt would drop the trained share to ~2% and multiply the compute per example by
eight without adding a single token of signal.

Below ~300 dialogues a LoRA learns the call *format* and stays unreliable on arguments and rare
branches. That failure looks like success in a demo.

## Which branches

The list below is the example scenario's, but the shape generalises: one or two happy paths, and a
longer tail of "the tool must not fire" cases.

| Branch | Share | Reaches the write tool |
|---|---:|---|
| `happy_path` — payee and amount given, confirm, send | 35% | yes |
| `missing_amount` — one field missing, ask, then send | 15% | yes |
| `out_of_scope` — request the assistant must decline | 14% | no |
| `ambiguous_payee` — two matches, ask which | 12% | no |
| `payee_not_found` — lookup returns nothing | 8% | no |
| `insufficient_funds` — a precondition fails | 8% | no |
| `cancelled` — user changes their mind mid-flow | 8% | no |

Two rules do the real work:

- **Cap the happy path at 40-50%.** Production data skews towards it far harder than this, because
  the messy conversations are the ones that got escalated or abandoned. Left uncorrected, the model
  learns that the write tool is the answer to everything.
- **Negative examples are 10-15% minimum**, and here they are half the corpus. A refusal, a
  clarifying question with no tool call, an action that must not happen — these are the only
  examples that teach *restraint*, and restraint is what makes a tool-calling model safe to deploy.

Production usually cannot supply the rare branches in useful numbers. Top them up synthetically:
that is what `tcsft generate` demonstrates, and what a simulator against your real tools does
properly.

## Keeping the split honest

`tcsft split` deduplicates and splits by **content hash of the dialogue**, so:

- a dialogue lands on the same side of the split regardless of its position in the file, which makes
  regenerating or re-exporting the data non-destructive to your eval set;
- near-identical dialogues cannot straddle the split — the most common way a tool-calling eval ends
  up reporting numbers it has not earned.

The tools list is part of the fingerprint, deliberately: a different tool catalogue renders a
different training prompt, so it is a different example.

## Bringing your own data

1. Emit JSONL in the schema in the [README](../README.md). Anything that can produce a list of
   messages with tool calls will do.
2. `tcsft anonymize` **first**, if the dialogues came from a real system, and read a sample of the
   output yourself.
3. `tcsft validate` — structural check plus a tool-call census, which is the fastest way to notice
   that one branch is missing entirely.
4. `tcsft stats` — confirm the length distribution and the trained-token share before you commit to
   a `max_seq_length`.
5. `tcsft filter --max-seq-length N` if there is a tail of over-long dialogues; it drops them through
   the same tokenization path training uses, so what survives is exactly what the trainer accepts.
6. `tcsft split`, then train.

Replace [scenario.py](../src/toolcall_sft/scenario.py) with your own prompt and tools. Keeping both
in one module is what stops the training copy and the serving copy from drifting apart — the failure
mode that degrades a tuned model faster than anything else.
