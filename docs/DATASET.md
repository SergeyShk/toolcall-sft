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

The bundled generator can supply up to about 2060 dialogues with the default weights before its
`out_of_scope` pool runs dry; it refuses rather than repeats, and the error names the ceiling.

### The token-budget correction

Training examples are per assistant turn, and what the model learns from is the target tokens of
each — everything before them is masked out. `tcsft stats --config configs/mac_mps.yaml` reports
the three sizes that matter, here for the 300-dialogue demo corpus under Qwen3-0.6B:

```
dialogues: 300 (0 failed to tokenize), examples: 975 (one per assistant turn)
longest example per dialogue: min 729  p50 877  p90 906  max 920   <- must fit max_seq_length
tokens per epoch per dialogue: min 1426  p50 3181  p90 4000  max 4032   (763485 total)
trained tokens per dialogue:   min 38  p50 109  p90 123  max 136   (3.3% of epoch)
```

1000 dialogues at ~110 trained tokens is ~110K tokens of signal. That is enough for LoRA at r=16 on
a small model. The other two numbers are cost: the longest example is what `max_seq_length` has to
cover, and the tokens per epoch — every turn's prompt re-processed — are what you pay for. The
*length* of your system prompt therefore matters to cost but not to learning, and it matters once
per turn: a 6000-token prompt would drop the trained share below 1% and multiply the compute per
dialogue by eight without adding a single token of signal.

Below ~300 dialogues a LoRA learns the call *format* and stays unreliable on arguments and rare
branches. That failure looks like success in a demo.

## Which branches

The list below is the example scenario's, but the shape generalises: one or two happy paths, and a
longer tail of "the tool must not fire" cases.

| Branch | Share | Reaches the write tool |
|---|---:|---|
| `happy_path` — payee and amount given, confirm, send | 30% | yes |
| `out_of_scope` — off-topic, leaves through `escalate` | 16% | no |
| `missing_amount` — one field missing, ask, then send | 13% | yes |
| `ambiguous_payee` — two matches, ask which | 12% | no |
| `payee_not_found` — lookup returns nothing | 11% | no |
| `cancelled` — user changes their mind mid-flow | 10% | no |
| `payment_declined` — the write is refused | 8% | yes |

Three rules do the real work:

- **Cap the happy path at 40-50%.** Production data skews towards it far harder than this, because
  the messy conversations are the ones that got escalated or abandoned. Left uncorrected, the model
  learns that the write tool is the answer to everything.
- **Negative examples are 10-15% minimum**, and here they are about half the corpus. A refusal, a
  clarifying question with no tool call, an action that must not happen — these are the only
  examples that teach *restraint*, and restraint is what makes a tool-calling model safe to deploy.
- **Give the model somewhere to go.** `out_of_scope` is the largest single non-happy branch because
  a narrow model with no exit will answer anything. An escape hatch works only if it is a tool call
  the model was trained to make; a prompt instruction alone does not survive fine-tuning on data
  that never demonstrates it. `payment_declined` is the same idea applied to the write path: a tool
  can come back refused, and the model has to say so instead of reporting the success it expected.
  Its amounts deliberately overlap with the happy path's, so a large number is not a tell — the
  model has to read the tool result.

Production usually cannot supply the rare branches in useful numbers. Top them up synthetically:
that is what `tcsft generate` demonstrates, and what a simulator against your real tools does
properly.

## Keeping the split honest

`tcsft split` deduplicates and splits by **content hash of the dialogue**, so:

- a dialogue lands on the same side of the split regardless of its position in the file, which makes
  regenerating or re-exporting the data non-destructive to your eval set;
- an exact duplicate — same messages, same tool calls, same tools, whitespace and argument order
  aside — cannot appear on both sides.

The tools list is part of the fingerprint, deliberately: a different tool catalogue renders a
different training prompt, so it is a different example.

What the hash split does **not** do is catch near-duplicates, and you should know how far that goes
for your data. On the demo corpus it goes far: template-generated dialogues differ from each other
by one phrase, so of the 50 tool-call targets in a 33-dialogue eval set, 33 appear verbatim in the
training set, as do 17 of 56 text replies. That is why the README says this dataset cannot rank
models — its held-out loss is mostly a measurement of leakage. Real dialogues are less repetitive
but not immune: a customer who retried three times, a conversation re-exported twice, a support
macro pasted into hundreds of chats. If your data has families like that, split by a family key
(customer, session, macro) rather than by dialogue, or build the eval set by hand from dialogues
you know are not paraphrases of training ones.

The split is not stratified either. The hash decides each dialogue's side independently, so a rare
branch can end up with two eval dialogues, or none. `tcsft split` prints the per-branch breakdown
(by `dialogue_id` prefix) for exactly this reason — check it, and reweight or grow the corpus when
a branch you care about is thin on the eval side.

## Bringing your own data

1. Emit JSONL in the schema in the [README](../README.md). Anything that can produce a list of
   messages with tool calls will do. Tool-call ids are optional and pass through when present.
2. `tcsft anonymize` **first**, if the dialogues came from a real system, and read a sample of the
   output yourself.
3. `tcsft validate` — structural check plus a tool-call census, which is the fastest way to notice
   that one branch is missing entirely.
4. `tcsft stats --config <your yaml>` — confirm the longest-example distribution and the
   trained-token share before you commit to a `max_seq_length`.
5. `tcsft render --config <your yaml>` — read one dialogue the way the trainer will see it. This is
   where a template that renders your tool calls differently from what you expected shows up.
6. `tcsft filter --config <your yaml>` if there is a tail of over-long dialogues; it drops them
   through the same tokenization path training uses, so what survives is exactly what the trainer
   accepts. The trainer itself refuses, by id, rather than dropping silently.
7. `tcsft split`, then train.

Replace [scenario.py](../src/toolcall_sft/scenario.py) and
[system_prompt.txt](../src/toolcall_sft/system_prompt.txt) with your own toolset and prompt. Keeping
them together, and serving exactly what they contain, is what stops the training copy and the
serving copy from drifting apart — the failure mode that degrades a tuned model faster than anything
else.
