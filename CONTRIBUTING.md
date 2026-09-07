# Contributing

Thanks for taking a look. Issues and pull requests are both welcome.

## Getting set up

```bash
uv sync
make test
make lint
```

`make lint` runs `ruff format --check`, `ruff check` and `pyright` in strict mode. CI runs exactly
these on Python 3.11 through 3.13, plus a CLI smoke on the light install and the training smoke
test on the full one, so a green `make lint && make test` locally means a green CI.

`make fmt` applies the formatter and the autofixable lint rules.

## What the code tries to be

Small enough to read in an afternoon. The value is in being obviously correct about a few things
that are easy to get silently wrong — chat-template fidelity per turn, loss masking, deterministic
splits — not in covering every training feature.

A few conventions that are load-bearing:

- **No silent fallbacks in the data path.** If a chat template cannot produce a prompt the server
  would build, if a tool result answers no pending call, if a dialogue exceeds the budget, if a
  config names a key that does not exist — say so and stop. A dataset bug that trains successfully
  is worse than one that crashes.
- **Comments explain *why*.** What the code does should be legible from the code. Comments are for
  the constraint that made it look like that.
- **Types are strict.** `pyright` runs in strict mode and the codebase has no `type: ignore`.
- **The root package stays free of torch, and light to import.** `schema`, `dataset`, `masking`,
  `metrics`, `parsing`, `config`, `stats`, `scenario`, `generate` and `anonymizer` import without a
  training install; transformers is imported only where a tokenizer is actually used, and only when
  it is used, so `tcsft --help` stays instant. Anything needing torch lives in `training/`.

## Tests

Every behavioural change needs a test. The existing suite is the guide to the style: real inputs,
assertions on outcomes, no mocking of the code under test. Tokenizer-dependent behaviour is tested
against small purpose-built tokenizers rather than a downloaded model, so the suite runs in seconds
and offline; the Qwen3 chat template itself is vendored under `tests/templates/` so the per-turn
rendering is tested against the real thing. `tests/test_training_smoke.py` trains a two-layer model
built in the test through the real `Trainer`; it is marked `training` and skipped when torch is not
installed.

## Pull requests

Keep them focused, explain the reasoning in the description, and make sure `make lint && make test`
passes. If you are changing behaviour people may depend on — the dialogue schema, the config keys,
the CLI — say so explicitly.

## Scope

Happy to take: base-model and device support, evaluation metrics, dataset tooling, documentation
fixes, and bug reports with a reproducing case.

Likely to decline: wrappers around other training frameworks, and features that only make sense for
one private deployment. If you are unsure, open an issue before writing the code.

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0, the
same as the project.
