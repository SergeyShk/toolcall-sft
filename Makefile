.PHONY: sync sync-cuda test lint fmt demo generate validate stats split train merge

sync:
	uv sync

sync-cuda:
	uv sync --group cuda

test:
	uv run pytest tests

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests
	uv run pyright

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

# End-to-end on synthetic data: generate -> validate -> split -> train -> merge.
demo:
	uv run tcsft generate --out data/all.jsonl --count 300
	uv run tcsft validate data/all.jsonl
	uv run tcsft split data/all.jsonl --train-out data/train.jsonl --eval-out data/eval.jsonl
	uv run tcsft-train train --config configs/mac_mps.yaml

# make generate out=data/all.jsonl count=1000
generate:
	uv run tcsft generate --out $(out) --count $(count)

# make validate path=data/train.jsonl
validate:
	uv run tcsft validate $(path)

# make stats path=data/all.jsonl
stats:
	uv run tcsft stats $(path)

# make split path=data/all.jsonl train_out=data/train.jsonl eval_out=data/eval.jsonl
split:
	uv run tcsft split $(path) --train-out $(train_out) --eval-out $(eval_out)

# make train config=configs/mac_mps.yaml
train:
	uv run tcsft-train train --config $(config)

# make merge adapter=outputs/<run>/adapter output=outputs/<run>/merged
merge:
	uv run tcsft-train merge --adapter $(adapter) --output $(output)
