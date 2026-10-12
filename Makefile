DB ?= results/results.db
OPTLAB = uv run optlab

.PHONY: setup test lint toy

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

# Re-runs the whole toy suite (CPU, minutes) and rebuilds table, plots and summary from the DB.
toy:
	$(OPTLAB) init-db --db $(DB)
	$(OPTLAB) toy --db $(DB) --config configs/experiments/toy.yaml --force
	$(OPTLAB) toy-report --db $(DB) --out results/toy
