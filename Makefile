DB ?= results/results.db
EXPERIMENT ?= baseline_timit_v3
CV_EXPERIMENT ?= calibrated_timit_cv
OPTLAB = uv run optlab

.PHONY: setup test test-python test-frontend lint build dev api snapshot toy baseline cv check

setup:
	uv sync --frozen --extra api
	pnpm --dir frontend install --frozen-lockfile

test: test-python test-frontend

test-python:
	uv run pytest

test-frontend:
	pnpm --dir frontend test

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	pnpm --dir frontend format:check

build:
	pnpm --dir frontend build

check: lint test build

dev:
	pnpm --dir frontend dev

api:
	$(OPTLAB) serve --db $(DB)

snapshot:
	$(OPTLAB) export-snapshot --db $(DB)

# Recomputes toy runs and rebuilds tables/plots. This explicitly rewrites toy results.
toy:
	$(OPTLAB) init-db --db $(DB)
	$(OPTLAB) toy --db $(DB) --config configs/experiments/toy.yaml --force
	$(OPTLAB) toy-report --db $(DB) --out results/toy --summary results/toy_summary.md

# Needs manifests and feature caches. Uses an explicit experiment name.
baseline:
	$(OPTLAB) init-db --db $(DB)
	$(OPTLAB) sweep --experiment $(EXPERIMENT) --db $(DB)
	$(OPTLAB) stats --experiment $(EXPERIMENT) --db $(DB)

cv:
	$(OPTLAB) init-db --db $(DB)
	$(OPTLAB) cv --experiment $(CV_EXPERIMENT) --db $(DB)
