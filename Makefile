.PHONY: sync lint format type-check test check doctor

sync:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

type-check:
	uv run mypy main.py tests

test:
	uv run pytest

check: lint type-check test

doctor:
	uv run python main.py doctor
