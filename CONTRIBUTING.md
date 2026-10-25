# Contributing to langgraph-short-long-term-memory

## TL;DR

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install poetry
poetry install                       # add -E anthropic / -E fastembed for the adapters
poetry run pytest
poetry run ruff check memory_agent tests examples && poetry run ruff format --check memory_agent tests examples
poetry run mypy memory_agent
poetry run bandit -c pyproject.toml -q -r memory_agent
poetry run pip-audit
poetry run python -m examples.quickstart
```

Open a PR against `main` from a `feat/...`, `fix/...`, `docs/...` or `chore/...` branch.

## What this project will not accept

- Tests that call a real model or touch the network.
- A tool, node or API that takes a namespace or another user's id from model output.
- Memory that `forget_user` doesn't delete.
- Hosted providers or tracing turned on by default.

## What it does accept

- Other backends behind the same interfaces: `PostgresSaver` / `PostgresStore`,
  another local embedder, another `BaseStore`.
- Better memory management: importance/recency scoring, TTLs, contradiction handling,
  memory review UIs, with tests for isolation and idempotence.
- Hardening, each with a hostile test.

## House rules

1. Tests alongside features; security-relevant changes get a hostile case.
2. README, ARCHITECTURE/SECURITY (when relevant) and CHANGELOG change with the code.
3. No secrets in commits; `.env.example` holds defaults only.
