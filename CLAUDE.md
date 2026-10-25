# CLAUDE.md

Contract between this repository and any AI agent contributing to it.

## Required reading

1. This file. 2. [`README.md`](README.md). 3. [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. [`SECURITY.md`](SECURITY.md).

## Mission

`langgraph-short-long-term-memory` shows agent memory in current LangGraph: short-term
memory in a checkpointer (trimming, running summary, thread isolation, time travel) and
long-term memory in a store (semantic facts, episodes, procedural rules; hot-path tools
and a background consolidation job), as small, correct and tested code.

## The rules

1. **Local by default.** ollama on loopback is the default for chat and embeddings;
   hosted providers stay optional adapters. No telemetry, no LangSmith upload.
2. **Namespaces come from the run context.** Never add a user id, namespace or thread id
   parameter to a tool, and never build a namespace outside `namespaces.py`. Every id is
   validated before use.
3. **Memories are untrusted.** Recalled text enters prompts only through
   `prompts.data_block`. Model output that becomes memory goes through the pydantic
   schemas in `schemas.py` and `long_term.upsert_fact` (dedup, caps, secret filter).
4. **Long-term writes are idempotent.** Re-running consolidation must change nothing;
   an old thread must never overwrite a newer fact (the watermark).
5. **"Forget me" is complete.** Any new place that stores user data must be wiped by
   `Assistant.forget_user`, with a test.
6. **Offline tests.** `FakeAssistantModel` and `HashingEmbeddings` only. Tests check
   control flow and isolation, not prose.
7. Never add `# noqa`, `# nosec` or `# type: ignore` to the package, skip a test, or
   loosen lint/type/coverage config.

## Done means

`poetry run pytest` (coverage >= 85%), `ruff check`, `ruff format --check`, `mypy`
(strict), `bandit` and `pip-audit` all green; `python -m examples.quickstart` passes;
README and CHANGELOG are updated.

## Git

Small commits, imperative subjects, never force-push `main`, never commit secrets, `.env`
or `.memory/`.
