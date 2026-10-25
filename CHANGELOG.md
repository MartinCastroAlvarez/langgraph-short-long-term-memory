# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-25

### Added

- Chat graph (`summarize -> recall -> agent <-> tools`) compiled with a `SqliteSaver`
  checkpointer and a semantic `BaseStore`, run with `context=UserContext(user_id=...)`.
- Short-term memory: token-budget trimming before every model call, a running summary
  that removes folded messages with `RemoveMessage`, cuts only at human turns,
  per-user thread ids, and time travel (`history`, `replay`, `fork`).
- Long-term memory in `("memories", user, kind)`: slot-keyed facts that update instead
  of duplicating (with similarity dedup for slotless facts and a 5-value history),
  episodes per thread recalled as examples, and procedural style rules rewritten from
  feedback.
- Hot-path `remember`/`forget` tools and a background consolidation graph (memory
  writer with structured output, episode writer, optimizer) behind a debounced,
  idempotent `Consolidator` with a per-thread watermark.
- `forget_user` ("forget me"): store namespace, checkpoints and pending jobs.
- Stores: `InMemoryStore`, `SqliteStore` + sqlite-vec, and an optional adapter for
  `S3Store` from s3-agent-memory. Embedders: ollama, fastembed (extra), hashing.
- `FakeAssistantModel` (supports `bind_tools` and `with_structured_output`), the
  two-thread demo, and `python -m memory_agent` (ollama by default, Anthropic optional,
  fake offline) with `--mermaid` diagrams.
