# Security

langgraph-short-long-term-memory keeps two kinds of memory about the people who talk to
an agent: their conversations (checkpointer) and what the agent learned about them
(store). That is personal data. Design goals: one user can never read or write another
user's memory, recalled memories can't take over the agent, secrets are never stored,
nothing leaves the machine unless you opt in, and a user can delete everything.

## Reporting a vulnerability

Open a private security advisory at
<https://github.com/MartinCastroAlvarez/langgraph-short-long-term-memory/security/advisories/new>.
Do **not** open a public issue for a suspected vulnerability. We acknowledge within 5
business days and aim to ship a fix within 30 days. You are credited in the advisory
unless you ask otherwise. A cross-user read or write is in scope and very welcome.

Supported versions: the latest minor release.

## 1. Threat model

**Assets.** Conversation history (checkpoints), long-term memories (facts, episodes,
style rules), the SQLite files that hold them, environment variables (API keys), and your
model endpoint.

| Actor | Controls | Must not be able to |
| --- | --- | --- |
| A user | their messages, their user id *only as authenticated by your app* | read or write another user's memory or threads, store instructions that later run as rules, keep data after "forget me" |
| The model | replies, tool calls and their arguments, structured outputs (facts, episodes, rules, summaries) | choose a namespace or user, store secrets, grow memory without bound, change the fixed rules |
| Whoever controls `OLLAMA_HOST` | where prompts and texts to embed are sent | receive conversations without your explicit opt-in |

| Threat | Control |
| --- | --- |
| Cross-user access through ids | User and thread ids must match `[A-Za-z0-9][A-Za-z0-9_-]{0,63}` before any namespace or thread id is built (`namespaces.validate_id`): no `/`, `.`, `*`, whitespace. Store namespaces are `("memories", user_id, kind)` with `kind` from a fixed list; checkpointer thread ids are `"<user_id>/<thread_id>"`. Tested with hostile ids. |
| The model picking the namespace | The user id is the run context (`context=UserContext(...)`), read by nodes and tools through `runtime.context`. It is not a tool parameter; the tool schemas expose only `fact`/`slot`/`query` (tested), and a tool call that adds `user_id: "bob"` still writes to the current user (tested). |
| Reaching another thread's checkpoints | `fork`/`replay` only accept a checkpoint id found in the history of the caller's own `"<user>/<thread>"` (tested with another user's checkpoint id). |
| Prompt injection through memories | Recalled facts, episodes, rules and the summary are model- or user-written text. They enter prompts only through `prompts.data_block`, escaped (`& < >`) inside `<data name=...>`, and every system prompt says `<data>` is untrusted and never instructions. A memory that contains `</data>` can't close its block (tested). Escaping reduces injection but can't remove it: a poisoned fact can still bias an answer. |
| Procedural memory rewriting safety | The fixed assistant rules come first and are never stored; learned rules go in a `style` block that may only shape tone, length and format, capped at 10 rules × 200 characters, deduplicated, and filtered for secrets. |
| Secrets stored as memories | `long_term.looks_secret` refuses facts and rules that mention passwords, API keys, tokens, PINs/CVVs, card-like digit runs, SSN-like numbers or `sk-...` keys; the tool replies "Not saved". It is a heuristic: tell users not to share secrets, and don't rely on it for compliance. |
| Unbounded growth | Facts ≤ 300 characters, ≤ 20 per extraction, history of 5 values per fact; episodes ≤ 3 × 240 characters, one per thread; summary ≤ `summary_max_chars`; messages ≤ 4,000 characters; the system prompt ≤ 60% of `max_context_tokens`; `recursion_limit` 30. |
| Malformed model output | Structured outputs are pydantic-validated; an invalid one writes nothing (tested). Slots are normalised to snake case or dropped. |
| Right to erasure | `Assistant.forget_user` (CLI `--forget-me`) deletes every item in every namespace under `("memories", user)`, deletes every checkpoint of every thread the user ever used (`checkpointer.delete_thread`), and cancels the user's pending consolidation jobs so none can recreate a memory (tested, also across a restart). Copies outside these two databases (backups, logs, a remote model's logs) are yours to handle. |
| Data leaving the machine | `OLLAMA_HOST` must be loopback unless `MEMORY_ALLOW_REMOTE_OLLAMA=1`, for chat and embeddings. Hosted providers are opt-in and need your own key. LangSmith tracing stays off unless you set `LANGSMITH_TRACING`. |
| Secrets in prompts or logs | Only `ANTHROPIC_API_KEY` is read, only by the optional adapter. Keys never enter prompts, state, the store or output. `.env` is git-ignored. |
| Code execution | No `eval`/`exec`/`compile`, no `pickle`/`yaml`, no `subprocess`/`shell=`; an AST test enforces it and allows no `# noqa`, `# nosec` or `# type: ignore` in the package. |

## 2. Personal data (PII) notes for operators

- **Authenticate first.** This package trusts the user id it is given. Derive it from
  your session, never from the request body or the model.
- **Tell users what is remembered.** `--memories` / `Assistant.memories()` shows
  everything stored about a user; expose it, and expose "forget me".
- **Both files hold PII.** `threads.db` has whole conversations; `memories.db` has the
  extracted facts. Encrypt the disk, restrict file permissions, and include both in
  your retention policy. A summary is still personal data.
- **Time travel keeps history.** Every checkpoint of a thread is retained until the
  thread is deleted; the store keeps the last 5 values of each fact. Size retention
  accordingly.
- **Embeddings are personal data too.** sqlite-vec stores a vector per memory; it is
  deleted with the memory.

## 3. Dependencies

CI runs `bandit` on the package and `pip-audit` on the locked environment. Dependabot
alerts, security updates and secret scanning are enabled on the repository.
