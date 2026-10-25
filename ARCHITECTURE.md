# Architecture

One package, `memory_agent`, two graphs, two databases.

```
memory_agent/
  config.py         Settings from MEMORY_* env vars (budgets, recall sizes, write mode)
  namespaces.py     id validation, UserContext, store namespaces, checkpointer thread ids
  short_term.py     count_tokens, plan_summary (where to cut), fit_to_budget (trim_messages)
  long_term.py      facts (upsert/forget), episodes, rules, recall, threads, watermark, wipe
  schemas.py        structured outputs: MemoryUpdate, FactCandidate, Episode, Instructions
  prompts.py        system prompts per role, data_block (escaped <data> for untrusted text)
  tools.py          remember / forget (hot path), user from runtime.context
  graph.py          chat graph: [summarize] -> recall -> agent <-> tools
  consolidation.py  background graph: extract -> write_facts -> write_episode -> [optimize];
                    Consolidator (debounced queue)
  assistant.py      Assistant: owns graphs, checkpointer, store; chat, time travel,
                    consolidate, forget_user
  stores.py         open_checkpointer (SqliteSaver), open_store (memory / sqlite / s3 adapter)
  embeddings.py     HashingEmbeddings, FastEmbedEmbeddings, make_embeddings (ollama default)
  llm.py            make_model: ChatOllama (loopback guard), ChatAnthropic (extra), fake
  fake.py           FakeAssistantModel: role-aware, bind_tools + with_structured_output
  demo.py           the two-thread demo with named checks
  cli.py            python -m memory_agent
```

## Data flow of one turn

1. `Assistant.chat(user, thread, text)` validates both ids, registers the thread under
   `("memories", user, "threads")`, and invokes the chat graph with
   `config.configurable.thread_id = "<user>/<thread>"` and `context=UserContext(user)`.
2. The checkpointer loads the thread's state (messages, summary). `add_messages`
   appends the new `HumanMessage`.
3. `START` routes to `summarize` when the messages exceed `summarize_after_tokens`.
   `plan_summary` picks the cut; the summarizer merges the old summary with the folded
   messages; `RemoveMessage`s delete them from state.
4. `recall` searches the user's `facts` and `episodes` namespaces with the new message
   as the query and loads the `style` rules.
5. `agent` renders the system prompt (`system_prompt`, shedding memories above 60% of
   the budget), trims the messages (`fit_to_budget`), and calls the model.
6. Tool calls go to `ToolNode` (`remember`/`forget` write the store with the user from
   `runtime.context`) and back to `agent`.
7. Every step is a checkpoint. After the turn, the thread is scheduled for consolidation.

## Consolidation

`Consolidator.run_pending()` calls `Assistant.consolidate(user, thread)` per pending
thread: read the thread state, keep the messages whose ids aren't in the thread's
watermark, run the consolidation graph (memory writer on the new messages, episode
writer on the whole thread, optimizer on feedback), then store the new watermark. A
thread with no new messages is skipped.

## Why these choices

- **Composite checkpointer thread ids** make per-user isolation structural: there is no
  lookup to forget.
- **Slots as store keys** turn "update instead of duplicate" into a key collision the
  store handles; similarity dedup is only the fallback.
- **Explicit `index=["content"]` on every `put`.** `InMemoryStore` reads the default
  field list from `index["fields"]` and `SqliteStore` from `index["text_fields"]`;
  passing the fields per write behaves the same on every store, including `S3Store`.
- **Approximate token counting** keeps the budget logic model-agnostic and testable;
  production deployments can pass the model's tokenizer instead.
- **Sync API.** `SqliteSaver`/`SqliteStore` and `invoke` keep the teaching code linear;
  the async variants (`AsyncSqliteSaver`, `AsyncSqliteStore`, `ainvoke`) are drop-ins.
