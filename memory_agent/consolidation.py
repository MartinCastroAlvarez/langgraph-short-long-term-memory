"""The background path: a deferred job that turns a finished conversation into memories.

    START -> extract -> write_facts -> write_episode -> [optimize] -> END

* ``extract`` - the **memory writer**: structured output (``MemoryUpdate``) with the
  durable facts in the transcript, each with a slot, and any feedback on style.
* ``write_facts`` - upserts each fact (update instead of duplicate, see
  ``long_term.upsert_fact``) or forgets it.
* ``write_episode`` - an ``Episode`` (situation, approach, outcome) for this thread.
* ``optimize`` - only when there was feedback: the **optimizer** rewrites the style
  rules (procedural memory), "prompt optimization" in LangMem's terms.

It is a separate graph with no checkpointer, run outside the request path by
``Consolidator``: ``schedule()`` after each turn (debounced: one pending job per thread)
and ``run_pending()`` when the conversation goes idle. The job is idempotent, so
running it twice on the same thread changes nothing the second time.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from pydantic import ValidationError

from memory_agent.config import Settings
from memory_agent.long_term import forget_fact
from memory_agent.long_term import get_rules
from memory_agent.long_term import list_facts
from memory_agent.long_term import save_episode
from memory_agent.long_term import save_rules
from memory_agent.long_term import upsert_fact
from memory_agent.namespaces import UserContext
from memory_agent.namespaces import validate_id
from memory_agent.prompts import episode_writer_prompt
from memory_agent.prompts import memory_writer_prompt
from memory_agent.prompts import optimizer_prompt
from memory_agent.schemas import Episode
from memory_agent.schemas import Instructions
from memory_agent.schemas import MemoryUpdate

__all__ = ["ConsolidationState", "Consolidator", "build_consolidation_graph"]


class ConsolidationState(TypedDict, total=False):
    thread_id: str
    transcript: str
    """Messages not consolidated before: the source of facts and feedback."""
    conversation: str
    """The whole thread (summary + messages): the source of the episode."""
    update: dict[str, Any]
    writes: list[dict[str, Any]]
    episode: dict[str, Any]
    rules: list[str]


def _structured(model: BaseChatModel, schema: type[Any], prompt: str) -> Any:
    """Structured output, or ``None`` when the model's reply doesn't validate."""
    runnable = model.with_structured_output(schema)
    try:
        result = runnable.invoke([SystemMessage(prompt), HumanMessage("Answer now.")])
    except (ValidationError, ValueError):
        return None
    return result if isinstance(result, schema) else None


def build_consolidation_graph(
    settings: Settings, model: BaseChatModel, *, store: BaseStore | None = None
) -> CompiledStateGraph[Any, Any, Any, Any]:
    def extract(state: ConsolidationState, runtime: Runtime[UserContext]) -> dict[str, Any]:
        if runtime.store is None:
            raise RuntimeError("consolidation needs a store")
        known = [str(i.value.get("content", "")) for i in list_facts(runtime.store, _uid(runtime))]
        prompt = memory_writer_prompt(state.get("transcript", ""), known)
        update = _structured(model, MemoryUpdate, prompt) or MemoryUpdate()
        return {"update": update.model_dump()}

    def write_facts(state: ConsolidationState, runtime: Runtime[UserContext]) -> dict[str, Any]:
        store, user_id = _store(runtime), _uid(runtime)
        update = MemoryUpdate.model_validate(state.get("update", {}))
        writes: list[dict[str, Any]] = []
        for fact in update.facts:
            if fact.action == "delete":
                deleted = forget_fact(store, user_id, fact.slot or fact.content)
                writes += [
                    {"action": "deleted", "key": k, "content": fact.content} for k in deleted
                ]
                continue
            w = upsert_fact(
                store,
                user_id,
                fact.content,
                fact.slot,
                source="background",
                dedup_threshold=settings.dedup_threshold,
            )
            writes.append({"action": w.action, "key": w.key, "content": w.content})
        return {"writes": writes}

    def write_episode(state: ConsolidationState, runtime: Runtime[UserContext]) -> dict[str, Any]:
        conversation = state.get("conversation") or state.get("transcript", "")
        if not conversation.strip():
            return {"episode": {}}
        episode = _structured(model, Episode, episode_writer_prompt(conversation))
        if episode is None:
            return {"episode": {}}
        save_episode(_store(runtime), _uid(runtime), state["thread_id"], episode)
        return {"episode": episode.model_dump()}

    def has_feedback(state: ConsolidationState) -> str:
        return "optimize" if state.get("update", {}).get("feedback") else END

    def optimize(state: ConsolidationState, runtime: Runtime[UserContext]) -> dict[str, Any]:
        store, user_id = _store(runtime), _uid(runtime)
        current = get_rules(store, user_id)
        feedback = str(state.get("update", {}).get("feedback", ""))
        proposal = _structured(model, Instructions, optimizer_prompt(current, feedback))
        if proposal is None:
            return {"rules": current}
        return {"rules": save_rules(store, user_id, proposal.rules)}

    builder = StateGraph(ConsolidationState, context_schema=UserContext)
    builder.add_node("extract", extract)
    builder.add_node("write_facts", write_facts)
    builder.add_node("write_episode", write_episode)
    builder.add_node("optimize", optimize)
    builder.add_edge(START, "extract")
    builder.add_edge("extract", "write_facts")
    builder.add_edge("write_facts", "write_episode")
    builder.add_conditional_edges("write_episode", has_feedback, ["optimize", END])
    builder.add_edge("optimize", END)
    return builder.compile(store=store, name="consolidate")


def _uid(runtime: Runtime[UserContext]) -> str:
    return runtime.context.user_id


def _store(runtime: Runtime[UserContext]) -> BaseStore:
    if runtime.store is None:
        raise RuntimeError("consolidation needs a store")
    return runtime.store


@dataclass
class Consolidator:
    """A debounced queue of consolidation jobs: at most one pending job per thread.

    ``run`` is given ``(user_id, thread_id)`` and does the work (the ``Assistant`` reads
    the thread's transcript and invokes the consolidation graph). ``cancel_user`` drops
    a user's pending jobs, so "forget me" can't be undone by a job that runs later.
    """

    run: Callable[[str, str], dict[str, Any]]
    _pending: dict[tuple[str, str], None] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def schedule(self, user_id: str, thread_id: str) -> None:
        key = (validate_id(user_id, "user id"), validate_id(thread_id, "thread id"))
        with self._lock:
            self._pending[key] = None

    @property
    def pending(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._pending)

    def cancel_user(self, user_id: str) -> int:
        with self._lock:
            keys = [k for k in self._pending if k[0] == user_id]
            for k in keys:
                del self._pending[k]
        return len(keys)

    def run_pending(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        while True:
            with self._lock:
                if not self._pending:
                    return results
                key = next(iter(self._pending))
                del self._pending[key]
            results.append(self.run(*key))
