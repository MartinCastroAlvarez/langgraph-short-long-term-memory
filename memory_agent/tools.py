"""The hot path: tools the assistant calls *during* the conversation to write memory.

The user id is not a tool argument. It comes from ``runtime.context`` (set by the
application per run), and the store from ``runtime.store``, so a model that writes
``{"fact": ..., "user_id": "bob"}`` still writes to the current user's namespace
(tested). The tools' return values are short confirmations, never other memories.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.store.base import BaseStore

from memory_agent.long_term import forget_fact
from memory_agent.long_term import upsert_fact
from memory_agent.namespaces import UserContext

__all__ = ["make_tools"]


def _context(runtime: ToolRuntime[UserContext]) -> tuple[str, BaseStore]:
    if not isinstance(runtime.context, UserContext) or runtime.store is None:
        raise RuntimeError("memory tools need context=UserContext(...) and a store")
    return runtime.context.user_id, runtime.store


def make_tools(*, dedup_threshold: float) -> list[BaseTool]:
    @tool
    def remember(fact: str, slot: str | None, runtime: ToolRuntime[UserContext]) -> str:
        """Save one durable fact about the user to long-term memory.

        Args:
            fact: the fact as one short third-person sentence, e.g. "Lives in Lisbon."
            slot: a stable snake_case attribute (name, home_city, diet, occupation...)
                so that a new value replaces the old one; null if none fits.
        """
        user_id, store = _context(runtime)
        result = upsert_fact(
            store, user_id, fact, slot, source="hot", dedup_threshold=dedup_threshold
        )
        if result.action == "refused":
            return "Not saved: memory never stores secrets or empty facts."
        return f"Memory {result.action}: {result.content}"

    @tool
    def forget(query: str, runtime: ToolRuntime[UserContext]) -> str:
        """Delete the fact the user asked you to forget (a slot name or a description)."""
        user_id, store = _context(runtime)
        deleted = forget_fact(store, user_id, query)
        return "Forgotten." if deleted else "I had nothing like that stored."

    return [remember, forget]
