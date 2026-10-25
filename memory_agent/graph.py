"""The chat graph: short-term memory in the checkpointer, long-term memory in the store.

    START -> [summarize] -> recall -> agent <-> tools -> END

* ``summarize`` runs only when the thread's messages exceed ``summarize_after_tokens``:
  it folds the older turns into ``state["summary"]`` and removes them from the
  checkpoint (``RemoveMessage``).
* ``recall`` searches the **current user's** namespaces in the store (facts and episodes
  by semantic similarity to the new message, plus the style rules).
* ``agent`` builds the system prompt from the recalled memories (as untrusted ``<data>``
  blocks) and the summary, trims the messages to ``max_context_tokens`` and calls the
  model, with the ``remember``/``forget`` tools when the write mode includes ``hot``.

Compile it with a checkpointer (per-thread state, time travel) and a store (per-user
memory, shared across threads), and run it with ``context=UserContext(user_id=...)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import RemoveMessage
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from memory_agent.config import Settings
from memory_agent.long_term import recall as recall_memories
from memory_agent.namespaces import UserContext
from memory_agent.prompts import assistant_prompt
from memory_agent.prompts import summarizer_prompt
from memory_agent.prompts import text_of
from memory_agent.short_term import count_tokens
from memory_agent.short_term import fit_to_budget
from memory_agent.short_term import plan_summary
from memory_agent.tools import make_tools

__all__ = ["ChatState", "build_chat_graph", "system_prompt"]

SYSTEM_SHARE = 0.6
"""The system prompt (rules + memories + summary) may use at most this share of the budget."""


class ChatState(MessagesState):
    summary: str
    recalled: dict[str, Any]
    summarized: int


def _last_human(messages: Sequence[BaseMessage]) -> str:
    return next((text_of(m) for m in reversed(messages) if isinstance(m, HumanMessage)), "")


def system_prompt(recalled: dict[str, Any], summary: str, *, max_tokens: int) -> SystemMessage:
    """The assistant's system prompt, dropping the lowest-ranked memories until it fits."""
    facts = [f["content"] for f in recalled.get("facts", [])]
    episodes = list(recalled.get("episodes", []))
    rules = list(recalled.get("rules", []))
    while True:
        message = SystemMessage(
            assistant_prompt(facts=facts, episodes=episodes, rules=rules, summary=summary)
        )
        if count_tokens([message]) <= max_tokens:
            return message
        if episodes:
            episodes.pop()
        elif facts:
            facts.pop()
        elif summary:
            summary = summary[: len(summary) // 2]
        elif rules:
            rules.pop()
        else:
            return message


def build_chat_graph(
    settings: Settings,
    model: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    hot = settings.write_mode in ("hot", "both")
    tools = make_tools(dedup_threshold=settings.dedup_threshold) if hot else []
    agent_model = model.bind_tools(tools) if tools else model
    system_budget = int(settings.max_context_tokens * SYSTEM_SHARE)

    def route(state: ChatState) -> str:
        total = count_tokens(state["messages"])
        return "summarize" if total > settings.summarize_after_tokens else "recall"

    def summarize(state: ChatState) -> dict[str, Any]:
        messages = state["messages"]
        cut = plan_summary(
            messages,
            trigger=settings.summarize_after_tokens,
            keep=settings.keep_recent_tokens,
        )
        if cut == 0:
            return {"summarized": 0}
        prompt = summarizer_prompt(state.get("summary", ""), messages[:cut])
        reply = model.invoke([SystemMessage(prompt), HumanMessage("Write the updated summary.")])
        summary = " ".join(text_of(reply).split())[: settings.summary_max_chars]
        removals = [RemoveMessage(id=m.id) for m in messages[:cut] if m.id]
        return {"summary": summary, "messages": removals, "summarized": cut}

    def recall(state: ChatState, runtime: Runtime[UserContext]) -> dict[str, Any]:
        if runtime.store is None:
            return {"recalled": {}}
        found = recall_memories(
            runtime.store,
            runtime.context.user_id,
            _last_human(state["messages"]),
            facts=settings.recall_facts,
            episodes=settings.recall_episodes,
        )
        return {"recalled": found.as_dict()}

    def agent(state: ChatState) -> dict[str, Any]:
        system = system_prompt(
            state.get("recalled", {}), state.get("summary", ""), max_tokens=system_budget
        )
        window = fit_to_budget(system, state["messages"], max_tokens=settings.max_context_tokens)
        reply = agent_model.invoke(window)
        if not isinstance(reply, AIMessage):  # pragma: no cover - every chat model returns one
            reply = AIMessage(content=text_of(reply))
        return {"messages": [reply]}

    builder = StateGraph(ChatState, context_schema=UserContext)
    builder.add_node("summarize", summarize)
    builder.add_node("recall", recall)
    builder.add_node("agent", agent)
    builder.add_conditional_edges(START, route, ["summarize", "recall"])
    builder.add_edge("summarize", "recall")
    builder.add_edge("recall", "agent")
    if tools:
        builder.add_node("tools", ToolNode(tools))
        builder.add_conditional_edges("agent", tools_condition, ["tools", END])
        builder.add_edge("tools", "agent")
    else:
        builder.add_edge("agent", END)
    return builder.compile(checkpointer=checkpointer, store=store, name="chat")
