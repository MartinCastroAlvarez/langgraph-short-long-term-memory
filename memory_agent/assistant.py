"""``Assistant``: one object that owns both memories and the jobs around them.

    with Assistant.open(Settings.offline()) as bot:
        bot.chat("alice", "a", "Hi, my name is Ada and I live in Lisbon.")
        bot.run_pending()                     # the deferred consolidation job
        bot.chat("alice", "b", "What do you know about me?").text

Every method takes the user id explicitly; the application is responsible for having
authenticated that user. The user id becomes the run context (store namespaces) and the
prefix of the checkpointer thread id, so neither memory can be reached with another
user's id by a model, a tool argument or a thread name.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from memory_agent.config import Settings
from memory_agent.consolidation import Consolidator
from memory_agent.consolidation import build_consolidation_graph
from memory_agent.embeddings import make_embeddings
from memory_agent.graph import build_chat_graph
from memory_agent.llm import make_model
from memory_agent.long_term import get_rules
from memory_agent.long_term import get_watermark
from memory_agent.long_term import list_episodes
from memory_agent.long_term import list_facts
from memory_agent.long_term import list_threads
from memory_agent.long_term import register_thread
from memory_agent.long_term import set_watermark
from memory_agent.long_term import wipe_user
from memory_agent.namespaces import UserContext
from memory_agent.namespaces import checkpoint_thread
from memory_agent.namespaces import validate_id
from memory_agent.prompts import data_block
from memory_agent.prompts import text_of
from memory_agent.prompts import transcript
from memory_agent.short_term import count_tokens
from memory_agent.stores import open_checkpointer
from memory_agent.stores import open_store

__all__ = ["MAX_MESSAGE_CHARS", "Assistant", "CheckpointInfo", "Reply"]

MAX_MESSAGE_CHARS = 4000
RECURSION_LIMIT = 30


@dataclass(frozen=True, slots=True)
class Reply:
    text: str
    user_id: str
    thread_id: str
    checkpoint_id: str | None
    recalled: dict[str, Any]
    tool_calls: list[str]
    summarized: int
    thread_tokens: int
    """Tokens of the messages kept in the checkpoint after this turn."""


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    checkpoint_id: str
    step: int
    next: tuple[str, ...]
    messages: int
    last: str


class Assistant:
    def __init__(
        self,
        settings: Settings,
        *,
        model: BaseChatModel,
        store: BaseStore,
        checkpointer: BaseCheckpointSaver[Any],
    ) -> None:
        self.settings = settings
        self.model = model
        self.store = store
        self.checkpointer = checkpointer
        self.graph = build_chat_graph(settings, model, checkpointer=checkpointer, store=store)
        self.consolidation = build_consolidation_graph(settings, model, store=store)
        self.consolidator = Consolidator(run=self.consolidate)

    @classmethod
    @contextlib.contextmanager
    def open(
        cls,
        settings: Settings,
        *,
        threads: str = ":memory:",
        memories: str = ":memory:",
        model: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
    ) -> Iterator[Assistant]:
        """``threads``: the SQLite checkpointer file. ``memories``: a store URL or file."""
        emb = embeddings or make_embeddings(settings)
        with (
            open_checkpointer(threads) as saver,
            open_store(memories, emb, settings.embed_dims) as store,
        ):
            bot = cls(
                settings, model=model or make_model(settings), store=store, checkpointer=saver
            )
            yield bot
            bot.run_pending()  # the conversation is over: run the deferred jobs

    # ------------------------------------------------------------------ #
    def config(
        self, user_id: str, thread_id: str, checkpoint_id: str | None = None
    ) -> RunnableConfig:
        configurable: dict[str, Any] = {"thread_id": checkpoint_thread(user_id, thread_id)}
        if checkpoint_id is not None:
            configurable["checkpoint_id"] = checkpoint_id
        return {"configurable": configurable, "recursion_limit": RECURSION_LIMIT}

    def _run(
        self, user_id: str, thread_id: str, inputs: dict[str, Any] | None, checkpoint_id: str | None
    ) -> Reply:
        context = UserContext(user_id=user_id)
        register_thread(self.store, user_id, thread_id)
        final = self.graph.invoke(
            inputs, self.config(user_id, thread_id, checkpoint_id), context=context
        )
        if self.settings.write_mode in ("background", "both"):
            self.consolidator.schedule(user_id, thread_id)
        return self._reply(user_id, thread_id, final)

    def _reply(self, user_id: str, thread_id: str, final: dict[str, Any]) -> Reply:
        messages: list[BaseMessage] = list(final.get("messages", []))
        last_human = max(
            (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1
        )
        turn = messages[last_human + 1 :]
        answer = next((m for m in reversed(turn) if isinstance(m, AIMessage)), None)
        snapshot = self.graph.get_state(self.config(user_id, thread_id))
        cfg = snapshot.config.get("configurable", {}) if snapshot.config else {}
        return Reply(
            text=text_of(answer) if answer is not None else "",
            user_id=user_id,
            thread_id=thread_id,
            checkpoint_id=cfg.get("checkpoint_id"),
            recalled=dict(final.get("recalled", {})),
            tool_calls=[c["name"] for m in turn if isinstance(m, AIMessage) for c in m.tool_calls],
            summarized=int(final.get("summarized", 0)),
            thread_tokens=count_tokens(messages),
        )

    def chat(self, user_id: str, thread_id: str, text: str) -> Reply:
        """One turn in ``thread_id`` for ``user_id``."""
        validate_id(thread_id, "thread id")
        if not text.strip() or len(text) > MAX_MESSAGE_CHARS:
            raise ValueError(f"a message must have 1-{MAX_MESSAGE_CHARS} characters")
        inputs = {"messages": [HumanMessage(text)], "summarized": 0}
        return self._run(user_id, thread_id, inputs, None)

    # -- short-term memory: state and time travel ---------------------------- #
    def messages(self, user_id: str, thread_id: str) -> list[BaseMessage]:
        values = self.graph.get_state(self.config(user_id, thread_id)).values
        return list(values.get("messages", [])) if values else []

    def summary(self, user_id: str, thread_id: str) -> str:
        values = self.graph.get_state(self.config(user_id, thread_id)).values
        return str(values.get("summary", "")) if values else ""

    def history(self, user_id: str, thread_id: str) -> list[CheckpointInfo]:
        """Every checkpoint of the thread, newest first."""
        out: list[CheckpointInfo] = []
        for snap in self.graph.get_state_history(self.config(user_id, thread_id)):
            msgs = list(snap.values.get("messages", []))
            out.append(
                CheckpointInfo(
                    checkpoint_id=str(snap.config["configurable"]["checkpoint_id"]),
                    step=int((snap.metadata or {}).get("step", -1)),
                    next=tuple(snap.next),
                    messages=len(msgs),
                    last=text_of(msgs[-1])[:80] if msgs else "",
                )
            )
        return out

    def _checkpoint(self, user_id: str, thread_id: str, checkpoint_id: str) -> CheckpointInfo:
        for info in self.history(user_id, thread_id):
            if info.checkpoint_id == checkpoint_id:
                return info
        raise KeyError(f"no checkpoint {checkpoint_id!r} in thread {thread_id!r}")

    def replay(self, user_id: str, thread_id: str, checkpoint_id: str) -> Reply:
        """Re-run the nodes that were still to run at ``checkpoint_id``."""
        info = self._checkpoint(user_id, thread_id, checkpoint_id)
        if not info.next:
            raise ValueError("that checkpoint ends a turn: nothing to replay (use fork)")
        return self._run(user_id, thread_id, None, checkpoint_id)

    def fork(self, user_id: str, thread_id: str, checkpoint_id: str, text: str) -> Reply:
        """Send ``text`` from ``checkpoint_id``, branching the thread from there."""
        self._checkpoint(user_id, thread_id, checkpoint_id)
        if not text.strip() or len(text) > MAX_MESSAGE_CHARS:
            raise ValueError(f"a message must have 1-{MAX_MESSAGE_CHARS} characters")
        inputs = {"messages": [HumanMessage(text)], "summarized": 0}
        return self._run(user_id, thread_id, inputs, checkpoint_id)

    # -- long-term memory ----------------------------------------------------- #
    def consolidate(self, user_id: str, thread_id: str) -> dict[str, Any]:
        """The background job for one thread (normally run by ``run_pending``)."""
        values = self.graph.get_state(self.config(user_id, thread_id)).values or {}
        messages: list[BaseMessage] = list(values.get("messages", []))
        done = get_watermark(self.store, user_id, thread_id)
        new = [m for m in messages if m.id not in done]
        if not new:
            return {"user_id": user_id, "thread_id": thread_id, "skipped": True}
        # Facts and feedback come only from messages not consolidated before, so re-running
        # an old thread can't overwrite a newer fact with a stale one. The summary is not a
        # source of facts for the same reason (it repeats old turns); the episode sees it all.
        conversation = transcript(messages)
        if values.get("summary"):
            conversation = data_block("summary", str(values["summary"])) + "\n" + conversation
        result = self.consolidation.invoke(
            {"thread_id": thread_id, "transcript": transcript(new), "conversation": conversation},
            {"recursion_limit": RECURSION_LIMIT},
            context=UserContext(user_id=user_id),
        )
        set_watermark(self.store, user_id, thread_id, [m.id for m in messages if m.id])
        return {"user_id": user_id, "thread_id": thread_id, **result}

    def run_pending(self) -> list[dict[str, Any]]:
        return self.consolidator.run_pending()

    def memories(self, user_id: str) -> dict[str, Any]:
        """Everything long-term memory holds about ``user_id`` (for "what do you know")."""
        return {
            "facts": {i.key: i.value for i in list_facts(self.store, user_id)},
            "episodes": {i.key: i.value for i in list_episodes(self.store, user_id)},
            "rules": get_rules(self.store, user_id),
            "threads": list_threads(self.store, user_id),
        }

    def forget_user(self, user_id: str) -> dict[str, Any]:
        """ "Forget me": every checkpoint of every thread, and the whole store namespace."""
        cancelled = self.consolidator.cancel_user(validate_id(user_id, "user id"))
        threads = list_threads(self.store, user_id)
        for thread_id in threads:
            self.checkpointer.delete_thread(checkpoint_thread(user_id, thread_id))
        items = wipe_user(self.store, user_id)
        return {"threads": threads, "items": items, "cancelled_jobs": cancelled}
