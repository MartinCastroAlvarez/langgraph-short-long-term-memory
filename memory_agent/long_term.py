"""Long-term memory: what one user's namespace in the store holds, and how it changes.

Three kinds, after the CoALA taxonomy that LangMem uses:

* **semantic** - ``("memories", user, "facts")``: one item per fact. A fact with a
  ``slot`` (``name``, ``home_city``...) is stored under that key, so a new value
  *updates* the old one (``version`` goes up, the old text goes to ``history``); a
  fact without a slot updates the most similar existing fact when the similarity is at
  least ``dedup_threshold``, and is added otherwise. Re-running extraction never
  duplicates.
* **episodic** - ``("memories", user, "episodes")``: one summary per conversation
  (keyed by thread), recalled by similarity as worked examples.
* **procedural** - ``("memories", user, "instructions")``: the assistant's own style
  rules for this user, rewritten from feedback.

``("memories", user, "threads")`` lists the user's threads, so "forget me" can delete
their checkpoints too. Every function takes the user id and builds the namespace
itself (``namespaces.namespace``); none takes a namespace from a caller.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from typing import Any
from typing import Literal

from langgraph.store.base import BaseStore
from langgraph.store.base import Item

from memory_agent.namespaces import InvalidIdError
from memory_agent.namespaces import namespace
from memory_agent.namespaces import user_prefix
from memory_agent.namespaces import validate_id
from memory_agent.namespaces import validate_slot
from memory_agent.schemas import MAX_FACT_CHARS
from memory_agent.schemas import Episode
from memory_agent.schemas import Instructions

__all__ = [
    "FORGET_THRESHOLD",
    "FactWrite",
    "Recalled",
    "RecalledFact",
    "forget_fact",
    "get_rules",
    "get_watermark",
    "iter_items",
    "list_episodes",
    "list_facts",
    "list_threads",
    "looks_secret",
    "recall",
    "register_thread",
    "save_episode",
    "save_rules",
    "set_watermark",
    "upsert_fact",
    "wipe_user",
]

FORGET_THRESHOLD = 0.5
_INDEX = ["content"]
_HISTORY = 5
_WATERMARK = 1000
_SECRET = re.compile(
    r"(passw(?:or)?d|passphrase|api[ _-]?key|secret|token|\bpin\b|\bcvv\b|"
    r"\b(?:\d[ -]?){13,19}\b|\b\d{3}-\d{2}-\d{4}\b|sk-[A-Za-z0-9]{8,})",
    re.I,
)


def looks_secret(text: str) -> bool:
    """Credentials, card or social security numbers: never written to memory."""
    return bool(_SECRET.search(text))


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def iter_items(store: BaseStore, ns: tuple[str, ...], page: int = 100) -> Iterator[Item]:
    """Every item under ``ns`` (a prefix), page by page."""
    offset = 0
    while True:
        batch = store.search(ns, limit=page, offset=offset)
        yield from batch
        if len(batch) < page:
            return
        offset += page


# --------------------------------------------------------------------------- #
# Semantic memory: facts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FactWrite:
    action: Literal["added", "updated", "unchanged", "refused"]
    key: str | None
    content: str


def upsert_fact(
    store: BaseStore,
    user_id: str,
    content: str,
    slot: str | None = None,
    *,
    source: str,
    dedup_threshold: float,
) -> FactWrite:
    """Add a fact, or update the one it supersedes. Never stores a duplicate."""
    ns = namespace(user_id, "facts")
    text = " ".join(content.split())[:MAX_FACT_CHARS]
    if not text or looks_secret(text):
        return FactWrite("refused", None, text)
    slot_key: str | None = None
    if slot:
        try:
            slot_key = validate_slot(slot)
        except InvalidIdError:
            slot_key = None
    key = slot_key
    existing = store.get(ns, key) if key else None
    if key is None:
        hits = store.search(ns, query=text, limit=1)
        if hits and hits[0].score is not None and hits[0].score >= dedup_threshold:
            existing = hits[0]
            key = existing.key
        else:
            key = "fact-" + hashlib.sha256(text.lower().encode()).hexdigest()[:12]
            existing = store.get(ns, key)
    if existing is not None:
        old = dict(existing.value)
        if _words(str(old.get("content", ""))) == _words(text):
            return FactWrite("unchanged", key, text)
        history = [*list(old.get("history", [])), old.get("content", "")][-_HISTORY:]
        value = {
            **old,
            "content": text,
            "version": int(old.get("version", 1)) + 1,
            "history": history,
            "source": source,
            "updated": _now(),
        }
        store.put(ns, key, value, index=_INDEX)
        return FactWrite("updated", key, text)
    value = {
        "content": text,
        "slot": slot_key,
        "version": 1,
        "history": [],
        "source": source,
        "updated": _now(),
    }
    store.put(ns, key, value, index=_INDEX)
    return FactWrite("added", key, text)


def forget_fact(store: BaseStore, user_id: str, query: str) -> list[str]:
    """Delete the fact a user asked to forget: by slot name, else the most similar one."""
    ns = namespace(user_id, "facts")
    try:
        slot = validate_slot(query)
    except InvalidIdError:
        slot = None
    if slot and store.get(ns, slot) is not None:
        store.delete(ns, slot)
        return [slot]
    hits = store.search(ns, query=query[:MAX_FACT_CHARS], limit=1)
    if hits and hits[0].score is not None and hits[0].score >= FORGET_THRESHOLD:
        store.delete(ns, hits[0].key)
        return [hits[0].key]
    return []


def list_facts(store: BaseStore, user_id: str) -> list[Item]:
    return sorted(iter_items(store, namespace(user_id, "facts")), key=lambda i: i.key)


# --------------------------------------------------------------------------- #
# Episodic and procedural memory
# --------------------------------------------------------------------------- #
def episode_text(value: dict[str, Any]) -> str:
    return (
        f"{value.get('situation', '')} | {value.get('approach', '')} | {value.get('outcome', '')}"
    )


def save_episode(store: BaseStore, user_id: str, thread_id: str, episode: Episode) -> None:
    value = {**episode.model_dump(), "thread": validate_id(thread_id, "thread id")}
    value["content"] = episode_text(value)
    value["updated"] = _now()
    store.put(namespace(user_id, "episodes"), thread_id, value, index=_INDEX)


def list_episodes(store: BaseStore, user_id: str) -> list[Item]:
    return list(iter_items(store, namespace(user_id, "episodes")))


def get_rules(store: BaseStore, user_id: str) -> list[str]:
    item = store.get(namespace(user_id, "instructions"), "style")
    if item is None:
        return []
    return Instructions.model_validate({"rules": item.value.get("rules", [])}).rules


def save_rules(store: BaseStore, user_id: str, rules: list[str]) -> list[str]:
    clean = Instructions(rules=[r for r in rules if not looks_secret(r)]).rules
    ns = namespace(user_id, "instructions")
    old = store.get(ns, "style")
    version = int(old.value.get("version", 0)) + 1 if old else 1
    store.put(ns, "style", {"rules": clean, "version": version, "updated": _now()}, index=False)
    return clean


# --------------------------------------------------------------------------- #
# Recall (the read path)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RecalledFact:
    key: str
    content: str
    score: float | None


@dataclass(frozen=True, slots=True)
class Recalled:
    facts: list[RecalledFact] = field(default_factory=list)
    episodes: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "facts": [{"key": f.key, "content": f.content, "score": f.score} for f in self.facts],
            "episodes": list(self.episodes),
            "rules": list(self.rules),
        }


def recall(store: BaseStore, user_id: str, query: str, *, facts: int, episodes: int) -> Recalled:
    """Semantic search in this user's namespaces only, ranked by the store's score."""
    q = query[:1000]
    ns = namespace(user_id, "facts")
    found = list(store.search(ns, query=q or None, limit=facts)) if facts else []
    if len(found) < facts:
        # Top up with the most recent facts: a lexical store (BM25) returns nothing for
        # "what do you know about me?", and a small profile is worth always showing.
        seen = {i.key for i in found}
        found += [i for i in store.search(ns, limit=facts) if i.key not in seen][
            : facts - len(found)
        ]
    past = (
        store.search(namespace(user_id, "episodes"), query=q or None, limit=episodes)
        if episodes
        else []
    )
    return Recalled(
        facts=[RecalledFact(i.key, str(i.value.get("content", "")), i.score) for i in found],
        episodes=[str(i.value.get("content", "")) for i in past],
        rules=get_rules(store, user_id),
    )


# --------------------------------------------------------------------------- #
# Threads and "forget me"
# --------------------------------------------------------------------------- #
def register_thread(store: BaseStore, user_id: str, thread_id: str) -> None:
    ns = namespace(user_id, "threads")
    tid = validate_id(thread_id, "thread id")
    if store.get(ns, tid) is None:
        store.put(ns, tid, {"created": _now()}, index=False)


def get_watermark(store: BaseStore, user_id: str, thread_id: str) -> set[str]:
    """Ids of the thread's messages that were already consolidated."""
    item = store.get(namespace(user_id, "threads"), validate_id(thread_id, "thread id"))
    return set(item.value.get("consolidated", [])) if item else set()


def set_watermark(store: BaseStore, user_id: str, thread_id: str, ids: list[str]) -> None:
    ns = namespace(user_id, "threads")
    tid = validate_id(thread_id, "thread id")
    old = store.get(ns, tid)
    value: dict[str, Any] = dict(old.value) if old else {"created": _now()}
    value["consolidated"] = ids[-_WATERMARK:]
    store.put(ns, tid, value, index=False)


def list_threads(store: BaseStore, user_id: str) -> list[str]:
    return sorted(i.key for i in iter_items(store, namespace(user_id, "threads")))


def wipe_user(store: BaseStore, user_id: str) -> int:
    """Delete every item in every namespace under ``("memories", user_id)``."""
    prefix = user_prefix(user_id)
    deleted = 0
    for ns in store.list_namespaces(prefix=prefix, limit=1000):
        keys = [i.key for i in iter_items(store, ns)]
        for key in keys:
            store.delete(ns, key)
        deleted += len(keys)
    return deleted
