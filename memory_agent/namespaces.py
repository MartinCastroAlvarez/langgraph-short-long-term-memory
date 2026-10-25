"""Who owns what: validated user and thread ids, and the store namespaces built from them.

Long-term memory lives in the store under ``("memories", <user_id>, <kind>)``. The user
id comes from the run's context (``context=UserContext(user_id=...)``), set by the
application after it authenticated the person; it is **never** a tool argument, so the
model cannot read or write another user's namespace by naming it. Ids are validated
against a strict pattern before any namespace or checkpoint thread id is built from
them, so ``"alice/../bob"``, ``"*"`` or ``"alice.bob"`` never reach the store.

Short-term memory lives in the checkpointer under the thread id ``"<user_id>/<thread_id>"``:
two users who both call their thread ``"main"`` get two different threads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "KINDS",
    "ROOT",
    "InvalidIdError",
    "Kind",
    "UserContext",
    "checkpoint_thread",
    "namespace",
    "user_prefix",
    "validate_id",
    "validate_slot",
]

ROOT = "memories"
Kind = Literal["facts", "episodes", "instructions", "threads"]
KINDS: tuple[Kind, ...] = ("facts", "episodes", "instructions", "threads")

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_SLOT = re.compile(r"[a-z][a-z0-9_]{0,39}")


class InvalidIdError(ValueError):
    """A user id, thread id or slot that could escape its namespace."""


def validate_id(value: object, what: str = "id") -> str:
    """``[A-Za-z0-9][A-Za-z0-9_-]{0,63}``: no dots, slashes, wildcards or whitespace."""
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise InvalidIdError(f"invalid {what}: {value!r}")
    return value


def validate_slot(value: object) -> str:
    """A fact slot (``name``, ``home_city``...): lowercase snake case, at most 40 chars."""
    if not isinstance(value, str):
        raise InvalidIdError(f"invalid slot: {value!r}")
    slot = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")[:40]
    if not _SLOT.fullmatch(slot):
        raise InvalidIdError(f"invalid slot: {value!r}")
    return slot


@dataclass(frozen=True, slots=True)
class UserContext:
    """The run context: whose memory this run may touch. Validated on construction."""

    user_id: str

    def __post_init__(self) -> None:
        validate_id(self.user_id, "user id")


def user_prefix(user_id: str) -> tuple[str, str]:
    return (ROOT, validate_id(user_id, "user id"))


def namespace(user_id: str, kind: Kind) -> tuple[str, str, str]:
    if kind not in KINDS:
        raise InvalidIdError(f"unknown memory kind {kind!r}")
    return (*user_prefix(user_id), kind)


def checkpoint_thread(user_id: str, thread_id: str) -> str:
    """The checkpointer thread id: per-user by construction."""
    return f"{validate_id(user_id, 'user id')}/{validate_id(thread_id, 'thread id')}"
