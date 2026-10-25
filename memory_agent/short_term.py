"""Short-term memory: the thread's messages, kept inside a token budget.

Two mechanisms, and they do different jobs:

* **Trimming** (``fit_to_budget``) is a *view*: before each model call the oldest
  messages that don't fit ``max_context_tokens`` are left out of the prompt. The
  checkpoint still has them. It is free, but whatever falls off is invisible.
* **Summarization** (``plan_summary`` + the ``summarize`` node) *rewrites state*: once
  the thread's messages exceed ``summarize_after_tokens``, everything but the last
  ``keep_recent_tokens`` is folded into a running summary and removed from the
  checkpoint with ``RemoveMessage``. It costs one model call, and keeps the gist.

Both cut only at a ``HumanMessage``, so a tool call is never separated from its result.
Tokens are counted with ``count_tokens_approximately`` (about 4 characters per token):
model-agnostic and deterministic, which is what a budget test needs. Swap in the
model's own tokenizer for production.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately

__all__ = ["count_tokens", "fit_to_budget", "plan_summary"]


def count_tokens(messages: Sequence[BaseMessage]) -> int:
    return int(count_tokens_approximately(list(messages)))


def plan_summary(messages: Sequence[BaseMessage], *, trigger: int, keep: int) -> int:
    """How many leading messages to fold into the summary (0: none yet).

    The kept tail is the longest suffix within ``keep`` tokens that starts at a
    ``HumanMessage``; the newest human message is always kept, even if it alone is
    larger than ``keep``.
    """
    if len(messages) < 2 or count_tokens(messages) <= trigger:
        return 0
    humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if not humans:
        return 0
    cut = humans[-1]
    for i in reversed(humans[:-1]):
        if count_tokens(messages[i:]) > keep:
            break
        cut = i
    return cut


def fit_to_budget(
    system: SystemMessage, messages: Sequence[BaseMessage], *, max_tokens: int
) -> list[BaseMessage]:
    """``[system, *the newest messages that fit]``, starting on a human turn.

    The system prompt is always kept; if even the newest turn doesn't fit, only that
    human message is sent (and the model sees a truncated history rather than none).
    """
    trimmed = trim_messages(
        [system, *messages],
        max_tokens=max_tokens,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )
    if len(trimmed) <= 1:
        last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        return [system] if last_human is None else [system, last_human]
    return list(trimmed)
