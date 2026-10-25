"""Time travel: checkpoint history, replay and fork."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from memory_agent import Assistant


def _human(bot: Assistant, user: str, thread: str) -> list[str]:
    return [str(m.content) for m in bot.messages(user, thread) if isinstance(m, HumanMessage)]


def test_history_is_newest_first_with_a_checkpoint_per_step(bot: Assistant) -> None:
    bot.chat("u", "t", "first")
    bot.chat("u", "t", "second")
    history = bot.history("u", "t")
    steps = [c.step for c in history]
    assert steps == sorted(steps, reverse=True)
    assert history[0].next == () and history[0].messages == 4
    assert history[-1].messages == 0
    assert sum(1 for c in history if c.next == ("__start__",)) == 2  # one input per turn


def test_fork_branches_from_a_past_checkpoint(bot: Assistant) -> None:
    bot.chat("u", "t", "first")
    bot.chat("u", "t", "second")
    before_second = [c for c in bot.history("u", "t") if c.next == ("__start__",)][0]
    assert before_second.messages == 2
    reply = bot.fork("u", "t", before_second.checkpoint_id, "second, rewritten")
    assert "second, rewritten" in reply.text
    assert _human(bot, "u", "t") == ["first", "second, rewritten"]
    # the original branch is still in the history
    assert any(c.last.startswith("Happy to help with that: second") for c in bot.history("u", "t"))


def test_replay_reruns_the_nodes_after_a_checkpoint(bot: Assistant) -> None:
    bot.chat("u", "t", "first")
    original = bot.messages("u", "t")[-1]
    before_agent = [c for c in bot.history("u", "t") if c.next == ("agent",)][0]
    replayed = bot.replay("u", "t", before_agent.checkpoint_id)
    assert replayed.text == str(original.content)
    assert _human(bot, "u", "t") == ["first"]
    assert bot.messages("u", "t")[-1].id != original.id  # a new run, not the cached message


def test_replay_of_a_finished_turn_is_refused(bot: Assistant) -> None:
    bot.chat("u", "t", "first")
    with pytest.raises(ValueError, match="nothing to replay"):
        bot.replay("u", "t", bot.history("u", "t")[0].checkpoint_id)


def test_checkpoints_of_another_thread_or_user_are_not_reachable(bot: Assistant) -> None:
    bot.chat("alice", "t", "alice's message")
    bot.chat("bob", "t", "bob's message")
    alice_cp = bot.history("alice", "t")[1].checkpoint_id
    with pytest.raises(KeyError):
        bot.fork("bob", "t", alice_cp, "steal")
    with pytest.raises(KeyError):
        bot.replay("alice", "other", alice_cp)
    assert _human(bot, "bob", "t") == ["bob's message"]


def test_fork_validates_the_message(bot: Assistant) -> None:
    bot.chat("u", "t", "first")
    cp = bot.history("u", "t")[-1].checkpoint_id
    with pytest.raises(ValueError):
        bot.fork("u", "t", cp, "   ")
