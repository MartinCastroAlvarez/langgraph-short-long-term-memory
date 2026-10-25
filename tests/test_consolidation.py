"""The background path: extraction, episodes, procedural rules, idempotence, debounce."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from memory_agent import Assistant
from memory_agent import Consolidator
from memory_agent import FakeAssistantModel
from memory_agent import Settings
from memory_agent.prompts import parse_blocks

BACKGROUND = Settings.offline(write_mode="background")


def test_background_mode_has_no_tools_and_learns_after_the_job_runs() -> None:
    model = FakeAssistantModel()
    with Assistant.open(BACKGROUND, model=model) as bot:
        reply = bot.chat("u", "a", "Hi, my name is Ada and I live in Lisbon.")
        assert reply.tool_calls == [] and "tools" not in bot.graph.get_graph().nodes
        assert bot.memories("u")["facts"] == {}  # nothing written on the hot path
        assert bot.consolidator.pending == [("u", "a")]
        [job] = bot.run_pending()
        assert {w["key"]: w["action"] for w in job["writes"]} == {
            "name": "added",
            "home_city": "added",
        }
        facts = bot.memories("u")["facts"]
        assert facts["name"]["source"] == "background"
        assert "Ada" in bot.chat("u", "b", "What is my name?").text


def test_hot_mode_writes_immediately_and_schedules_nothing() -> None:
    with Assistant.open(Settings.offline(write_mode="hot")) as bot:
        reply = bot.chat("u", "a", "My name is Ada.")
        assert reply.tool_calls == ["remember"]
        assert bot.memories("u")["facts"]["name"]["source"] == "hot"
        assert bot.consolidator.pending == []


def test_the_job_is_debounced_and_idempotent(bot: Assistant) -> None:
    for text in ("My name is Ada.", "I live in Lisbon.", "Please be more concise."):
        bot.chat("u", "a", text)
    assert bot.consolidator.pending == [("u", "a")]
    first = bot.run_pending()
    assert len(first) == 1 and first[0]["rules"] == ["Be more concise."]
    snapshot = bot.memories("u")
    assert bot.consolidate("u", "a") == {"user_id": "u", "thread_id": "a", "skipped": True}
    assert bot.memories("u") == snapshot


def test_an_old_thread_never_overwrites_a_newer_fact(bot: Assistant) -> None:
    bot.chat("u", "a", "I live in Lisbon.")
    bot.run_pending()
    bot.chat("u", "b", "Actually, I moved to Porto.")
    bot.chat("u", "a", "Thanks!")  # thread a gets a new message: it is consolidated again
    bot.run_pending()
    city = bot.memories("u")["facts"]["home_city"]
    assert city["content"] == "Lives in Porto." and city["version"] == 2


def test_episodes_are_recalled_as_examples_in_another_thread() -> None:
    model = FakeAssistantModel()
    with Assistant.open(Settings.offline(), model=model) as bot:
        bot.chat("u", "a", "Help me plan a vegetarian dinner party.")
        bot.run_pending()
        episodes = bot.memories("u")["episodes"]
        assert list(episodes) == ["a"] and "dinner party" in episodes["a"]["situation"]
        reply = bot.chat("u", "b", "Plan another dinner party for Friday.")
        prompt = parse_blocks(str(model.calls_for("assistant")[-1].messages[0].content))
        assert "dinner party" in prompt["episodes"]
        assert "Last time:" in reply.text


def test_feedback_rewrites_the_procedural_rules_used_in_new_threads() -> None:
    model = FakeAssistantModel()
    with Assistant.open(Settings.offline(), model=model) as bot:
        long_before = bot.chat("u", "x", "Explain vector databases.").text
        bot.chat("u", "a", "Please use bullet points.")
        bot.chat("u", "a", "And be more concise.")
        bot.run_pending()
        assert bot.memories("u")["rules"] == ["Use bullet points.", "Be more concise."]
        after = bot.chat("u", "b", "Explain vector databases.").text
        assert after.startswith("- ") and len(after) < len(long_before)
        style = parse_blocks(str(model.calls_for("assistant")[-1].messages[0].content))["style"]
        assert style == "- Use bullet points.\n- Be more concise."


def test_malformed_structured_output_writes_nothing() -> None:
    bad = AIMessage("", tool_calls=[{"name": "MemoryUpdate", "args": {"facts": 3}, "id": "x"}])
    model = FakeAssistantModel(
        script={"memory-writer": [bad], "episode-writer": ["not json at all"]}
    )
    with Assistant.open(Settings.offline(write_mode="background"), model=model) as bot:
        bot.chat("u", "a", "My name is Ada.")
        [job] = bot.run_pending()
        assert job["writes"] == [] and job["episode"] == {}
        assert bot.memories("u")["facts"] == {} and bot.memories("u")["episodes"] == {}


def test_a_failing_optimizer_keeps_the_current_rules() -> None:
    model = FakeAssistantModel(script={"optimizer": ["garbage"]})
    with Assistant.open(Settings.offline(), model=model) as bot:
        bot.chat("u", "a", "Please be more concise.")
        [job] = bot.run_pending()
        assert job["rules"] == [] and bot.memories("u")["rules"] == []


def test_forgetting_in_the_background(bot: Assistant) -> None:
    with Assistant.open(Settings.offline(write_mode="background")) as bg:
        bg.chat("u", "a", "I am vegetarian.")
        bg.run_pending()
        assert "diet" in bg.memories("u")["facts"]
        bg.chat("u", "a", "Forget that I am vegetarian.")
        [job] = bg.run_pending()
        assert job["writes"] == [{"action": "deleted", "key": "diet", "content": "I am vegetarian"}]
        assert bg.memories("u")["facts"] == {}


def test_hot_path_forget_tool(bot: Assistant) -> None:
    bot.chat("u", "a", "My dog is called Pixel.")
    assert "pet" in bot.memories("u")["facts"]
    reply = bot.chat("u", "a", "Please forget my pet Pixel.")
    assert reply.tool_calls == ["forget"] and "Forgotten" in reply.text
    assert bot.chat("u", "a", "Forget about quantum physics.").text.startswith("Noted (I had")


def test_consolidator_queue() -> None:
    ran: list[tuple[str, str]] = []
    queue = Consolidator(run=lambda u, t: ran.append((u, t)) or {})
    queue.schedule("a", "1")
    queue.schedule("a", "1")
    queue.schedule("a", "2")
    queue.schedule("b", "1")
    assert queue.cancel_user("a") == 2
    queue.run_pending()
    assert ran == [("b", "1")] and queue.pending == []
    with pytest.raises(ValueError):
        queue.schedule("a/../b", "1")
