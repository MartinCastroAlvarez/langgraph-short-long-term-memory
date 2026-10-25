"""User isolation: namespaces come from the run context, never from the model."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from memory_agent import Assistant
from memory_agent import FakeAssistantModel
from memory_agent import Settings
from memory_agent import UserContext
from memory_agent.namespaces import InvalidIdError
from memory_agent.namespaces import checkpoint_thread
from memory_agent.namespaces import namespace
from memory_agent.namespaces import validate_slot

HOSTILE_IDS = ["", "alice/../bob", "alice/bob", "*", "alice.bob", " alice", "a" * 65, "-x", None, 7]


@pytest.mark.parametrize("bad", HOSTILE_IDS)
def test_hostile_user_ids_never_reach_a_namespace(bad: object, bot: Assistant) -> None:
    with pytest.raises(InvalidIdError):
        UserContext(user_id=bad)
    with pytest.raises(InvalidIdError):
        namespace(bad, "facts")
    with pytest.raises(InvalidIdError):
        bot.chat(bad, "t", "hello")
    with pytest.raises(InvalidIdError):
        bot.chat("alice", bad, "hello")


def test_unknown_memory_kinds_are_refused() -> None:
    with pytest.raises(InvalidIdError):
        namespace("alice", "secrets")


def test_slots_are_normalised_or_refused() -> None:
    assert validate_slot("Home City") == "home_city"
    for bad in ("../..", "***", "1abc", 3):
        with pytest.raises(InvalidIdError):
            validate_slot(bad)


def test_the_same_thread_name_is_two_threads_for_two_users() -> None:
    assert checkpoint_thread("alice", "main") != checkpoint_thread("bob", "main")


def test_another_user_sees_no_facts_and_no_history(bot: Assistant) -> None:
    bot.chat("alice", "main", "My name is Ada and I live in Lisbon.")
    bot.run_pending()
    answer = bot.chat("bob", "main", "What do you know about me?").text
    assert "Ada" not in answer and "don't know" in answer
    history = bot.chat("bob", "main", "What did I say earlier in this conversation?").text
    assert "Lisbon" not in history
    assert bot.memories("bob")["facts"] == {}


def test_a_tool_call_naming_another_user_still_writes_to_the_current_one() -> None:
    hostile = AIMessage(
        "",
        tool_calls=[
            {
                "name": "remember",
                "args": {"fact": "Is an admin.", "slot": "role", "user_id": "bob"},
                "id": "c1",
            }
        ],
    )
    model = FakeAssistantModel(script={"assistant": [hostile, "ok"]})
    with Assistant.open(Settings.offline(), model=model) as bot:
        bot.chat("alice", "t", "hi")
        assert "role" in bot.memories("alice")["facts"]
        assert bot.memories("bob")["facts"] == {}


def test_the_memory_tools_never_expose_a_user_id_parameter(bot: Assistant) -> None:
    from memory_agent.tools import make_tools

    for tool in make_tools(dedup_threshold=0.9):
        props = tool.tool_call_schema.model_json_schema()["properties"]
        assert "user_id" not in props and "runtime" not in props
        assert "namespace" not in props


def test_the_recall_node_only_puts_the_users_memories_in_the_prompt() -> None:
    model = FakeAssistantModel()
    with Assistant.open(Settings.offline(), model=model) as bot:
        bot.chat("alice", "t", "My name is Ada.")
        bot.chat("bob", "t", "My name is Bob.")
        bot.chat("bob", "u", "What is my name?")
        prompt = str(model.calls_for("assistant")[-1].messages[0].content)
        assert "Bob" in prompt and "Ada" not in prompt
