"""The two-thread demo: what crosses threads (long-term facts) and what doesn't (history).

Alice talks in thread ``a``, the deferred job consolidates it, then she opens thread
``b``: the assistant knows her name, city and diet there, updates her city instead of
adding a second one, and answers in the style she asked for in ``a``. Thread ``b``'s
messages never appear in thread ``a``, Bob sees none of Alice's memories, and "forget
me" removes all of Alice's memories and checkpoints while Bob's stay.
"""

from __future__ import annotations

from typing import TextIO

from memory_agent.assistant import Assistant
from memory_agent.long_term import list_facts
from memory_agent.namespaces import checkpoint_thread

__all__ = ["run_demo"]


def run_demo(bot: Assistant, out: TextIO | None = None) -> dict[str, bool]:
    """Run the scripted conversations and return named checks (all ``True`` on success)."""

    def say(user: str, thread: str, text: str) -> str:
        reply = bot.chat(user, thread, text)
        if out is not None:
            tools = f"  [tools: {', '.join(reply.tool_calls)}]" if reply.tool_calls else ""
            print(f"{user}/{thread} > {text}", file=out)
            print(f"{' ' * len(user + thread)}  < {reply.text}{tools}", file=out)
        return reply.text

    def note(text: str) -> None:
        if out is not None:
            print(f"\n-- {text}", file=out)

    note("thread a: Alice introduces herself (hot path: the remember tool)")
    say("alice", "a", "Hi! My name is Ada and I live in Lisbon.")
    say("alice", "a", "I'm vegetarian, and my favorite programming language is Rust.")
    say("alice", "a", "Please keep your answers short.")
    history_a = say("alice", "a", "What did I say earlier in this conversation?")

    note("background job: consolidate thread a (facts, episode, style rules)")
    jobs = bot.run_pending()
    rules = bot.memories("alice")["rules"]
    if out is not None:
        for job in jobs:
            writes = ", ".join(f"{w['action']} {w['key']}" for w in job.get("writes", []))
            print(f"consolidated {job['thread_id']}: {writes}; rules={job.get('rules')}", file=out)

    note("thread b: a new conversation, same user")
    known_b = say("alice", "b", "What do you know about me?")
    say("alice", "b", "Actually, I moved to Porto.")
    city_b = say("alice", "b", "Where do I live?")
    history_b = say("alice", "b", "What did I say earlier in this conversation?")
    long_b = say("alice", "b", "Explain what a checkpointer does.")

    note("back to thread a, and another user")
    history_a2 = say("alice", "a", "What did I say earlier in this conversation?")
    bob = say("bob", "a", "What do you know about me?")
    say("bob", "a", "My name is Bob.")
    bot.run_pending()

    facts = {i.key: i.value for i in list_facts(bot.store, "alice")}
    city = facts.get("home_city", {})
    note("forget me: Alice")
    report = bot.forget_user("alice")
    if out is not None:
        print(f"deleted {report['items']} items and threads {report['threads']}", file=out)
    alice_left = bot.memories("alice")
    alice_checkpoints = [
        bot.checkpointer.get_tuple({"configurable": {"thread_id": checkpoint_thread("alice", t)}})
        for t in report["threads"]
    ]

    checks: dict[str, bool] = {
        "thread a remembers its own history": "Lisbon" in history_a,
        "facts from thread a are recalled in thread b": all(
            w in known_b for w in ("Ada", "vegetarian", "Rust")
        ),
        "a changed fact is updated, not duplicated": "Porto" in city_b
        and "Lisbon" not in city_b
        and city.get("version") == 2
        and sum(1 for v in facts.values() if "Lives in" in str(v.get("content"))) == 1,
        "thread b's history doesn't include thread a": "Lisbon" not in history_b,
        "thread a's history doesn't include thread b": "Porto" not in history_a2
        and "know about me" not in history_a2,
        "style learned in thread a applies in thread b": bool(rules) and long_b.count(".") <= 1,
        "another user sees nothing": "don't know" in bob,
        "forget me wipes the store namespace": not any(
            alice_left[k] for k in ("facts", "episodes", "rules", "threads")
        ),
        "forget me wipes the checkpoints": all(c is None for c in alice_checkpoints),
        "forget me leaves other users alone": bool(bot.memories("bob")["facts"]),
    }
    if out is not None:
        print("\nchecks:", file=out)
        for name, ok in checks.items():
            print(f"  [{'ok' if ok else 'FAIL'}] {name}", file=out)
    return checks
