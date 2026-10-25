"""A deterministic, role-aware fake chat model, so the graphs run with no model at all.

``FakeAssistantModel`` is a real LangChain ``BaseChatModel``: the graphs call it exactly
as they call ``ChatOllama``, ``bind_tools`` works, and ``with_structured_output`` works
through tool calls. It reads the ``Role: <name>.`` line of the system prompt and answers
like that node would, with a few regular expressions instead of a language model:

* **assistant** - calls ``remember`` when the user states a fact about themselves
  ("my name is Ada", "I live in Lisbon") and the tool is bound, ``forget`` when asked
  to forget; answers "what do you know about me" from the ``facts`` block it was
  given, "what did I say earlier" from the messages it was given, and follows the
  ``style`` block (``concise`` keeps one sentence). So what it answers proves what was
  in its context window, which is exactly what the isolation tests need.
* **summarizer** - "Earlier, the user said: ..." built from the messages it was given.
* **memory-writer** - a ``MemoryUpdate`` with the same fact patterns and any feedback.
* **episode-writer** - an ``Episode`` from the first and last turns.
* **optimizer** - the current rules plus the feedback.

``script={"role": [reply, ...]}`` overrides a role with canned replies (strings,
``AIMessage``s, dicts of tool arguments, or exceptions to raise). Every call is recorded
in ``calls`` with its role and messages.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field
from pydantic import PrivateAttr

from memory_agent.prompts import parse_blocks
from memory_agent.prompts import role_of
from memory_agent.prompts import text_of

__all__ = ["FakeAssistantModel", "FakeCall", "extract_facts", "extract_feedback", "extract_forget"]

_END = r"(?=[.,;!?]|\s+and\s|$)"
# A case-insensitive cue, then a capture; proper names must be capitalised.
_FACT_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("name", r"(?i:\bmy name is) ([A-Z][\w-]+)", "The user's name is {}."),
    ("name", r"(?i:\bcall me) ([A-Z][\w-]+)", "The user's name is {}."),
    (
        "home_city",
        r"(?i:\bi (?:live|moved|relocated) (?:in|to)) ([A-Z][\w-]*(?: [A-Z][\w-]*)*)",
        "Lives in {}.",
    ),
    ("diet", r"(?i:\bi(?:'m| am) (?:a |now )?(vegetarian|vegan|pescatarian))", "Is {}."),
    ("diet", r"(?i:\bi eat (meat|fish|everything))", "Eats {}."),
    ("occupation", r"(?i:\bi work as (?:an? )?([\w -]+?))" + _END, "Works as {}."),
    (
        "favorite_language",
        r"(?i:\bmy favou?rite (?:programming )?language is ([\w+#]+))",
        "Favourite programming language is {}.",
    ),
    (
        "pet",
        r"(?i:\bmy (?:dog|cat)(?:'s name)? is (?:called |named )?)([A-Z]\w+)",
        "Has a pet named {}.",
    ),
    ("timezone", r"(?i:\bmy time ?zone is ([\w/+-]+))", "Time zone is {}."),
)
_FEEDBACK = re.compile(
    r"\b(be (?:more )?(?:concise|brief)|keep (?:your )?answers (?:short|brief)|"
    r"use bullet points|answer in [A-Z]\w+|stop using emojis)",
    re.I,
)
_FORGET = re.compile(r"\bforget (?:that |about |my )?(.+?)(?:[.!?]|$)", re.I)
_ABOUT_ME = re.compile(
    r"\b(what(?:'s| is| do you know)|where do i|who am i|do you remember|what do i)\b", re.I
)
_EARLIER = re.compile(r"\b(what did i (?:say|tell you)|earlier in this|so far)\b", re.I)


def extract_facts(text: str) -> list[tuple[str, str]]:
    """``(slot, sentence)`` for every fact pattern found in ``text``."""
    found: list[tuple[str, str]] = []
    for slot, pattern, template in _FACT_PATTERNS:
        m = re.search(pattern, text)
        if m and all(s != slot for s, _ in found):
            found.append((slot, template.format(m.group(1).strip())))
    return found


def extract_feedback(text: str) -> str | None:
    m = _FEEDBACK.search(text)
    if m is None:
        return None
    rule = m.group(1)
    return rule[0].upper() + rule[1:].rstrip(".") + "."


def extract_forget(text: str) -> str | None:
    m = _FORGET.search(text)
    return m.group(1).strip() if m else None


def _lines(block: str) -> list[str]:
    return [line[2:].strip() for line in block.splitlines() if line.startswith("- ")]


def _first_sentence(text: str) -> str:
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    return m.group(1) if m else text


@dataclass
class FakeCall:
    role: str | None
    messages: list[BaseMessage]
    tools: list[str]


class FakeAssistantModel(BaseChatModel):
    """Scripted stand-in for ``ChatOllama`` (see the module docstring)."""

    script: dict[str, list[Any]] = Field(default_factory=dict)
    _calls: list[FakeCall] = PrivateAttr(default_factory=list)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _ids: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "fake-assistant"

    @property
    def calls(self) -> list[FakeCall]:
        return self._calls

    def calls_for(self, role: str) -> list[FakeCall]:
        return [c for c in self._calls if c.role == role]

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, AIMessage]:
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, **kwargs)

    # ------------------------------------------------------------------ #
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = next((text_of(m) for m in messages if isinstance(m, SystemMessage)), "")
        role = role_of(system)
        tools = [t["function"]["name"] for t in kwargs.get("tools", [])]
        with self._lock:
            self._calls.append(FakeCall(role, list(messages), tools))
            scripted = self.script.get(role or "", [])
            canned = scripted.pop(0) if scripted else None
        if isinstance(canned, BaseException):
            raise canned
        if canned is not None:
            message = self._canned(canned, tools)
        elif role == "assistant":
            message = self._assistant(system, messages, tools)
        elif role == "summarizer":
            message = AIMessage(content=self._summary(system))
        elif role == "memory-writer":
            message = self._structured(tools, self._memory_update(system))
        elif role == "episode-writer":
            message = self._structured(tools, self._episode(system))
        elif role == "optimizer":
            message = self._structured(tools, self._instructions(system))
        else:
            message = AIMessage(content="OK.")
        return ChatResult(generations=[ChatGeneration(message=message)])

    # -- helpers ----------------------------------------------------------- #
    def _next_id(self) -> str:
        with self._lock:
            self._ids += 1
            return f"call_{self._ids}"

    def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"name": name, "args": args, "id": self._next_id(), "type": "tool_call"}

    def _structured(self, tools: list[str], args: dict[str, Any]) -> AIMessage:
        if not tools:  # asked without with_structured_output: answer in prose
            return AIMessage(content=str(args))
        return AIMessage(content="", tool_calls=[self._call(tools[0], args)])

    def _canned(self, canned: Any, tools: list[str]) -> AIMessage:
        if isinstance(canned, AIMessage):
            return canned
        if isinstance(canned, dict):
            return self._structured(tools, canned)
        return AIMessage(content=str(canned))

    # -- roles ------------------------------------------------------------- #
    def _assistant(self, system: str, messages: list[BaseMessage], tools: list[str]) -> AIMessage:
        blocks = parse_blocks(system)
        facts = _lines(blocks.get("facts", ""))
        rules = " ".join(_lines(blocks.get("style", ""))).lower()
        convo = [m for m in messages if not isinstance(m, SystemMessage)]
        last = convo[-1] if convo else HumanMessage("")
        if isinstance(last, ToolMessage):
            results: list[str] = []
            for m in reversed(convo):
                if not isinstance(m, ToolMessage):
                    break
                results.insert(0, text_of(m))
            done = "; ".join(r.rstrip(".") for r in results)
            return AIMessage(content=self._style(f"Noted ({done}).", rules))
        text = text_of(last)
        forget = extract_forget(text)
        if forget and "forget" in tools:
            return AIMessage(content="", tool_calls=[self._call("forget", {"query": forget})])
        stated = extract_facts(text)
        if stated and "remember" in tools:
            calls = [self._call("remember", {"fact": s, "slot": slot}) for slot, s in stated]
            return AIMessage(content="", tool_calls=calls)
        if _EARLIER.search(text):
            earlier = [text_of(m).rstrip(".!? ") for m in convo[:-1] if isinstance(m, HumanMessage)]
            said = "; ".join(earlier) if earlier else "nothing yet"
            reply = f"Earlier in this conversation you said: {said}."
            if blocks.get("summary"):
                reply += f" Before that: {blocks['summary']}"
            return AIMessage(content=reply)
        if _ABOUT_ME.search(text) and re.search(r"\b(my|me|i|about me)\b", text, re.I):
            if not facts:
                return AIMessage(content="I don't know that yet. Tell me and I'll remember.")
            if re.search(r"\b(know about me|remember about me)\b", text, re.I):
                listed = "; ".join(f.rstrip(".") for f in facts)
                return AIMessage(content=self._style(f"What I remember: {listed}.", rules))
            return AIMessage(content=self._style(f"{facts[0]} That's what you told me.", rules))
        if extract_feedback(text):
            return AIMessage(content="Understood. I'll answer that way from now on.")
        if stated:
            reply = f"Nice to know: {stated[0][1]} I'll keep it in mind."
        else:
            reply = (
                f"Happy to help with that: {text[:160].rstrip('.?!')}. "
                "Here is a longer explanation with examples and caveats."
            )
        episodes = _lines(blocks.get("episodes", ""))
        if episodes:
            reply += f" Last time: {episodes[0]}"
        return AIMessage(content=self._style(reply, rules))

    @staticmethod
    def _style(reply: str, rules: str) -> str:
        if any(w in rules for w in ("concise", "brief", "short")):
            reply = _first_sentence(reply)
        if "bullet" in rules:
            reply = "- " + reply
        return reply

    @staticmethod
    def _summary(system: str) -> str:
        blocks = parse_blocks(system)
        previous = blocks.get("previous_summary", "(none)")
        said = [
            line.removeprefix("user: ")
            for line in blocks.get("messages", "").splitlines()
            if line.startswith("user: ")
        ]
        head = "" if previous == "(none)" else previous + " "
        return head + "Earlier, the user said: " + "; ".join(said) + "."

    @staticmethod
    def _user_lines(system: str) -> list[str]:
        return [
            line.removeprefix("user: ")
            for line in parse_blocks(system).get("transcript", "").splitlines()
            if line.startswith("user: ")
        ]

    def _memory_update(self, system: str) -> dict[str, Any]:
        facts: list[dict[str, Any]] = []
        feedback: list[str] = []
        for line in self._user_lines(system):
            forget = extract_forget(line)
            if forget:
                facts.append({"slot": None, "content": forget, "action": "delete"})
                continue
            facts += [{"slot": slot, "content": s} for slot, s in extract_facts(line)]
            rule = extract_feedback(line)
            if rule:
                feedback.append(rule)
        return {"facts": facts, "feedback": " ".join(feedback) or None}

    def _episode(self, system: str) -> dict[str, Any]:
        lines = parse_blocks(system).get("transcript", "").splitlines()
        users = [x.removeprefix("user: ") for x in lines if x.startswith("user: ")]
        bots = [x.removeprefix("assistant: ") for x in lines if x.startswith("assistant: ")]
        return {
            "situation": f"The user asked: {users[0] if users else 'nothing'}",
            "approach": f"Answered {len(users)} message(s) directly.",
            "outcome": f"Ended with: {bots[-1] if bots else 'no answer'}",
        }

    @staticmethod
    def _instructions(system: str) -> dict[str, Any]:
        blocks = parse_blocks(system)
        rules = _lines(blocks.get("current_rules", ""))
        new = re.findall(r"[^.!?]+[.!?]", blocks.get("feedback", ""))
        return {"rules": [*rules, *(r.strip() for r in new)]}
