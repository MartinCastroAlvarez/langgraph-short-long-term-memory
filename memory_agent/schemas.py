"""Structured outputs of the memory-writing roles (``with_structured_output`` schemas).

Everything here is model output, so every field is bounded and validated: lengths are
capped, slots are normalised, and anything malformed is dropped rather than stored.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from memory_agent.namespaces import InvalidIdError
from memory_agent.namespaces import validate_slot

__all__ = ["Episode", "FactCandidate", "Instructions", "MemoryUpdate"]

MAX_FACT_CHARS = 300
MAX_RULES = 10
MAX_RULE_CHARS = 200


def _clip(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


class FactCandidate(BaseModel):
    """One durable fact about the user, or a request to forget one."""

    slot: str | None = Field(
        default=None,
        description="A stable snake_case attribute name such as name, home_city, diet, "
        "occupation, favorite_language, pet. A new value for the same slot replaces the old one.",
    )
    content: str = Field(description="The fact as one short third-person sentence.")
    action: Literal["upsert", "delete"] = "upsert"

    @field_validator("slot", mode="before")
    @classmethod
    def _slot(cls, value: object) -> str | None:
        if value is None or value == "":
            return None
        try:
            return validate_slot(value)
        except InvalidIdError:
            return None

    @field_validator("content")
    @classmethod
    def _content(cls, value: str) -> str:
        return _clip(value, MAX_FACT_CHARS)


class MemoryUpdate(BaseModel):
    """What the memory writer learned from a conversation."""

    facts: list[FactCandidate] = Field(
        default_factory=list,
        description="Durable facts about the user (profile, preferences). Not small talk, "
        "not facts about the world, not anything the assistant said.",
    )
    feedback: str | None = Field(
        default=None,
        description="Feedback about HOW the assistant should answer (tone, length, format), "
        "or null when there is none.",
    )

    @field_validator("facts")
    @classmethod
    def _cap(cls, value: list[FactCandidate]) -> list[FactCandidate]:
        return [f for f in value if f.content][:20]

    @field_validator("feedback")
    @classmethod
    def _feedback(cls, value: str | None) -> str | None:
        return _clip(value, MAX_RULE_CHARS) or None if value else None


class Episode(BaseModel):
    """One past interaction, kept to be recalled as a worked example."""

    situation: str = Field(description="What the user wanted, in one sentence.")
    approach: str = Field(description="What the assistant did, in one sentence.")
    outcome: str = Field(description="How it ended and whether the user was satisfied.")

    @field_validator("situation", "approach", "outcome")
    @classmethod
    def _short(cls, value: str) -> str:
        return _clip(value, 240)


class Instructions(BaseModel):
    """The assistant's learned style rules for one user (procedural memory)."""

    rules: list[str] = Field(
        default_factory=list,
        description="Short imperative rules about tone, length and format only.",
    )

    @field_validator("rules")
    @classmethod
    def _rules(cls, value: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for rule in value:
            text = _clip(rule, MAX_RULE_CHARS)
            key = text.lower().rstrip(".")
            if text and key not in seen:
                seen.add(key)
                out.append(text)
        return out[-MAX_RULES:]
