"""Shared fixtures: an offline assistant (fake model, hashing embedder, no network)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from langgraph.store.memory import InMemoryStore

from memory_agent import Assistant
from memory_agent import FakeAssistantModel
from memory_agent import HashingEmbeddings
from memory_agent import Settings


@pytest.fixture(autouse=True)
def _no_hosted_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("ANTHROPIC_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def settings() -> Settings:
    return Settings.offline()


@pytest.fixture
def model() -> FakeAssistantModel:
    return FakeAssistantModel()


@pytest.fixture
def bot(settings: Settings, model: FakeAssistantModel) -> Iterator[Assistant]:
    with Assistant.open(settings, model=model) as assistant:
        yield assistant


@pytest.fixture
def sqlite_paths(tmp_path: Path) -> tuple[str, str]:
    return str(tmp_path / "threads.db"), str(tmp_path / "memories.db")


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore(index={"dims": 256, "embed": HashingEmbeddings(256)})
