"""CLI, settings and factories."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore

from memory_agent import FakeAssistantModel
from memory_agent import HashingEmbeddings
from memory_agent import Settings
from memory_agent import make_embeddings
from memory_agent import make_model
from memory_agent.cli import main
from memory_agent.llm import ModelConfigError
from memory_agent.llm import is_loopback_url
from memory_agent.stores import StoreConfigError
from memory_agent.stores import open_store


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), out=out)
    return code, out.getvalue()


def test_mermaid_prints_both_graphs() -> None:
    code, chat = run("--mermaid", "chat")
    assert code == 0 and "summarize(summarize)" in chat and "tools -->" in chat.replace("\t", "")
    code, job = run("--mermaid", "consolidate")
    assert code == 0 and "write_episode" in job


def test_demo_passes_offline() -> None:
    code, out = run("--provider", "fake", "--demo")
    assert code == 0 and "[FAIL]" not in out and out.count("[ok]") == 10
    code, out = run("--provider", "fake", "--demo", "--json")
    assert code == 0 and json.loads(out)["passed"] is True


def test_turns_memories_history_fork_and_forget_me(tmp_path: Path) -> None:
    d = ["--provider", "fake", "--data-dir", str(tmp_path), "--user", "alice"]
    code, out = run(*d, "--thread", "a", "Hi, my name is Ada.")
    assert code == 0 and "[tools: remember]" in out
    code, out = run(*d, "--thread", "b", "--json", "What is my name?")
    assert code == 0 and "Ada" in json.loads(out)["text"]
    code, out = run(*d, "--memories")
    assert json.loads(out)["facts"]["name"]["content"] == "The user's name is Ada."
    code, out = run(*d, "--thread", "a", "--history", "--json")
    rows = json.loads(out)
    start = [r for r in rows if r["next"] == ["__start__"]][-1]
    code, out = run(*d, "--thread", "a", "--history")
    assert start["checkpoint_id"] in out
    code, out = run(*d, "--thread", "a", "--fork", start["checkpoint_id"], "Call me Grace.")
    assert code == 0 and "Grace" in out
    agent = [r for r in json.loads(run(*d, "--thread", "a", "--history", "--json")[1])][1]
    code, out = run(*d, "--thread", "a", "--replay", agent["checkpoint_id"])
    assert code == 0
    code, out = run(*d, "--forget-me")
    assert json.loads(out)["threads"] == ["a", "b"]
    assert json.loads(run(*d, "--memories")[1])["facts"] == {}


def test_summaries_are_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_MAX_CONTEXT_TOKENS", "400")
    monkeypatch.setenv("MEMORY_SUMMARIZE_AFTER_TOKENS", "200")
    monkeypatch.setenv("MEMORY_KEEP_RECENT_TOKENS", "60")
    d = ["--provider", "fake", "--data-dir", str(tmp_path)]
    outs = [
        run(*d, f"Turn {i} with enough words to use up the small budget quickly.")[1]
        for i in range(8)
    ]
    assert any("[summarized" in o for o in outs)


def test_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = ["--provider", "fake", "--data-dir", str(tmp_path)]
    assert run(*d)[0] == 2
    assert run(*d, "--user", "../bob", "hi")[0] == 2
    assert "invalid user id" in capsys.readouterr().err
    assert run(*d, "x" * 4001)[0] == 2
    assert run(*d, "--thread", "a", "--fork", "nope", "hi")[0] == 2
    assert run(*d, "--store", "ftp://x", "hi")[0] == 2


def test_settings_from_env_and_validation() -> None:
    s = Settings.from_env({})
    assert (s.provider, s.model, s.embeddings, s.embed_dims) == (
        "ollama",
        "qwen3:4b",
        "ollama",
        768,
    )
    f = Settings.from_env({"MEMORY_PROVIDER": "fake", "MEMORY_RECALL_FACTS": "3"})
    assert (f.embeddings, f.embed_dims, f.recall_facts) == ("hashing", 256, 3)
    a = Settings.from_env({"MEMORY_PROVIDER": "anthropic", "MEMORY_EMBEDDINGS": "fastembed"})
    assert (a.model, a.embed_model) == ("claude-sonnet-5-5", "BAAI/bge-small-en-v1.5")
    for env in (
        {"MEMORY_PROVIDER": "openai"},
        {"MEMORY_WRITE_MODE": "sometimes"},
        {"MEMORY_KEEP_RECENT_TOKENS": "5000"},
        {"MEMORY_MAX_CONTEXT_TOKENS": "100"},
        {"MEMORY_SUMMARIZE_AFTER_TOKENS": "9000"},
        {"MEMORY_SUMMARY_MAX_CHARS": "10"},
        {"MEMORY_RECALL_FACTS": "99"},
        {"MEMORY_DEDUP_THRESHOLD": "0"},
        {"MEMORY_EMBED_DIMS": "2"},
    ):
        with pytest.raises(ValueError):
            Settings.from_env(env)


def test_model_and_embedding_factories() -> None:
    assert isinstance(make_model(Settings.offline()), FakeAssistantModel)
    assert isinstance(make_model(Settings()), ChatOllama)
    assert isinstance(make_embeddings(Settings.offline()), HashingEmbeddings)
    assert isinstance(make_embeddings(Settings()), OllamaEmbeddings)
    remote = Settings(ollama_host="http://10.0.0.5:11434")
    with pytest.raises(ModelConfigError):
        make_model(remote)
    with pytest.raises(ModelConfigError):
        make_embeddings(remote)
    allowed = Settings(ollama_host="http://10.0.0.5:11434", allow_remote_ollama=True)
    assert isinstance(make_model(allowed), ChatOllama)
    assert is_loopback_url("localhost:11434") and is_loopback_url("http://[::1]:1")
    assert not is_loopback_url("http://example.com")


def test_store_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    emb = HashingEmbeddings(16)
    with open_store("memory://", emb, 16) as store:
        assert isinstance(store, InMemoryStore)
    with open_store(f"sqlite:///{tmp_path / 'm.db'}", emb, 16) as store:
        assert isinstance(store, SqliteStore)
    with pytest.raises(StoreConfigError), open_store("redis://x", emb, 16):
        pass
    monkeypatch.setitem(sys.modules, "s3_agent_memory", None)
    with (
        pytest.raises(StoreConfigError, match="s3-agent-memory"),
        open_store("s3+memory://", emb, 16),
    ):
        pass
