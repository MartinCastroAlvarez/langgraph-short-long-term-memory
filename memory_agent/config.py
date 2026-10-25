"""Settings, read from ``MEMORY_*`` environment variables (every one is optional)."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from typing import Literal
from typing import cast

__all__ = [
    "DEFAULT_EMBED_MODELS",
    "DEFAULT_MODELS",
    "EMBEDDERS",
    "PROVIDERS",
    "WRITE_MODES",
    "Embedder",
    "Provider",
    "Settings",
    "WriteMode",
]

Provider = Literal["ollama", "anthropic", "fake"]
PROVIDERS: tuple[Provider, ...] = ("ollama", "anthropic", "fake")
Embedder = Literal["ollama", "fastembed", "hashing"]
EMBEDDERS: tuple[Embedder, ...] = ("ollama", "fastembed", "hashing")
WriteMode = Literal["hot", "background", "both"]
WRITE_MODES: tuple[WriteMode, ...] = ("hot", "background", "both")

DEFAULT_MODELS: dict[Provider, str] = {
    "ollama": "qwen3:4b",
    "anthropic": "claude-sonnet-5-5",
    "fake": "fake-assistant",
}
DEFAULT_EMBED_MODELS: dict[Embedder, tuple[str, int]] = {
    "ollama": ("nomic-embed-text", 768),
    "fastembed": ("BAAI/bge-small-en-v1.5", 384),
    "hashing": ("hashing-256", 256),
}


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    return default if raw is None or raw == "" else int(raw)


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    return default if raw is None or raw == "" else float(raw)


def _choice(env: Mapping[str, str], key: str, default: str, choices: tuple[str, ...]) -> str:
    value = (env.get(key) or default).strip().lower()
    if value not in choices:
        raise ValueError(f"{key} must be one of {choices}, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the chat graph, the consolidation job and the CLI can be tuned with."""

    provider: Provider = "ollama"
    model: str = DEFAULT_MODELS["ollama"]
    ollama_host: str = "http://127.0.0.1:11434"
    allow_remote_ollama: bool = False
    temperature: float = 0.0

    embeddings: Embedder = "ollama"
    embed_model: str = DEFAULT_EMBED_MODELS["ollama"][0]
    embed_dims: int = DEFAULT_EMBED_MODELS["ollama"][1]

    write_mode: WriteMode = "both"
    """``hot``: only the in-graph ``remember``/``forget`` tools write long-term memory.
    ``background``: only the deferred consolidation job does. ``both``: both."""

    # -- short-term memory (the thread) ------------------------------------ #
    max_context_tokens: int = 3000
    """Hard budget for one model call: system prompt + the trimmed messages."""
    summarize_after_tokens: int = 2000
    """When the thread's messages exceed this, older turns are folded into a summary."""
    keep_recent_tokens: int = 600
    """How much recent conversation survives a summarization verbatim."""
    summary_max_chars: int = 1500

    # -- long-term memory (the store) --------------------------------------- #
    recall_facts: int = 5
    recall_episodes: int = 2
    dedup_threshold: float = 0.9
    """A new fact without a slot updates an existing fact at least this similar."""

    def __post_init__(self) -> None:
        if not 200 <= self.max_context_tokens <= 200_000:
            raise ValueError("max_context_tokens must be between 200 and 200000")
        if not 0 < self.keep_recent_tokens < self.summarize_after_tokens:
            raise ValueError("keep_recent_tokens must be > 0 and < summarize_after_tokens")
        if self.summarize_after_tokens > self.max_context_tokens:
            raise ValueError("summarize_after_tokens must not exceed max_context_tokens")
        if not 100 <= self.summary_max_chars <= 20_000:
            raise ValueError("summary_max_chars must be between 100 and 20000")
        if not 0 <= self.recall_facts <= 50 or not 0 <= self.recall_episodes <= 20:
            raise ValueError("recall_facts must be 0-50 and recall_episodes 0-20")
        if not 0 < self.dedup_threshold <= 1:
            raise ValueError("dedup_threshold must be in (0, 1]")
        if not 8 <= self.embed_dims <= 8192:
            raise ValueError("embed_dims must be between 8 and 8192")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        provider = cast(Provider, _choice(e, "MEMORY_PROVIDER", "ollama", PROVIDERS))
        default_embedder = "hashing" if provider == "fake" else "ollama"
        embedder = cast(Embedder, _choice(e, "MEMORY_EMBEDDINGS", default_embedder, EMBEDDERS))
        write_mode = cast(WriteMode, _choice(e, "MEMORY_WRITE_MODE", "both", WRITE_MODES))
        embed_model, embed_dims = DEFAULT_EMBED_MODELS[embedder]
        d = cls()
        return cls(
            provider=provider,
            model=e.get("MEMORY_MODEL") or DEFAULT_MODELS[provider],
            ollama_host=e.get("OLLAMA_HOST") or d.ollama_host,
            allow_remote_ollama=e.get("MEMORY_ALLOW_REMOTE_OLLAMA", "") == "1",
            temperature=_float(e, "MEMORY_TEMPERATURE", d.temperature),
            embeddings=embedder,
            embed_model=e.get("MEMORY_EMBED_MODEL") or embed_model,
            embed_dims=_int(e, "MEMORY_EMBED_DIMS", embed_dims),
            write_mode=write_mode,
            max_context_tokens=_int(e, "MEMORY_MAX_CONTEXT_TOKENS", d.max_context_tokens),
            summarize_after_tokens=_int(
                e, "MEMORY_SUMMARIZE_AFTER_TOKENS", d.summarize_after_tokens
            ),
            keep_recent_tokens=_int(e, "MEMORY_KEEP_RECENT_TOKENS", d.keep_recent_tokens),
            summary_max_chars=_int(e, "MEMORY_SUMMARY_MAX_CHARS", d.summary_max_chars),
            recall_facts=_int(e, "MEMORY_RECALL_FACTS", d.recall_facts),
            recall_episodes=_int(e, "MEMORY_RECALL_EPISODES", d.recall_episodes),
            dedup_threshold=_float(e, "MEMORY_DEDUP_THRESHOLD", d.dedup_threshold),
        )

    @classmethod
    def offline(cls, **overrides: Any) -> Settings:
        """The fake model and the hashing embedder: no model server, no network."""
        model, dims = DEFAULT_EMBED_MODELS["hashing"]
        base = cls(
            provider="fake",
            model=DEFAULT_MODELS["fake"],
            embeddings="hashing",
            embed_model=model,
            embed_dims=dims,
        )
        return dataclasses.replace(base, **overrides)
