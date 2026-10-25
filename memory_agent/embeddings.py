"""Local embedders for the store's semantic index.

* ``ollama`` - ``OllamaEmbeddings`` (``nomic-embed-text``, 768 dims) on your ollama.
* ``fastembed`` - ONNX on the CPU, no server (``pip install ".[fastembed]"``).
* ``hashing`` - a deterministic bag-of-words embedder: lowercase word stems hashed into
  256 buckets, L2-normalised. No model, no download, same vector on every machine, and
  cosine similarity still means "shares words". The tests and the offline demo use it.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from langchain_core.embeddings import Embeddings

from memory_agent.config import Settings

__all__ = ["FastEmbedEmbeddings", "HashingEmbeddings", "make_embeddings"]

_STOP = frozenset(
    re.findall(
        r"\S+",
        "a am an and are as at be by do does for from has have i i'm in is it its me my of on "
        "or our so that the their them they this to was we what when where which who why will "
        "with you your user user's",
    )
)


def _stem(word: str) -> str:
    """A tiny suffix stripper: ``lives``/``live``/``living`` all become ``live``/``liv``."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


class HashingEmbeddings(Embeddings):
    """Deterministic feature-hashing embeddings (the offline default)."""

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for word in re.findall(r"[a-z0-9']+", text.lower()):
            if word in _STOP:
                continue
            digest = hashlib.sha256(_stem(word).encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedEmbeddings(Embeddings):  # pragma: no cover - needs the optional extra
    """``fastembed.TextEmbedding`` behind the LangChain interface."""

    def __init__(self, model: str) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                'pip install "langgraph-short-long-term-memory[fastembed]" for fastembed'
            ) from exc
        self._model: Any = TextEmbedding(model_name=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def make_embeddings(settings: Settings) -> Embeddings:
    if settings.embeddings == "hashing":
        return HashingEmbeddings(settings.embed_dims)
    if settings.embeddings == "fastembed":  # pragma: no cover - needs the optional extra
        return FastEmbedEmbeddings(settings.embed_model)
    from langchain_ollama import OllamaEmbeddings

    from memory_agent.llm import check_ollama_host

    check_ollama_host(settings)
    return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_host)
