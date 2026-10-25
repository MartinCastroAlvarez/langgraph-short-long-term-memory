"""Where each kind of memory is persisted.

* Short-term: ``SqliteSaver`` (``langgraph-checkpoint-sqlite``), one row per checkpoint
  of each thread. ``PostgresSaver`` is the drop-in for several processes.
* Long-term: any ``BaseStore`` with a semantic index (``index={"dims", "embed"}``):

  ========================== ==================================================
  ``memory://``              ``InMemoryStore`` (tests, one process)
  ``sqlite:///path.db``      ``SqliteStore`` + sqlite-vec (the default)
  ``s3+memory://``           ``S3Store`` from s3-agent-memory, objects in memory
  ``s3+file:///dir``         ``S3Store``, one JSON object per memory in a folder
  ``s3://bucket/prefix``     ``S3Store`` on S3, R2, SeaweedFS or MinIO
  ========================== ==================================================

  The ``s3`` schemes are an optional adapter (``pip install "s3-agent-memory[langgraph]
  @ git+https://github.com/MartinCastroAlvarez/s3-agent-memory"``). That store ranks
  with BM25 instead of this package's embedder, so its scores live on another scale;
  slot-keyed facts still update in place.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.base import IndexConfig
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore

__all__ = ["StoreConfigError", "open_checkpointer", "open_store"]


class StoreConfigError(RuntimeError):
    """The store URL can't be opened as given."""


@contextlib.contextmanager
def open_checkpointer(path: str) -> Iterator[SqliteSaver]:
    """A ``SqliteSaver`` on ``path`` (``":memory:"`` for a throwaway one)."""
    with SqliteSaver.from_conn_string(path) as saver:
        saver.setup()
        yield saver


def _s3_store(url: str) -> BaseStore:  # pragma: no cover - optional sibling package
    try:
        from s3_agent_memory import MemoryStore
        from s3_agent_memory.langgraph_store import S3Store
    except ImportError as exc:
        raise StoreConfigError(
            "s3 stores need the optional adapter: pip install "
            '"s3-agent-memory[langgraph] @ git+https://github.com/MartinCastroAlvarez/s3-agent-memory"'
        ) from exc
    store: BaseStore = S3Store(MemoryStore.from_url(url.removeprefix("s3+")), index=["content"])
    return store


@contextlib.contextmanager
def open_store(url: str, embeddings: Embeddings, dims: int) -> Iterator[BaseStore]:
    index: IndexConfig = {"dims": dims, "embed": embeddings}
    if url in (":memory:", "memory://"):
        yield InMemoryStore(index=index)
        return
    if url.startswith(("s3://", "s3+")):
        yield _s3_store(url)
        return
    path = url.removeprefix("sqlite:///") if url.startswith("sqlite:///") else url
    if "://" in path:
        raise StoreConfigError(f"unsupported store URL {url!r}")
    sqlite_index: Any = index
    with SqliteStore.from_conn_string(path, index=sqlite_index) as store:
        store.setup()
        yield store
