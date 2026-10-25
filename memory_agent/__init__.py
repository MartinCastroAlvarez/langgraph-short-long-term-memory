"""Short-term vs long-term agent memory in LangGraph.

Short-term memory is the thread: messages in a checkpointer, trimmed and summarized to
a token budget, isolated per thread, with time travel. Long-term memory is the store:
facts, episodes and style rules in a per-user namespace, shared across that user's
threads and recalled by semantic search.
"""

from memory_agent.assistant import Assistant
from memory_agent.assistant import CheckpointInfo
from memory_agent.assistant import Reply
from memory_agent.config import Settings
from memory_agent.consolidation import Consolidator
from memory_agent.consolidation import build_consolidation_graph
from memory_agent.embeddings import HashingEmbeddings
from memory_agent.embeddings import make_embeddings
from memory_agent.fake import FakeAssistantModel
from memory_agent.graph import build_chat_graph
from memory_agent.llm import make_model
from memory_agent.namespaces import UserContext

__all__ = [
    "Assistant",
    "CheckpointInfo",
    "Consolidator",
    "FakeAssistantModel",
    "HashingEmbeddings",
    "Reply",
    "Settings",
    "UserContext",
    "__version__",
    "build_chat_graph",
    "build_consolidation_graph",
    "make_embeddings",
    "make_model",
]

__version__ = "0.1.0"
