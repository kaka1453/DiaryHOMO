"""Opt-in local RAG; not a dependency of training or legacy manual cards."""

from .provider import SqliteMemoryProvider
from .schema import (
    Alias,
    EventRecord,
    MemoryErrorV1,
    MemoryQuery,
    RetrievalMemoryProvider,
    RetrievalResult,
    SourceSpan,
    TrustedScope,
)
from .store import MemoryStore

__all__ = [
    "Alias",
    "EventRecord",
    "MemoryErrorV1",
    "MemoryQuery",
    "MemoryStore",
    "RetrievalMemoryProvider",
    "RetrievalResult",
    "SourceSpan",
    "SqliteMemoryProvider",
    "TrustedScope",
]
