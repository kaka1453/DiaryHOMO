"""Versioned, server-side contracts. Never deserialize request JSON as a scope."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from lm_trainer.chat.memory import CARD_KINDS, ContextCard
from lm_trainer.chat.types import CardAudit, valid_id

EVENT_SCHEMA = "lm_trainer.memory-event.v1"
QUERY_SCHEMA = "lm_trainer.memory-query.v1"
BUNDLE_SCHEMA = "lm_trainer.memory-import.v1"
UNKNOWN = "unknown"


class MemoryErrorV1(ValueError):
    """Fixed-code error; never includes evidence, query text or file paths."""


def require(ok, code="MEMORY_SCHEMA_INVALID"):
    if not ok:
        raise MemoryErrorV1(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_ok(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def text_ok(value, limit):
    return (
        isinstance(value, str)
        and 0 < len(value) <= limit
        and bool(value.strip())
        and not re.search(r"[\ud800-\udfff]", value)
    )


def clock(value):
    """Parse an aware ISO timestamp; use canonical UTC text for SQLite comparisons."""
    try:
        parsed = datetime.fromisoformat(value)
        require(parsed.tzinfo is not None and "T" in value)
        return parsed.astimezone(UTC).isoformat(timespec="microseconds")
    except (ValueError, TypeError, OverflowError):
        raise MemoryErrorV1("MEMORY_TIME_INVALID") from None


@dataclass(frozen=True, slots=True)
class TrustedScope:
    scope_id: str
    namespace: str
    purpose: str
    kind: Literal["group", "workbench"]
    channel: str | None = None
    client_id: str | None = None
    room_id: str | None = None
    thread_id: str | None = None
    web_account: str | None = None
    session_id: str | None = None

    def __post_init__(self):
        group = (self.channel, self.client_id, self.room_id, self.thread_id)
        workbench = (self.web_account, self.session_id)
        require(all(map(valid_id, (self.scope_id, self.namespace, self.purpose))))
        require(
            (
                self.kind == "group"
                and all(map(valid_id, group))
                and not any(x is not None for x in workbench)
            )
            or (
                self.kind == "workbench"
                and all(map(valid_id, workbench))
                and not any(x is not None for x in group)
            )
        )


@dataclass(frozen=True, slots=True)
class SourceSpan:
    source_id: str
    source_ref: str = field(repr=False)
    char_start: int
    char_end: int
    content_hash: str
    origin: str

    def __post_init__(self):
        require(
            valid_id(self.source_id)
            and text_ok(self.source_ref, 512)
            and type(self.char_start) is int
            and type(self.char_end) is int
            and 0 <= self.char_start < self.char_end < 2**63
            and digest_ok(self.content_hash)
            and valid_id(self.origin)
        )


@dataclass(frozen=True, slots=True)
class Alias:
    text: str = field(repr=False)
    entity_id: str

    def __post_init__(self):
        require(text_ok(self.text, 64) and valid_id(self.entity_id))


@dataclass(frozen=True, slots=True)
class EventRecord:
    event_id: str
    namespace: str
    kind: str
    facts: tuple[str, ...] = field(repr=False)
    fact_type: str
    uncertainty: str = field(repr=False)
    event_time: str
    recorded_at: str
    version: int
    status: Literal["active", "superseded", "revoked"]
    supersedes: int | None
    content_hash: str
    sources: tuple[SourceSpan, ...] = field(repr=False)
    entity_ids: tuple[str, ...]  # empty = unknown
    scope_ids: tuple[str, ...]
    purpose: str
    read: bool
    train: bool
    group_disclose: bool
    index: bool
    terms: tuple[str, ...] = field(default=(), repr=False)
    aliases: tuple[Alias, ...] = field(default=(), repr=False)
    related_event_ids: tuple[str, ...] = ()
    schema: str = EVENT_SCHEMA

    def __post_init__(self):
        require(
            self.schema == EVENT_SCHEMA
            and all(
                map(
                    valid_id,
                    (self.event_id, self.namespace, self.purpose, self.fact_type),
                )
            )
            and self.kind in CARD_KINDS
            and text_ok(self.uncertainty, 160)
            and type(self.version) is int
            and 0 < self.version < 2**63
            and self.status in ("active", "superseded", "revoked")
            and (
                self.supersedes is None
                or (type(self.supersedes) is int and 0 < self.supersedes < self.version)
            )
            and all(
                type(x) is bool
                for x in (self.read, self.train, self.group_disclose, self.index)
            )
        )
        for ids in (self.scope_ids, self.entity_ids, self.related_event_ids):
            require(
                isinstance(ids, tuple)
                and len(ids) <= 64
                and all(map(valid_id, ids))
                and len(set(ids)) == len(ids)
            )
        require(
            isinstance(self.sources, tuple)
            and 0 < len(self.sources) <= 16
            and all(isinstance(s, SourceSpan) for s in self.sources)
            and isinstance(self.aliases, tuple)
            and len(self.aliases) <= 16
            and all(
                isinstance(a, Alias) and a.entity_id in self.entity_ids
                for a in self.aliases
            )
            and isinstance(self.terms, tuple)
            and len(self.terms) <= 32
            and all(text_ok(t, 64) for t in self.terms)
            and isinstance(self.facts, tuple)
            and 0 < len(self.facts) <= 6
            and all(text_ok(f, 320) for f in self.facts)
            and sum(map(len, self.facts)) <= 640
        )
        require(sha(canonical(self.facts)) == self.content_hash, "MEMORY_HASH_MISMATCH")
        for value in (self.event_time, self.recorded_at):
            if value != UNKNOWN:
                clock(value)

    @classmethod
    def from_dict(cls, value):
        try:
            row = dict(value)
            for key in (
                "facts",
                "entity_ids",
                "scope_ids",
                "terms",
                "related_event_ids",
            ):
                if key in row:
                    require(isinstance(row[key], list))
                    row[key] = tuple(row[key])
            row["sources"] = tuple(SourceSpan(**s) for s in row["sources"])
            row["aliases"] = tuple(Alias(**a) for a in row.get("aliases", ()))
            require(row.get("schema") == EVENT_SCHEMA)
            return cls(**row)
        except (TypeError, KeyError):
            raise MemoryErrorV1("MEMORY_SCHEMA_INVALID") from None


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    scope: TrustedScope
    text: str = field(repr=False)
    context_keywords: tuple[str, ...] = field(default=(), repr=False)
    entity_ids: tuple[str, ...] = ()
    event_from: str | None = None
    event_until: str | None = None  # exclusive
    knowledge_as_of: str | None = None  # can only narrow; never revives old ACL/version
    k: int = 3
    related_limit: int = 0
    schema: str = QUERY_SCHEMA

    def __post_init__(self):
        require(
            self.schema == QUERY_SCHEMA
            and isinstance(self.scope, TrustedScope)
            and text_ok(self.text, 4096)
            and isinstance(self.context_keywords, tuple)
            and len(self.context_keywords) <= 8
            and all(text_ok(t, 64) for t in self.context_keywords)
            and isinstance(self.entity_ids, tuple)
            and len(self.entity_ids) <= 8
            and all(map(valid_id, self.entity_ids))
            and type(self.k) is int
            and 0 <= self.k <= 16
            and type(self.related_limit) is int
            and 0 <= self.related_limit <= 2
        )
        for value in (self.event_from, self.event_until, self.knowledge_as_of):
            if value is not None:
                clock(value)
        if self.event_from is not None and self.event_until is not None:
            require(clock(self.event_from) < clock(self.event_until))


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    cards: tuple[ContextCard, ...] = field(repr=False)
    audit: tuple[CardAudit, ...]
    reason: str
    elapsed_ms: float
    scored_count: int


class RetrievalMemoryProvider(Protocol):
    def retrieve(self, query: MemoryQuery) -> RetrievalResult: ...
