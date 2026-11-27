"""Private SQLite authority and rebuildable (non-authoritative) lexical postings."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
import unicodedata
from collections import Counter
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from lm_trainer.chat.memory import CardPermissions, ContextCard, audit
from lm_trainer.chat.types import valid_id

from .schema import UNKNOWN, EventRecord, TrustedScope, canonical, clock, require, sha

STORE_VERSION = 1
STORE_NAME = "memory-v1.sqlite3"
REPO = Path(__file__).resolve().parents[2]
SCHEMA = """
CREATE TABLE scopes (id TEXT PRIMARY KEY, binding TEXT NOT NULL UNIQUE);
CREATE TABLE family (
    id TEXT PRIMARY KEY, version INTEGER NOT NULL, state TEXT NOT NULL,
    digest TEXT, payload TEXT, namespace TEXT, purpose TEXT,
    read_allowed INTEGER, disclose INTEGER, indexed INTEGER,
    occurred TEXT, recorded TEXT
);
CREATE TABLE seen (id TEXT NOT NULL, version INTEGER NOT NULL, digest TEXT NOT NULL,
                   PRIMARY KEY(id, version)) WITHOUT ROWID;
CREATE TABLE grants (scope TEXT NOT NULL REFERENCES scopes(id),
                     id TEXT NOT NULL REFERENCES family(id), PRIMARY KEY(scope, id)) WITHOUT ROWID;
CREATE INDEX grants_family ON grants(id);
CREATE TABLE source_members (source TEXT NOT NULL, id TEXT NOT NULL, PRIMARY KEY(source, id)) WITHOUT ROWID;
CREATE TABLE source_denials (source TEXT PRIMARY KEY);
CREATE TABLE postings (token TEXT NOT NULL, id TEXT NOT NULL REFERENCES family(id),
                       PRIMARY KEY(token, id)) WITHOUT ROWID;
CREATE INDEX postings_family ON postings(id);
"""


def norm(text):
    return unicodedata.normalize("NFKC", text).casefold()


_CJK = r"\u3400-\u9fff\uf900-\ufaff"
_WORDS = re.compile(rf"[{_CJK}]+|[a-z0-9]+")


def terms_for(text):
    normalized = norm(text)
    terms = set()
    for word in _WORDS.findall(normalized):
        if ord(word[0]) >= 128:
            if len(word) == 1:
                terms.add(word)
            else:
                terms.update(word[i : i + 2] for i in range(len(word) - 1))
        elif len(word) <= 64:
            terms.add(word)
    return terms


def search_document(event):
    return (
        *event.facts,
        *event.terms,
        *event.entity_ids,
        *(a.text for a in event.aliases),
    )


def _stamp(value):
    return None if value == UNKNOWN else clock(value)


def _scope_value(scope):
    return canonical(asdict(scope))


def _card(event, scope):
    return ContextCard(
        event.event_id,
        event.namespace,
        event.kind,
        event.facts,
        event.event_time,
        event.recorded_at,
        event.version,
        "active",
        event.supersedes,
        event.sources[0].source_ref,
        f"{event.fact_type}: {event.uncertainty}",
        CardPermissions(event.read, event.train),
        "session_only",
        scope.scope_id,
    )


class MemoryStore:
    """The directory is chosen by a trusted operator, not a request or environment default."""

    def __init__(self, directory, *, create=False):
        supplied = Path(directory).absolute()
        directory = supplied.resolve()
        require(
            directory == supplied
            and not directory.is_relative_to(REPO)
            and all(not (p / ".git").exists() for p in (directory, *directory.parents)),
            "MEMORY_PRIVATE_PATH_REQUIRED",
        )
        if create:
            directory.mkdir(mode=0o700, exist_ok=True)
        self._private(directory, stat.S_ISDIR)
        self.path = directory / STORE_NAME
        if create and not self.path.exists():
            with closing_directory_file(self.path), closing(self._connect()) as db:
                db.executescript(
                    f"BEGIN IMMEDIATE; {SCHEMA} PRAGMA user_version={STORE_VERSION}; COMMIT;"
                )
        self._private(self.path, stat.S_ISREG)
        with self.transaction():
            pass

    @staticmethod
    def _private(path, kind):
        require(path.exists(), "MEMORY_AUTHORITY_REQUIRED")
        st = path.lstat()
        require(
            kind(st.st_mode) and st.st_uid == os.getuid() and not st.st_mode & 0o077,
            "MEMORY_PRIVATE_PATH_REQUIRED",
        )
        if kind is stat.S_ISREG:
            require(st.st_nlink == 1, "MEMORY_PRIVATE_PATH_REQUIRED")

    def _connect(self):
        db = sqlite3.connect(f"{self.path.as_uri()}?mode=rw", uri=True, timeout=1.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA secure_delete=ON")
        db.execute("PRAGMA trusted_schema=OFF")
        return db

    @contextmanager
    def transaction(self, *, write=False):
        self._private(self.path, stat.S_ISREG)
        with closing(self._connect()) as db, db:
            require(
                db.execute("PRAGMA user_version").fetchone()[0] == STORE_VERSION,
                "MEMORY_AUTHORITY_REQUIRED",
            )
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db

    @staticmethod
    def scope_known(db, scope):
        row = db.execute(
            "SELECT binding FROM scopes WHERE id=?", (scope.scope_id,)
        ).fetchone()
        return row is not None and row[0] == _scope_value(scope)

    @staticmethod
    def _purge(db, event_id):
        db.execute("DELETE FROM postings WHERE id=?", (event_id,))
        db.execute("DELETE FROM grants WHERE id=?", (event_id,))

    @classmethod
    def _terminal(cls, db, event_id, state, version=0):
        cls._purge(db, event_id)
        db.execute(
            """INSERT INTO family (id, version, state) VALUES (?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET version=max(version, excluded.version), state=excluded.state,
               payload=NULL, digest=NULL, namespace=NULL, purpose=NULL, read_allowed=NULL,
               disclose=NULL, indexed=NULL, occurred=NULL, recorded=NULL""",
            (event_id, version, state),
        )

    @staticmethod
    def _post(db, event):
        if event.status == "active" and event.index and event.read and event.scope_ids:
            tokens = set().union(*(terms_for(t) for t in search_document(event)))
            db.executemany(
                "INSERT INTO postings VALUES (?, ?)",
                ((t, event.event_id) for t in sorted(tokens)),
            )

    def update(self, records, *, scopes=(), revoked_event_ids=()):
        """Trusted ingestion only; versions and conflict/denial tombstones survive rebuilds."""
        outcomes = {}
        with self.transaction(write=True) as db:
            for scope in scopes:
                require(isinstance(scope, TrustedScope))
                db.execute(
                    "INSERT OR IGNORE INTO scopes VALUES (?, ?)",
                    (scope.scope_id, _scope_value(scope)),
                )
                require(self.scope_known(db, scope), "MEMORY_SCOPE_CONFLICT")
            for id_ in revoked_event_ids:
                require(valid_id(id_))
                row = db.execute(
                    "SELECT state FROM family WHERE id=?", (id_,)
                ).fetchone()
                if row is None or row[0] != "deleted":
                    self._terminal(db, id_, "revoked")
            for event in sorted(records, key=lambda r: (r.event_id, r.version)):
                outcomes[(event.event_id, event.version)] = self._upsert(db, event)
        return outcomes

    @classmethod
    def _upsert(cls, db, event):
        require(isinstance(event, EventRecord))
        event_id = event.event_id
        old = db.execute(
            "SELECT version, state, namespace FROM family WHERE id=?", (event_id,)
        ).fetchone()
        if old and old["state"] in ("revoked", "deleted", "duplicate_version"):
            return old["state"]
        payload = canonical(asdict(event))
        digest = sha(payload)
        seen = db.execute(
            "SELECT digest FROM seen WHERE id=? AND version=?",
            (event_id, event.version),
        ).fetchone()
        if (seen and seen[0] != digest) or (
            old and old["namespace"] != event.namespace
        ):
            cls._terminal(db, event_id, "duplicate_version", event.version)
            return "duplicate_version"
        if event.status == "revoked":
            cls._terminal(db, event_id, "revoked", event.version)
            return "revoked"
        if old and old["version"] > event.version:
            return "stale"
        if seen:
            return "unchanged"
        if old:
            require(event.supersedes == old["version"], "MEMORY_VERSION_CHAIN_INVALID")
        for scope_id in event.scope_ids:
            row = db.execute(
                "SELECT binding FROM scopes WHERE id=?", (scope_id,)
            ).fetchone()
            require(row is not None, "MEMORY_SCOPE_UNKNOWN")
            binding = json.loads(row[0])
            require(
                (event.namespace, event.purpose)
                == (binding["namespace"], binding["purpose"]),
                "MEMORY_SCOPE_CONFLICT",
            )
        for source_id in {s.source_id for s in event.sources}:
            if db.execute(
                "SELECT 1 FROM source_denials WHERE source=?", (source_id,)
            ).fetchone():
                cls._terminal(db, event_id, "revoked", event.version)
                return "revoked"
            db.execute(
                "INSERT OR IGNORE INTO source_members VALUES (?, ?)",
                (source_id, event_id),
            )
        cls._purge(db, event_id)
        db.execute(
            "INSERT INTO seen VALUES (?, ?, ?)", (event_id, event.version, digest)
        )
        db.execute(
            """INSERT OR REPLACE INTO family VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                event.version,
                event.status,
                digest,
                payload,
                event.namespace,
                event.purpose,
                event.read,
                event.group_disclose,
                event.index,
                _stamp(event.event_time),
                _stamp(event.recorded_at),
            ),
        )
        db.executemany(
            "INSERT INTO grants VALUES (?, ?)", ((s, event_id) for s in event.scope_ids)
        )
        cls._post(db, event)
        return "updated"

    def deny(self, *, event_id=None, source_id=None, delete=False):
        require(
            (valid_id(event_id) and source_id is None)
            or (valid_id(source_id) and event_id is None)
        )
        reason = "deleted" if delete else "revoked"
        with self.transaction(write=True) as db:
            if source_id is not None:
                db.execute(
                    "INSERT OR IGNORE INTO source_denials VALUES (?)", (source_id,)
                )
                ids = [
                    row[0]
                    for row in db.execute(
                        "SELECT id FROM source_members WHERE source=?", (source_id,)
                    )
                ]
            else:
                ids = [event_id]
            for id_ in ids:
                row = db.execute(
                    "SELECT state FROM family WHERE id=?", (id_,)
                ).fetchone()
                if row is None or row[0] != "deleted":
                    self._terminal(db, id_, reason)
            return len(ids)

    @staticmethod
    def event(row):
        require(
            row["payload"] and sha(row["payload"]) == row["digest"],
            "MEMORY_AUTHORITY_INCONSISTENT",
        )
        event = EventRecord.from_dict(json.loads(row["payload"]))
        require(
            (
                event.event_id,
                event.version,
                event.status,
                event.namespace,
                event.purpose,
                event.read,
                event.group_disclose,
                event.index,
                _stamp(event.event_time),
                _stamp(event.recorded_at),
            )
            == tuple(
                row[k]
                for k in (
                    "id",
                    "version",
                    "state",
                    "namespace",
                    "purpose",
                    "read_allowed",
                    "disclose",
                    "indexed",
                    "occurred",
                    "recorded",
                )
            ),
            "MEMORY_AUTHORITY_INCONSISTENT",
        )
        return event

    def rebuild(self):
        with self.transaction(write=True) as db:
            db.execute("DELETE FROM postings")
            for row in db.execute("SELECT * FROM family WHERE state='active'"):
                self._post(db, self.event(row))
            return db.execute("SELECT count(DISTINCT id) FROM postings").fetchone()[0]

    def audit(self, *, scope_id=None):
        with self.transaction() as db:
            if scope_id is not None:
                row = db.execute(
                    "SELECT binding FROM scopes WHERE id=?", (scope_id,)
                ).fetchone()
                require(row is not None, "MEMORY_SCOPE_UNKNOWN")
                scope = TrustedScope(**json.loads(row[0]))
                sql, params = self.allow_clause(scope, *self.bounds())
                return {
                    "eligible_events": sum(
                        self.allows_payload(self.event(r), scope)
                        for r in db.execute(sql, params)
                    )
                }
            return {
                "states": dict(
                    db.execute("SELECT state,count(*) FROM family GROUP BY state")
                ),
                "indexed_events": db.execute(
                    "SELECT count(DISTINCT id) FROM postings"
                ).fetchone()[0],
                "denied_sources": db.execute(
                    "SELECT count(*) FROM source_denials"
                ).fetchone()[0],
                "registered_scopes": db.execute(
                    "SELECT count(*) FROM scopes"
                ).fetchone()[0],
            }

    @staticmethod
    def allow_clause(scope, lower, upper, as_of):
        return (
            """
            SELECT f.* FROM grants g JOIN family f ON f.id=g.id
            WHERE g.scope=? AND f.namespace=? AND f.purpose=? AND f.state='active'
                  AND f.read_allowed=1 AND f.indexed=1 AND (?=0 OR f.disclose=1)
                  AND (f.occurred IS NULL OR f.occurred<=?)
                  AND f.recorded<=?
                  AND (? IS NULL OR f.occurred>=?) AND (? IS NULL OR f.occurred<?)
        """,
            (
                scope.scope_id,
                scope.namespace,
                scope.purpose,
                int(scope.kind == "group"),
                as_of,
                as_of,
                lower,
                lower,
                upper,
                upper,
            ),
        )

    @staticmethod
    def bounds(query=None):
        now = datetime.now(UTC).isoformat(timespec="microseconds")
        lower = clock(query.event_from) if query and query.event_from else None
        upper = clock(query.event_until) if query and query.event_until else None
        as_of = (
            min(now, clock(query.knowledge_as_of))
            if query and query.knowledge_as_of
            else now
        )
        return lower, upper, as_of

    @staticmethod
    def allows_payload(event, scope):
        return (
            (event.namespace, event.purpose) == (scope.namespace, scope.purpose)
            and event.status == "active"
            and event.read
            and event.index
            and scope.scope_id in event.scope_ids
            and (scope.kind != "group" or event.group_disclose)
        )

    @contextmanager
    def prompt_guard(self, scope, cards, *, query=None):
        """Hold a DB writer reservation across final checks and synchronous prompt assembly."""
        with self.transaction(write=True) as db:
            allowed, rejected = [], []
            known = scope is not None and self.scope_known(db, scope)
            sql, params = (
                self.allow_clause(scope, *self.bounds(query)) if known else (None, ())
            )
            copies = Counter((c.card_id, c.version) for c in cards)
            for card in cards:
                if copies[(card.card_id, card.version)] != 1:
                    rejected.append(audit(card, "duplicate_version"))
                    continue
                reason = "scope"
                if (
                    known
                    and card.namespace == scope.namespace
                    and card.session_id == scope.scope_id
                ):
                    row = db.execute(
                        "SELECT * FROM family WHERE id=?", (card.card_id,)
                    ).fetchone()
                    granted = db.execute(
                        f"SELECT 1 FROM ({sql}) WHERE id=?", (*params, card.card_id)
                    ).fetchone()
                    if granted:
                        current = self.event(row)
                        if not self.allows_payload(current, scope):
                            reason = "scope"
                        elif card.version != current.version:
                            reason = "stale"
                        elif card == _card(current, scope):
                            allowed.append(card)
                            continue
                        else:
                            reason = "duplicate_version"
                    elif row is not None and row["state"] in (
                        "revoked",
                        "deleted",
                        "duplicate_version",
                    ):
                        reason = (
                            "duplicate_version"
                            if row["state"] == "duplicate_version"
                            else "revoked"
                        )
                    else:
                        reason = "read_denied"
                rejected.append(audit(card, reason))
            yield tuple(allowed), tuple(rejected)


@contextmanager
def closing_directory_file(path):
    """Reserve an owner-only file before SQLite writes; never unlink a failed initialization."""
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_WRONLY, 0o600)
    try:
        yield
    finally:
        os.close(fd)
