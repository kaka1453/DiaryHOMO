"""Query and Task C adapter. No history writes, inference calls, or caches."""

from __future__ import annotations

import collections
import logging
import re
from time import perf_counter
from types import MappingProxyType

from lm_trainer.chat.memory import audit

from .schema import MemoryQuery, RetrievalResult, TrustedScope, require
from .store import _card, norm, terms_for

AUDIT_LOG = logging.getLogger("lm_trainer.memory.audit")
MAX_CANDIDATES = 256
MAX_LOOKUP_TERMS = 192


def _matches(phrase, text):
    phrase = norm(phrase).strip()
    if len(phrase) == 1:
        return phrase == text.strip()
    if phrase.isascii():
        return (
            re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text)
            is not None
        )
    return phrase in text


# Exact multi-character domain anchor fallback, not a general Chinese segmenter.
_CJK_ANCHOR = re.compile(r"[\u3400-\u9fff]{5}")


def _score(event, text, hints, entities):
    terms = set(event.terms) | set(event.entity_ids)
    exact = {t for t in terms if _matches(t, text)}
    context = sum(any(_matches(t, hint) for hint in hints) for t in terms - exact)
    if exact or context:
        return (
            4 * len(exact)
            + 2 * context
            + int(bool(entities.intersection(event.entity_ids)))
        )
    # Reviewed terms are preferred; conservative fallback for prose with an exact
    # contiguous anchor. Neither date nor a known entity alone proves relevance.
    for fact in event.facts:
        normalized = norm(fact)
        if any(
            _CJK_ANCHOR.fullmatch(text[i : i + 5]) and text[i : i + 5] in normalized
            for i in range(len(text) - 4)
        ):
            return 1
    return 0


class SqliteMemoryProvider:
    def __init__(self, store, *, inference_scopes=(), allow_heldout_inference=False):
        self.store = store
        by_id = {}
        for scope in inference_scopes:
            require(isinstance(scope, TrustedScope) and scope.scope_id not in by_id)
            require(
                scope.purpose != "heldout_eval" or allow_heldout_inference is True,
                "MEMORY_HELDOUT_INFERENCE_DENIED",
            )
            by_id[scope.scope_id] = scope
        self._scopes = MappingProxyType(by_id)

    def retrieve(self, query: MemoryQuery):
        require(isinstance(query, MemoryQuery))
        start = perf_counter()
        cards, reason, scored = self._retrieve(query)
        result = _result(cards, reason, scored, start)
        AUDIT_LOG.info(
            "reason=%s card_count=%d scored_count=%d retrieval_ms=%.3f",
            result.reason,
            len(result.cards),
            result.scored_count,
            result.elapsed_ms,
        )
        return result

    def _retrieve(self, query):
        text, hints = norm(query.text), tuple(map(norm, query.context_keywords))
        tokens = sorted(
            set().union(*(terms_for(x) for x in (text, *hints, *query.entity_ids)))
        )
        if query.k == 0 or not tokens or len(tokens) > MAX_LOOKUP_TERMS:
            return (), "budget" if tokens or query.k == 0 else "no_evidence", 0
        with self.store.transaction() as db:
            if not self.store.scope_known(db, query.scope):
                return (), "scope", 0
            sql, params = self.store.allow_clause(
                query.scope, *self.store.bounds(query)
            )
            placeholders = ",".join("?" for _ in tokens)
            rows = db.execute(
                f"""WITH allowed AS MATERIALIZED ({sql})
                SELECT a.* FROM allowed a WHERE EXISTS
                (SELECT 1 FROM postings p WHERE p.id=a.id AND p.token IN ({placeholders}))
                ORDER BY a.id LIMIT ?""",
                (*params, *tokens, MAX_CANDIDATES + 1),
            ).fetchall()
            if len(rows) > MAX_CANDIDATES:
                return (), "budget", 0
            events = [self.store.event(row) for row in rows]
            require(
                all(self.store.allows_payload(e, query.scope) for e in events),
                "MEMORY_AUTHORITY_INCONSISTENT",
            )
            explicit = set(query.entity_ids)
            matches = collections.defaultdict(set)
            for event in events:
                for alias in event.aliases:
                    if _matches(alias.text, text) or any(
                        _matches(alias.text, h) for h in hints
                    ):
                        matches[norm(alias.text)].add(alias.entity_id)
            if any(
                len(ids & explicit if explicit else ids) != 1
                for ids in matches.values()
            ):
                return (), "ambiguous_entity", 0
            resolved = explicit or set().union(*matches.values())
            scored = []
            for event in events:
                if resolved and not resolved.intersection(event.entity_ids):
                    continue
                score = _score(event, text, hints, resolved)
                if score:
                    scored.append((score, event))
            scored.sort(
                key=lambda pair: (
                    -pair[0],
                    pair[1].kind == "device_spec",
                    pair[1].event_id,
                )
            )
            if not scored:
                return (), "no_evidence", len(events)
            # Keep high-relevance direct matches first. An optional verified link
            # may contribute at most two same-entity, independently authorized neighbors.
            chosen = [event for _, event in scored[: query.k]]
            if query.related_limit and scored[0][0] >= 4 and len(chosen) < query.k:
                anchor = chosen[0]
                related = anchor.related_event_ids[:64]
                added = 0
                for id_ in related:
                    if id_ in {e.event_id for e in chosen}:
                        continue
                    row = db.execute(
                        f"SELECT * FROM ({sql}) WHERE id=?", (*params, id_)
                    ).fetchone()
                    if row is None:
                        continue
                    neighbor = self.store.event(row)
                    if self.store.allows_payload(neighbor, query.scope) and set(
                        anchor.entity_ids
                    ).intersection(neighbor.entity_ids):
                        chosen.append(neighbor)
                        added += 1
                        if added >= query.related_limit or len(chosen) >= query.k:
                            break
            return (
                tuple(_card(event, query.scope) for event in chosen),
                "matched",
                len(events),
            )

    def _binding_scope(self, context):
        scope = self._scopes.get(context.binding.scope)
        if (
            scope is None
            or scope.namespace != context.role.namespace
            or (scope.kind == "group") != (context.binding.conversation_mode == "group")
        ):
            return None
        return scope

    def select(self, context):
        """Task C v1 hook; only already-observed text and server binding are provided."""
        scope = self._binding_scope(context)
        if (
            scope is None
            or not context.observed_texts
            or len(context.observed_texts[-1]) > 4096
        ):
            return (), ()
        result = self.retrieve(MemoryQuery(scope, context.observed_texts[-1]))
        return result.cards, result.audit

    def prompt_guard(self, context, cards):
        return self.store.prompt_guard(self._binding_scope(context), cards)


def _result(cards, reason, scored, start):
    return RetrievalResult(
        cards,
        tuple(audit(card, "active") for card in cards),
        reason,
        (perf_counter() - start) * 1000,
        scored,
    )
