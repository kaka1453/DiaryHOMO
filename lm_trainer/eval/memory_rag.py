"""M5 fixture-only retrieval and real native-tokenizer budget evaluation (no model generation)."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from statistics import median

from examples.conversation_renderer.trace import DEFAULT_TOKENIZER, STOP
from examples.memory_rag.fixture import ARCHIVE, adapt, read_archive
from lm_trainer.chat import ContextBuilder, RoleConfig
from lm_trainer.memory import (
    MemoryQuery,
    MemoryStore,
    SqliteMemoryProvider,
    TrustedScope,
)
from lm_trainer.memory.schema import UNKNOWN, clock, require
from lm_trainer.memory.store import STORE_NAME
from lm_trainer.template_policy import resolve_policy

SYSTEM = (
    "Fictional industrial service support. Separate observations, hypotheses and plans."
)


def load_tokenizer(path=DEFAULT_TOKENIZER):
    from transformers import AutoTokenizer

    require(Path(path).is_dir(), "LOCAL_TOKENIZER_REQUIRED")
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    _, policy = resolve_policy(tokenizer, {"stop_tokens": [STOP]})
    return tokenizer, policy


def _ref(card_or_audit):
    return f"{card_or_audit.card_id}@{card_or_audit.version}"


def _eligibility(tables, records, case, scope, actual):
    # Independent fixture oracle, including forbidden gold refs, not a rerun of SQL checks.
    latest = {
        event.event_id: event for event in sorted(records, key=lambda e: e.version)
    }
    revoked = {r["event_id"] for r in tables["revocations"]}
    forbidden = set(case["forbidden_by_policy"])
    unauthorized, stale = set(), set()
    for ref in actual:
        id_, ver = ref.rsplit("@", 1)
        e = latest.get(id_)
        if (
            id_ in revoked
            or e is not None
            and (int(ver) != e.version or e.status != "active")
        ):
            stale.add(ref)
        elif (
            e is None
            or ref in forbidden
            or not e.index
            or not e.read
            or scope.scope_id not in e.scope_ids
            or (e.namespace, e.purpose) != (scope.namespace, scope.purpose)
            or (scope.kind == "group" and not e.group_disclose)
            or e.recorded_at == UNKNOWN
            or clock(e.recorded_at) > clock(case["knowledge_as_of"])
            or e.event_time != UNKNOWN
            and clock(e.event_time) > clock(case["knowledge_as_of"])
        ):
            unauthorized.add(ref)
    return unauthorized, stale


def run_fixture(store, tokenizer, policy, *, tables, k=3, card_token_limit=400):
    scopes, records = adapt(tables)
    store.update(
        records,
        scopes=scopes,
        revoked_event_ids=[r["event_id"] for r in tables["revocations"]],
    )
    by_scope = {s.scope_id: s for s in scopes}
    provider = SqliteMemoryProvider(store)
    builder = ContextBuilder(max_cards=k, card_token_limit=card_token_limit)
    rows = []
    for case in tables["queries"]:
        scope = by_scope.get(case["scope_id"]) or TrustedScope(
            case["scope_id"],
            "unregistered",
            "development",
            "workbench",
            web_account="fixture-unknown",
            session_id="fixture-unknown",
        )
        query = MemoryQuery(
            scope, case["query"], knowledge_as_of=case["knowledge_as_of"], k=k
        )
        retrieved = provider.retrieve(query)
        built = builder.build_inference_v1(
            tokenizer,
            policy,
            role=RoleConfig("fixture", scope.namespace, "1", SYSTEM),
            messages=(("user", query.text),),
            observation=None,
            target_person=None,
            cards=retrieved.cards,
            audit_entries=retrieved.audit,
            context_limit=2048,
            max_new_tokens=32,
            card_guard=partial(store.prompt_guard, scope, query=query),
        )
        recalled = [_ref(c) for c in retrieved.cards]
        inserted = [_ref(a) for a in built.card_audit if a.reason == "injected"]
        required = set(case["required_event_refs"])
        relevant = required | set(case["alternative_event_refs"])
        bad, stale = _eligibility(
            tables, records, case, scope, set(recalled) | set(inserted)
        )
        rows.append(
            {
                "query_id": case["query_id"],
                "reason": retrieved.reason,
                "scored_count": retrieved.scored_count,
                "required": sorted(required),
                "recalled": recalled,
                "inserted": inserted,
                "recall_at_k": len(required.intersection(recalled)) / len(required)
                if required
                else None,
                "unauthorized_hit_count": len(bad),
                "revoked_or_stale_hit_count": len(stale),
                "irrelevant_insert_count": len(set(inserted) - relevant),
                "empty_expected": case["expect_no_relevant_evidence"],
                "injected_tokens": built.injected_tokens,
                "retrieval_ms": round(retrieved.elapsed_ms, 3),
                "audit": [
                    {"ref": _ref(a), "reason": a.reason} for a in built.card_audit
                ],
            }
        )
    return _report(rows, k, card_token_limit)


def _report(rows, k, card_token_limit):
    required = sum(len(row["required"]) for row in rows)
    recalled = sum(
        len(set(row["required"]).intersection(row["recalled"])) for row in rows
    )
    inserted = sum(len(row["inserted"]) for row in rows)
    irrelevant = sum(row["irrelevant_insert_count"] for row in rows)
    return {
        "schema": "lm_trainer.eval-memory-rag.v1",
        "condition": "M5",
        "fixture_only": True,
        "k": k,
        "generation_status": "MODEL_ARTIFACT_PENDING",
        "query_count": len(rows),
        "card_token_limit": card_token_limit,
        "recall_at_k": recalled / required if required else None,
        "required_hits": recalled,
        "required_total": required,
        "inserted_count": inserted,
        "unauthorized_hit_count": sum(row["unauthorized_hit_count"] for row in rows),
        "revoked_or_stale_hit_count": sum(
            row["revoked_or_stale_hit_count"] for row in rows
        ),
        "irrelevant_insert_rate": irrelevant / inserted if inserted else 0,
        "injected_tokens": sum(row["injected_tokens"] for row in rows),
        "retrieval_p50_ms": median(row["retrieval_ms"] for row in rows),
        "retrieval_max_ms": max(row["retrieval_ms"] for row in rows),
        "queries": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="fresh private directory outside the repo",
    )
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--k", type=int, choices=range(1, 17), default=3)
    parser.add_argument("--card-tokens", type=int, default=400)
    args = parser.parse_args(argv)
    if not 0 <= args.card_tokens <= 1900:
        parser.error("--card-tokens must be between 0 and 1900")
    if (args.data_dir / STORE_NAME).exists():
        parser.error("M5 requires a fresh authority directory")
    tables = read_archive(args.archive)
    tokenizer, policy = load_tokenizer(args.tokenizer)
    report = run_fixture(
        MemoryStore(args.data_dir, create=True),
        tokenizer,
        policy,
        tables=tables,
        k=args.k,
        card_token_limit=args.card_tokens,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(
        bool(report["unauthorized_hit_count"] or report["revoked_or_stale_hit_count"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
