"""Adapt the supplied fictional example archive, without executing its scripts."""

import json
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

from lm_trainer.memory import Alias, EventRecord, SourceSpan, TrustedScope
from lm_trainer.memory.schema import BUNDLE_SCHEMA, UNKNOWN, canonical, require, sha

ARCHIVE = (
    Path(__file__).resolve().parents[1]
    / "1"
    / "rag参考文件"
    / "Intelligent_Support_RAG_Fixtures.zip"
)

# Reviewed synthetic search vocabulary, not part of event facts and not used for ACL.
TERMS = {
    "incident_a_status": ("修好", "恢复在线", "网口松动", "复机", "证实是"),
    "incident_a_initial": ("报障", "最开始", "打印任务", "查到", "打印机离线"),
    "plan_a_cable": ("计划检查", "计划换线", "拟更换"),
    "repair_a_cable": ("换网线", "无需更换", "固定网口", "线缆更换"),
    "incident_b_scan": ("扫码", "条码识别"),
    "topic_a_parts": ("托盘", "到货", "备件"),
    "private_a_address": ("上门", "私下", "上门安排"),
    "tg_a_status": ("标签机", "缺纸"),
    "faq_port": ("设备离线", "先检查", "网口指示灯"),
    "other_network": ("网络设备", "重启故障"),
    "eval_motor": ("电机", "线束"),
    "unindexed_detail": ("检查记录",),
    "restricted_detail": ("检查资料", "内部检查"),
    "revoked_a_note": ("重置", "全部设置"),
    "future_a_delivery": ("物流", "到货"),
    "guess_a": ("路由器",),
    "unknown_date_a": ("类似故障", "哪天"),
    "injection_note": ("演示日志",),
    "alias_a_device": ("小蓝",),
    "chat_a_light": ("查到", "网口灯", "灯不亮"),
    "derived_a_intersection": ("共享事件",),
}

TABLES = (
    "scopes",
    "sources",
    "chunks",
    "permissions",
    "events",
    "aliases",
    "entities",
    "event_links",
    "queries",
    "revocations",
    "negative_cases",
)


def read_archive(path=ARCHIVE):
    with ZipFile(path) as archive:
        manifest = dict(
            line.split()[::-1]
            for line in archive.read("SHA256SUMS").decode().splitlines()
        )
        metadata = archive.read("manifest.json")
        require(
            sha256(metadata).hexdigest() == manifest.get("manifest.json"),
            "MEMORY_HASH_MISMATCH",
        )
        info = json.loads(metadata)
        require(
            info["schema"] == "support.memory.fixture.v1"
            and info["fixture_only"] is True
            and info["production_authorized"] is False
            and info["training_authorized"] is False
        )
        tables = {}
        for name in TABLES:
            member = f"{name}.jsonl"
            require(archive.getinfo(member).file_size <= 256_000)
            content = archive.read(member)
            require(
                sha256(content).hexdigest() == manifest.get(member),
                "MEMORY_HASH_MISMATCH",
            )
            rows = [json.loads(line) for line in content.splitlines()]
            require(
                len(rows) == info["counts"][name]
                and all(row["fixture_only"] is True for row in rows)
            )
            tables[name] = rows
        return tables


def unique(rows, key):
    by_id = {row[key]: row for row in rows}
    require(len(by_id) == len(rows))
    return by_id


def scopes_for(tables):
    scopes = []
    for s in tables["scopes"]:
        identity = (s["scope_id"], s["namespace"], s["purpose"])
        if s["platform"] == "web":
            scopes.append(
                TrustedScope(
                    *identity,
                    "workbench",
                    web_account=s["chat_id"],
                    session_id=s["session_id"],
                )
            )
        else:
            scopes.append(
                TrustedScope(
                    *identity,
                    "group",
                    channel=s["platform"],
                    client_id=s["bot_id"],
                    room_id=s["chat_id"],
                    thread_id=s["thread_id"],
                )
            )
    return tuple(scopes)


def adapt(tables):
    """Require exact source hashes/spans; intersect every event, chunk and source ACL."""
    scopes = scopes_for(tables)
    acl = unique(tables["permissions"], "permission_id")
    sources = unique(tables["sources"], "source_id")
    chunks = unique(tables["chunks"], "chunk_id")
    entities = unique(tables["entities"], "entity_id")
    for src in sources.values():
        require(sha(src["text"]) == src["content_sha256"], "MEMORY_HASH_MISMATCH")
    for chunk in chunks.values():
        src = sources.get(chunk["source_id"])
        start, end = chunk["char_start"], chunk["char_end"]
        require(
            src is not None
            and 0 <= start < end <= len(src["text"])
            and src["text"][start:end] == chunk["text"]
            and sha(chunk["text"]) == chunk["content_sha256"],
            "MEMORY_SOURCE_SPAN_INVALID",
        )
    records, evidence = _records(tables, acl, sources, chunks, entities, scopes)
    records = _links(tables, records, evidence, acl)
    return scopes, tuple(records.values())


def _allow(permissions, scope):
    return all(
        p["namespace"] == scope.namespace
        and scope.purpose in p["allowed_purposes"]
        and scope.scope_id in p["allowed_scope_ids"]
        and p["read_authorized"]
        and p["index_authorized"]
        and (scope.kind != "group" or p["disclosure_authorized"])
        for p in permissions
    )


def _records(tables, acl, sources, chunks, entities, scopes):
    evidence, records = {}, {}
    for e in tables["events"]:
        if e["review_status"] != "verified_fixture":
            continue
        ref = (e["event_id"], e["version"])
        require(ref not in records, "MEMORY_FIXTURE_DUPLICATE")
        namespace, purpose = e["namespace"], e["split"]
        ids = sorted(
            set(e["evidence_chunk_ids"]).union(
                *(c["evidence_chunk_ids"] for c in e["claims"])
            )
        )
        evidence[ref] = set(ids)
        pointers, approvals = [], [acl[e["permission_id"]]]
        for chunk_id in ids:
            require(chunk_id in chunks, "MEMORY_EVIDENCE_UNKNOWN")
            chunk = chunks[chunk_id]
            src = sources[chunk["source_id"]]
            require((src["namespace"], src["split"]) == (namespace, purpose))
            approvals.extend((acl[chunk["permission_id"]], acl[src["permission_id"]]))
            pointers.append(
                SourceSpan(
                    src["source_id"],
                    f"fixture:{src['source_id']}#{chunk_id}",
                    chunk["char_start"],
                    chunk["char_end"],
                    chunk["content_sha256"],
                    src["origin"],
                )
            )
        allowed = tuple(
            s.scope_id for s in scopes if s.purpose == purpose and _allow(approvals, s)
        )
        aliases = []
        for a in tables["aliases"]:
            if (
                a["review_status"] == "verified_fixture"
                and a["entity_id"] in e["subject_ids"]
                and set(a["evidence_chunk_ids"]) <= set(ids)
                and all(
                    _allow([acl[a["permission_id"]]], s)
                    for s in scopes
                    if s.scope_id in allowed
                )
            ):
                aliases.append(Alias(a["alias"], a["entity_id"]))
        require(all(id_ in entities for id_ in e["subject_ids"]))
        claims = e["claims"]
        types = {c["claim_type"] for c in claims}
        require(len(types) == 1, "MEMORY_SPLIT_MIXED_FACT_TYPES")
        uncertainty = " / ".join(dict.fromkeys(c["uncertainty"] for c in claims))
        facts = tuple(c["text"] for c in claims)
        kind = {
            "guess_a": "historical_judgment",
            "chat_a_light": "shift_log",
            "repair_a_cable": "confirmed_fact",
        }.get(e["event_id"], e["kind"])
        records[ref] = EventRecord(
            e["event_id"],
            namespace,
            kind,
            facts,
            types.pop(),
            uncertainty,
            e["event_time"] or UNKNOWN,
            e["recorded_at"] or UNKNOWN,
            e["version"],
            e["status"],
            e["supersedes"],
            sha(canonical(facts)),
            tuple(pointers),
            tuple(e["subject_ids"]),
            allowed,
            purpose,
            all(p["read_authorized"] for p in approvals),
            all(p["training_authorized"] for p in approvals),
            all(p["disclosure_authorized"] for p in approvals),
            all(p["index_authorized"] for p in approvals),
            TERMS.get(e["event_id"], ()),
            tuple(aliases),
        )
    return records, evidence


def _links(tables, records, evidence, acl):
    latest = {id_: ref for ref in sorted(records) for id_ in ref[:1]}
    for link in tables["event_links"]:
        if link["review_status"] != "verified_fixture":
            continue
        origin, target = latest.get(link["from_event"]), latest.get(link["to_event"])
        require(
            origin is not None
            and target is not None
            and set(link["evidence_chunk_ids"]) <= (evidence[origin] | evidence[target])
        )
        event, neighbor = records[origin], records[target]
        permission = acl[link["permission_id"]]
        if (
            set(event.scope_ids) <= set(neighbor.scope_ids)
            and event.namespace == neighbor.namespace
            and permission["namespace"] == event.namespace
            and event.purpose in permission["allowed_purposes"]
            and set(event.scope_ids) <= set(permission["allowed_scope_ids"])
            and permission["read_authorized"]
            and permission["index_authorized"]
            and permission["disclosure_authorized"]
        ):
            records[origin] = replace(
                event, related_event_ids=(*event.related_event_ids, neighbor.event_id)
            )
    return records


def bundle(tables):
    scopes, records = adapt(tables)
    return {
        "schema": BUNDLE_SCHEMA,
        "scopes": [asdict(s) for s in scopes],
        "events": [asdict(e) for e in records],
        "revoked_event_ids": [r["event_id"] for r in tables["revocations"]],
    }


if __name__ == "__main__":
    print(json.dumps(bundle(read_archive()), ensure_ascii=False))
