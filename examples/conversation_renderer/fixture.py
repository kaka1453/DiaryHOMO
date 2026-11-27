"""Fictional candidate/source fixtures, not an import or release tool for real data.

Run from the repository root: python -m examples.conversation_renderer.fixture NEW_DIR
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lm_trainer.conversation import SOURCE_SCHEMA, WINDOW_SCHEMA
from lm_trainer.train.schema import CONVERSATION_RELEASE_SCHEMA, GROUP_SCHEMA

TARGET = "support_agent"
AGENT = ("acct-agent", TARGET)
CUSTOMER = ("acct-customer", "other")
TECH = ("acct-tech", "other")
OTHER = ("acct-other-support", "other_support")
PERMISSIONS = {"local_review": True, "train": True, "external": False}
AT = datetime(2026, 1, 1, 9, 2, tzinfo=UTC)


def event(conv, position, author, second, text=None, **extra):
    return {
        "source_id": f"{conv}-m{position:02d}",
        "position": position,
        "speaker": author[0],
        "person": author[1],
        "display_name": "同名班员" if author in (CUSTOMER, TECH) else "虚构客服",
        "timestamp": (AT + timedelta(seconds=second)).isoformat(),
        "kind": "text",
        "readable": text is not None,
        "body_available": text is not None,
        "selection": "text" if text is not None else "structure",
        "selection_basis": "hidden"
        if extra.get("selection") == "hidden"
        else "witness"
        if "witness_source_ids" in extra
        else "online",
        **({"content": text} if text is not None else {}),
        **(
            {"available_for_training": True}
            if author == AGENT and text is not None
            else {}
        ),
        **extra,
    }


def edge(owner, name, kind, resolution="resolved", **extra):
    return {
        "owner_id": owner,
        "id": name,
        "kind": kind,
        "resolution": resolution,
        **extra,
    }


def conv(name, mode, rows, relations=()):
    return {
        "id": name,
        "event_group": f"fiction:{name}",
        "mode": mode,
        "events": list(rows),
        "relations": list(relations),
    }


def window(conversation, name, targets, *, positions=None, merges=(), **extra):
    events = [
        e
        for e in conversation["events"]
        if positions is None or e["position"] in positions
    ]
    targets = {e["source_id"] for e in events if e["position"] in targets}
    messages, timeline, blocks, edges = [], [], [], []
    for e in events:
        sid, selected = e["source_id"], e["source_id"] in targets
        text = e["selection"] == "text"
        timeline.append(
            {
                "id": sid,
                "text_slot": text,
                "target": selected,
                "event": not text or e["kind"] != "text",
                "body_available": e["body_available"],
                "structure_selected": not text,
            }
        )
        if text:
            messages.append(
                {
                    k: e[k]
                    for k in (
                        "source_id",
                        "position",
                        "speaker",
                        "person",
                        "timestamp",
                        "content",
                    )
                }
                | {"supervised": selected}
            )
            edges.extend(
                [sid, edge["id"]]
                for edge in conversation["relations"]
                if edge["owner_id"] == sid
            )
            if e["position"] in merges:
                blocks[-1]["ids"].append(sid)
            else:
                blocks.append(
                    {
                        "ids": [sid],
                        "speaker": e["speaker"],
                        "person": e["person"],
                        "target": selected,
                    }
                )
    return {
        "id": name,
        "schema": WINDOW_SCHEMA,
        "target_person": TARGET,
        "conversation": conversation["id"],
        "platform": "fixture_im",
        "view": "dialogue_continuous",
        "disposition": "adapter_candidate",
        "training_authorized": False,
        "split": "unassigned",
        "messages": messages,
        "timeline": timeline,
        "generation_blocks": blocks,
        "target_mask": [m["supervised"] for m in messages],
        "target_ids": [m["source_id"] for m in messages if m["supervised"]],
        "target_causal_prefix_lengths": {
            slot["id"]: i for i, slot in enumerate(timeline) if slot["target"]
        },
        "history_relation_refs": edges,
        "target_relation_audit_only": [pair for pair in edges if pair[0] in targets],
        **extra,
    }


def scenario_data():
    direct = conv(
        "direct",
        "direct",
        (
            event("direct", 1, CUSTOMER, 0, "设备状态灯闪烁"),
            event("direct", 2, AGENT, 2, "确认"),
            event(
                "direct",
                3,
                AGENT,
                5,
                "确认\n\n```text\ncheck lamp\n```\n记录状态🙂后回复",
            ),
        ),
    )
    # Joint is the field projection from the reference Markdown, with independent
    # synthetic canonical source/edge facts. The companion ZIP is not required.
    joint = conv(
        "joint",
        "group",
        (
            event("joint", 1, CUSTOMER, 0, "网关重启后仍然报错"),
            event("joint", 2, AGENT, 2, "请先确认电源指示灯"),
            event("joint", 3, TECH, 5, "指示灯正常，日志也能导出"),
            event("joint", 4, AGENT, 8, "那先不要恢复出厂设置"),
        ),
        (
            edge("joint-m02", "edge-003", "reply", anchor_id="joint-m01"),
            edge("joint-m04", "edge-004", "reply", anchor_id="joint-m03"),
        ),
    )
    group = conv(
        "group",
        "group",
        (
            event("group", 1, CUSTOMER, 0, "重复内容"),
            event("group", 2, AGENT, 2, "旧客服观察"),
            event("group", 3, TECH, 5, "重复内容"),
            event("group", 4, OTHER, 8, "其他客服也已核对"),
            event("group", 5, AGENT, 9, "重复内容"),
        ),
        (edge("group-m05", "own", "reply", anchor_id="group-m03"),),
    )
    media = conv(
        "media",
        "group",
        (
            event("media", 1, CUSTOMER, 0, "请核对工位指示"),
            event(
                "media",
                2,
                TECH,
                1,
                None,
                kind="file",
                body_available=True,
                readable=True,
            ),
            event("media", 3, AGENT, 2, "记录第一个状态"),
            event("media", 4, AGENT, 3, None, kind="image"),
            event("media", 5, TECH, 4, None, kind="unknown"),
            event("media", 6, AGENT, 5, "记录下一个状态"),
        ),
    )
    anchor = conv(
        "anchor",
        "group",
        (
            event(
                "anchor",
                1,
                TECH,
                -7200,
                "旧锚点只在后续可见",
                witness_source_ids=["anchor-m05"],
                visible_from_position=6,
            ),
            event("anchor", 2, CUSTOMER, 0, "今早新观察"),
            event("anchor", 3, OTHER, 1, None, kind="file", selection="hidden"),
            event("anchor", 4, AGENT, 2, "记录现象"),
            event("anchor", 5, CUSTOMER, 5, "补充旧记录和@同名班员"),
            event("anchor", 6, AGENT, 6, "对比两次结果"),
        ),
        (
            edge("anchor-m05", "old", "quote", anchor_id="anchor-m01"),
            edge("anchor-m05", "visible-mention", "mention", speaker="acct-agent"),
            edge("anchor-m05", "loose-at", "mention", "display_handle_only"),
            edge("anchor-m05", "unknown-quote", "quote", "unknown"),
            edge("anchor-m05", "protected-quote", "quote", "protected"),
            edge("anchor-m05", "unreadable-quote", "quote", anchor_id="anchor-m03"),
        ),
    )
    opening = conv(
        "opening", "direct", (event("opening", 1, AGENT, 0, "当前值班开始"),)
    )
    continuation = conv(
        "continuation",
        "direct",
        (
            event("continuation", 1, CUSTOMER, 0, "设备状态已登记"),
            event("continuation", 2, AGENT, 2, "前一个回复"),
            event("continuation", 3, AGENT, 30, "无观察的新客服气泡"),
        ),
    )
    catalog = {
        "schema": SOURCE_SCHEMA,
        "fixture_only": True,
        "conversations": [direct, joint, group, media, anchor, opening, continuation],
    }
    rows = [
        window(direct, "win-direct", (2, 3), merges=(3,)),
        window(joint, "win-joint-main", (2, 4)),
        window(
            group, "win-group", (5,), scene={"window_speakers": ["FUTURE_ROSTER_NONCE"]}
        ),
        window(media, "win-media", (3, 6)),
        window(anchor, "win-anchor", (4, 6), positions=(1, 2, 4, 5, 6)),
        window(joint, "win-joint-primary-a", (2,), positions=(1, 2)),
        window(joint, "win-joint-primary-b", (4,)),
        window(opening, "win-opening", (1,), view="independent_opening"),
        window(continuation, "win-continuation", (3,), view="author_continuation"),
        window(group, "win-review", (5,), disposition="review_only"),
    ]
    routes = [{"window_id": row["id"], "default": i < 5} for i, row in enumerate(rows)]
    return catalog, rows, routes


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return file_ref(path)


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return file_ref(path)


def file_ref(path):
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_bundle(destination):
    """Only authorizes these hard-coded fictional fixtures, into a new directory."""
    destination.mkdir(parents=True, exist_ok=False)
    catalog, rows, routes = scenario_data()
    write_json(destination / "candidate-sources.json", catalog)
    write_rows(destination / "candidates.jsonl", rows)
    write_rows(destination / "candidate-routing.jsonl", routes)
    defaults = [r for r in routes if r["default"]]
    selected = {
        row["conversation"]
        for row, route in zip(rows, routes, strict=True)
        if route["default"]
    }
    source = write_json(
        destination / "source_fixture.json",
        {
            **catalog,
            "split": "train",
            "conversations": [
                conv for conv in catalog["conversations"] if conv["id"] in selected
            ],
        },
    )
    groups, records = [], []
    for row, route in zip(rows, routes):
        if not route["default"]:
            continue
        assignment = {
            "event_group": f"fiction:{row['conversation']}",
            "kind": "sft",
            "split": "train",
        }
        records.append(
            {
                **row,
                **assignment,
                "scope": "fixture",
                "fixture_only": True,
                "review_status": "approved",
                "candidate_status": "approved",
                "permissions": PERMISSIONS,
                "source_index_ref": source["path"],
            }
        )
        groups.append(
            {
                "id": row["id"],
                "schema": GROUP_SCHEMA,
                **assignment,
                "target_person": TARGET,
                "source_ids": [slot["id"] for slot in row["timeline"]],
            }
        )
    manifest = {
        "schema": CONVERSATION_RELEASE_SCHEMA,
        "scope": "fixture",
        "fixture_only": True,
        "training_authorized": False,
        "review_status": "approved",
        "release_id": "renderer-fixture",
        "parent_version": "none",
        "target_person": TARGET,
        "partition_splits": ["train"],
        "source_group_manifest": write_rows(destination / "groups.jsonl", groups),
        "conversation_source": source,
        "routing": write_rows(destination / "routing.jsonl", defaults),
        "files": [
            {
                **write_rows(destination / "sft.jsonl", records),
                "kind": "sft",
                "source": "fictional",
                "split": "train",
                "records": len(records),
            }
        ],
    }
    path = destination / "manifest.json"
    write_json(path, manifest)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(write_bundle(args.destination))


if __name__ == "__main__":
    main()
