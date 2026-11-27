"""Build an isolated release-shaped bundle from fictional approved fixtures only."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from lm_trainer.train.schema import (
    CORPUS_SCHEMA,
    GROUP_SCHEMA,
    INDEX_SCHEMA,
    RELEASE_SCHEMA,
)

HERE = Path(__file__).resolve().parent


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_bundle(destination, *, release=False, negative=None):
    destination.mkdir(parents=True, exist_ok=False)
    dialog = json.loads((HERE / "sft_approved_test.jsonl").read_text(encoding="utf-8"))
    dialog.update(schema=CORPUS_SCHEMA, source_index_ref="synthetic:approved-dialog")
    shared = {
        k: dialog[k]
        for k in ("schema", "target_person", "permissions", "review_status", "split")
    }
    cpt = [
        {
            **shared,
            "kind": "cpt",
            "id": "FIX-CHAT-1",
            "event_group": dialog["event_group"],
            "source_index_ref": "synthetic:chat-1",
            "source_ids": dialog["messages"][4]["source_ids"],
            "text": dialog["messages"][4]["content"],
        },
        {
            **shared,
            "kind": "cpt",
            "id": "FIX-CHAT-2",
            "event_group": "demo:vision:lights",
            "source_index_ref": "synthetic:chat-2",
            "source_ids": ["fixture_chat_2"],
            "text": "先用同一批样件比对。\n- 保存 `histogram` 与光源设置 🙂",
        },
    ]
    notes = [
        {
            **shared,
            "kind": "cpt",
            "id": "FIX-NOTES-1",
            "event_group": "demo:vision:notes",
            "source_index_ref": "synthetic:notes-1",
            "source_ids": ["fixture_notes_1"],
            "text": "排查笔记：先记录样件批次与各工位的曝光设置。\n\n换班后复查光源角度，分别记录触发计数和存图数量。",
        }
    ]
    if negative == "permission":
        notes[0]["permissions"] = {**shared["permissions"], "train": False}
    if negative == "pending":
        notes[0]["review_status"] = "pending_owner"
    if negative == "owner":
        notes[0]["review_status"] = "owner_review_only"
    if negative == "multi":
        dialog["messages"][3]["speaker"] = "FIXTURE_ENGINEER"

    def ledger_row(row):
        ids = (
            row["source_ids"]
            if row["kind"] == "cpt"
            else sorted({sid for m in row["messages"] for sid in m["source_ids"]})
        )
        return {
            **{
                k: row[k]
                for k in ("id", "event_group", "kind", "target_person", "split")
            },
            "schema": GROUP_SCHEMA,
            "source_ids": ids,
        }

    groups = [ledger_row(row) for row in (*cpt, *notes, dialog)]
    if negative == "closure":
        groups.append(
            {
                **copy.deepcopy(groups[-1]),
                "id": "FIX-BAD-VIEW",
                "target_person": "CUSTOMER",
                "split": "dev",
            }
        )
    selected = [
        sid
        for m, yes in zip(dialog["messages"], dialog["target_mask"])
        if yes
        for sid in m["source_ids"]
    ]
    index = [
        {
            "schema": INDEX_SCHEMA,
            "source_id": sid,
            "disposition": "F",
            "review_status": "approved",
        }
        for sid in selected
    ]
    specs = []
    for source, kind, rows in (
        ("chat", "cpt", cpt),
        ("notes", "cpt", notes),
        ("dyadic", "sft", [dialog]),
    ):
        specs.append(
            {
                **write_jsonl(destination / f"{source}.jsonl", rows),
                "source": source,
                "kind": kind,
                "split": "train",
                "records": len(rows),
            }
        )
    manifest = {
        "schema": RELEASE_SCHEMA,
        "release_id": "fictional_vision_support_v1",
        "target_person": dialog["target_person"],
        "parent_version": "fictional_vision_parent_v0",
        "scope": "release" if release else "fixture",
        "fixture_only": not release,
        "training_authorized": release,
        "review_status": "approved",
        "approval_ref": "fictional:test-only-never-a-production-approval"
        if release
        else None,
        "source_group_manifest": write_jsonl(destination / "groups.jsonl", groups),
        "source_index": write_jsonl(destination / "index.jsonl", index),
        "files": specs,
    }
    path = destination / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if negative == "hash":
        with (destination / "chat.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination", type=Path, help="A fresh directory; never overwrites"
    )
    parser.add_argument(
        "--mock-release",
        action="store_true",
        help="Fictional release gate unit test only",
    )
    parser.add_argument(
        "--negative",
        choices=("permission", "pending", "owner", "multi", "closure", "hash"),
    )
    args = parser.parse_args()
    write_bundle(args.destination, release=args.mock_release, negative=args.negative)


if __name__ == "__main__":
    main()
