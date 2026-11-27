"""Prepare only the known fictional Phase 2 B example, never approve external rows."""

import argparse
import json
import re
import shutil
from pathlib import Path

from examples.conversation_renderer.fixture import (
    AGENT,
    CUSTOMER,
    PERMISSIONS,
    TARGET,
    conv,
    event,
    scenario_data,
    write_bundle,
    write_json,
    write_rows,
)
from examples.phase2_task_b.eval_fixture import write_cases
from lm_trainer.conversation import author_blocks, inspect_candidate, parse_catalog
from lm_trainer.train.schema import CORPUS_SCHEMA, GROUP_SCHEMA, RELEASE_SCHEMA

HERE = Path(__file__).resolve().parent


def checked_projection(markdown):
    blocks = re.findall(
        r"^```json[^\S\n]*\n(.*?)^```",
        markdown.read_text(encoding="utf-8"),
        re.MULTILINE | re.DOTALL,
    )
    projections = [json.loads(block) for block in blocks]
    (joint,) = [
        obj
        for obj in projections
        if isinstance(obj, dict) and obj.get("id") == "win-joint-main"
    ]
    catalog, candidates, _ = scenario_data()
    expected = next(row for row in candidates if row["id"] == joint["id"])
    # Only two derived, audit-only timeline fields are omitted from the checked
    # comparison. Nothing from Markdown can override any authorization field.
    projection = {
        **joint,
        "timeline": [
            {k: v for k, v in slot.items() if k not in {"gap_seconds", "gap_class"}}
            for slot in joint["timeline"]
        ],
    }
    if projection != {k: v for k, v in expected.items() if k != "split"}:
        raise ValueError(
            "Markdown is not the known fictional win-joint-main projection"
        )
    candidate = {**projection, "split": "unassigned"}
    inspect_candidate(candidate, parse_catalog(catalog))
    return candidate


def _read_rows(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def build_sft(path, joint, *, primary):
    bundle = write_bundle(path)
    manifest = json.loads(bundle.read_text(encoding="utf-8"))
    manifest["release_id"] = "phase2-b-sft-primary" if primary else "phase2-b-sft-joint"
    manifest["parent_version"] = "phase2-b-sft-fixture-root"
    raw = _read_rows(path / "sft.jsonl")
    ledger = _read_rows(path / "groups.jsonl")
    routes = _read_rows(path / "routing.jsonl")
    rows, groups = {}, []
    for row, allocation in zip(raw, ledger, strict=True):
        if row["id"] != joint["id"]:
            source = (
                "single_person"
                if row["conversation"] == "direct"
                else "group_continuous"
            )
            rows.setdefault(source, []).append(row)
            groups.append(allocation)
            continue
        if not primary:
            rows.setdefault("group_joint", []).append(
                {**row, **joint, "split": "train"}
            )
            groups.append(allocation)
            continue
        routes = [route for route in routes if route["window_id"] != joint["id"]]
        # Re-materialize the alternative route in isolation. It replaces the
        # joint row; both route releases cover the exact same target source set.
        for alt in _read_rows(path / "candidates.jsonl"):
            if not alt["id"].startswith("win-joint-primary-"):
                continue
            rows.setdefault("group_primary", []).append(
                {**row, **alt, "split": "train"}
            )
            groups.append(
                {
                    **allocation,
                    "id": alt["id"],
                    "source_ids": [slot["id"] for slot in alt["timeline"]],
                }
            )
            routes.append({"window_id": alt["id"], "default": True})
    manifest["files"] = [
        {
            **write_rows(path / f"{source}.jsonl", records),
            "kind": "sft",
            "source": source,
            "split": "train",
            "records": len(records),
        }
        for source, records in sorted(rows.items())
    ]
    manifest["source_group_manifest"] = write_rows(path / "groups.jsonl", groups)
    manifest["routing"] = write_rows(path / "routing.jsonl", routes)
    write_json(bundle, manifest)
    return bundle


def build_cpt(path):
    path.mkdir()
    # A distinct, invented history: do not turn input-only or held-out content
    # into CPT text. Source provenance stays separate from the text bodies.
    history = conv(
        "cpt-history",
        "group",
        (
            event("cpt-history", 1, AGENT, 0, "先记录离线时间。"),
            event("cpt-history", 2, AGENT, 2, "再核对备份指示灯。"),
            event("cpt-history", 3, CUSTOMER, 4, "这是第三方旁注，不进 CPT。"),
            event("cpt-history", 4, AGENT, 6, "请保留故障日志。"),
            event("cpt-history", 5, CUSTOMER, 8, kind="image"),
            event("cpt-history", 6, AGENT, 10, "复测前记下控制器版本。"),
        ),
    )
    catalog, _, _ = scenario_data()
    parsed = parse_catalog({**catalog, "conversations": [history]})
    blocks = author_blocks(
        parsed, history["id"], TARGET, [row["source_id"] for row in history["events"]]
    )
    chat = [
        (
            "\n".join(e.content for e in block),
            [e.source_id for e in block],
            history["event_group"],
        )
        for block in blocks
        if all(e.available_for_training for e in block)
    ]
    notes = [
        (
            "值班记录：检查样件编号，再对照公开测试基线。",
            ["phase2-b-note-1"],
            "fiction:cpt-note-1",
        ),
        (
            "记录测量条件；无证据不要归因于电源。",
            ["phase2-b-note-2"],
            "fiction:cpt-note-2",
        ),
    ]
    files, groups = [], []
    for source, items in (("support_history", chat), ("shift_notes", notes)):
        rows = []
        for index, (text, ids, event_group) in enumerate(items):
            common = {
                "id": f"cpt-{source}-{index}",
                "kind": "cpt",
                "target_person": TARGET,
                "event_group": event_group,
                "split": "train",
                "source_ids": ids,
            }
            rows.append(
                {
                    **common,
                    "schema": CORPUS_SCHEMA,
                    "source_index_ref": "fixture:phase2-b-cpt",
                    "permissions": PERMISSIONS,
                    "review_status": "approved",
                    "fixture_only": True,
                    "text": text,
                }
            )
            groups.append({**common, "schema": GROUP_SCHEMA})
        files.append(
            {
                **write_rows(path / f"{source}.jsonl", rows),
                "kind": "cpt",
                "source": source,
                "split": "train",
                "records": len(rows),
            }
        )
    write_json(
        path / "manifest.json",
        {
            "schema": RELEASE_SCHEMA,
            "release_id": "phase2-b-cpt",
            "target_person": TARGET,
            "parent_version": "none",
            "files": files,
            "source_group_manifest": write_rows(path / "groups.jsonl", groups),
            "scope": "fixture",
            "fixture_only": True,
            "training_authorized": False,
            "review_status": "approved",
        },
    )


def prepare(destination, markdown, *, with_tiny_base=False):
    joint = checked_projection(markdown)
    destination.mkdir(parents=True, exist_ok=False)
    build_sft(destination / "joint", joint, primary=False)
    build_sft(destination / "primary", joint, primary=True)
    build_cpt(destination / "cpt")
    write_cases(destination / "eval-dev.json")
    for name in ("model.yaml", "direct_sft.yaml", "mixed_cpt_sft.yaml", "roles.json"):
        shutil.copyfile(HERE / name, destination / name)
    import yaml

    for name in ("direct_sft", "mixed_cpt_sft"):
        # Real Qwen needs its native thinking flag. The synthetic Qwen2 template
        # intentionally has no such protocol and uses the unchanged fixture YAML.
        settings = yaml.safe_load(
            (destination / f"{name}.yaml").read_text(encoding="utf-8")
        )
        settings["template_policy"]["enable_thinking"] = False
        write_json(destination / f"qwen_{name}.yaml", settings)
    if with_tiny_base:
        from examples.vision_support.task_b_workload import tiny_base

        tiny_base(destination / "base")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument(
        "--with-tiny-base",
        action="store_true",
        help="Generate a random, synthetic CPU-only base",
    )
    args = parser.parse_args()
    prepare(args.destination, args.markdown, with_tiny_base=args.with_tiny_base)
    print(args.destination)


if __name__ == "__main__":
    main()
