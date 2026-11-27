"""Anonymous blind review: packaging, review records and reveal summaries.

The blind pack contains **no model identity**. The stable left/right mapping is
kept in a separate private file (``blind-map.json``) that the review UI never
reads. Side assignment is derived from ``sha256(seed || case_id || event_group)``
so the same seed always reproduces the same layout.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lm_trainer.eval.schema import EvalError, canonical_json, sha256_text

BLIND_SCHEMA = "lm_trainer.eval-blind.v1"
MAP_SCHEMA = "lm_trainer.eval-blind-map.v1"
REVIEW_SCHEMA = "lm_trainer.eval-review.v1"

DIMENSIONS = (
    "technical_accuracy",
    "expression_clarity",
    "multi_turn_consistency",
    "evidence_use",
    "instruction_following",
    "engineering_isolation",
)

DIMENSION_LABELS = {
    "technical_accuracy": "技术准确性与判断力",
    "expression_clarity": "表达自然度与沟通节奏",
    "multi_turn_consistency": "多轮一致性",
    "evidence_use": "外部证据/卡片使用",
    "instruction_following": "基础能力（指令遵循/终止/无复读）",
    "engineering_isolation": "工程与隔离红线",
}

CHOICES = ("left", "right", "tie", "both_fail", "skip")
CHOICE_LABELS = {
    "left": "左优",
    "right": "右优",
    "tie": "平局",
    "both_fail": "均不合格",
    "skip": "跳过",
}

FORBIDDEN_IN_BLIND = ("model_id", "model", "stage", "vram", "adapter", "base")


@dataclass(frozen=True, slots=True)
class BlindItem:
    item_id: str
    case_id: str
    event_group: str
    split: str
    memory_condition: str
    kind: str
    prompt: str = field(repr=False)
    left: str = field(repr=False)
    right: str = field(repr=False)
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class BlindPack:
    pack_id: str
    seed: int
    created_at: str
    items: list[BlindItem]
    duplicate_seed_note: str
    mode: str

    def to_dict(self) -> dict:
        return {
            "schema": BLIND_SCHEMA,
            "fixture_only": True,
            "pack_id": self.pack_id,
            "seed": self.seed,
            "created_at": self.created_at,
            "mode": self.mode,
            "duplicate_seed_note": self.duplicate_seed_note,
            "item_count": len(self.items),
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(slots=True)
class ReviewRecord:
    item_id: str
    choice: str
    scores: dict[str, int]
    reason: str
    reviewer: str
    submitted_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def _stable_position(seed: int, case_id: str, event_group: str) -> str:
    """Return the model_id that should appear on the left side."""

    digest = hashlib.sha256(
        canonical_json({"seed": seed, "case_id": case_id, "event_group": event_group}).encode()
    ).hexdigest()
    return "left" if int(digest, 16) % 2 == 0 else "right"


def _pair_case(records: list[dict]) -> list[tuple[dict, dict]]:
    """Group equal-turn records by (case, condition, turn) and return A/B pairs."""

    groups: dict[tuple, list[dict]] = {}
    for record in records:
        if record.get("kind") != "fixed_history":
            continue
        key = (
            record["case_id"],
            record.get("memory_condition", "M0"),
            record.get("turn_index", 0),
        )
        groups.setdefault(key, []).append(record)
    pairs = []
    for key in sorted(groups):
        group = groups[key]
        if len(group) != 2:
            continue
        if len({r["model_id"] for r in group}) != 2:
            continue
        a, b = sorted(group, key=lambda r: r["model_id"])
        pairs.append((a, b))
    return pairs


def make_blind_pack(
    records: list[dict],
    *,
    seed: int,
    prompts: dict[str, str] | None = None,
    pack_id: str | None = None,
    created_at: str = "",
    mode: str = "mock",
) -> tuple[BlindPack, dict]:
    """Build a blind pack and its private mapping from evaluation records.

    ``records`` is the ``records`` list of a runner result. Only
    ``fixed_history`` units are paired: rollout branches are not comparable
    turn-by-turn and are reported separately.
    """

    pairs = _pair_case(records)
    if not pairs:
        raise EvalError("BLIND_NO_PAIR", "no two-model fixed-history pairs found")
    if len({p[0]["model_id"] for p in pairs}) != 1 and len(pairs) > 1:
        pass
    items: list[BlindItem] = []
    mapping: dict[str, dict] = {}
    seen: set[str] = set()
    for left, right in pairs:
        left_pos = _stable_position(seed, left["case_id"], left.get("event_group", ""))
        if left_pos == "right":
            left, right = right, left
        item_id = "item-" + sha256_text(
            canonical_json(
                {
                    "seed": seed,
                    "case_id": left["case_id"],
                    "condition": left.get("memory_condition", "M0"),
                    "turn": left.get("turn_index", 0),
                    "left": left["model_id"],
                    "right": right["model_id"],
                }
            )
        )[:16]
        if item_id in seen:
            continue
        seen.add(item_id)
        prompt = (prompts or {}).get(left["case_id"], "")
        items.append(
            BlindItem(
                item_id=item_id,
                case_id=left["case_id"],
                event_group=left.get("event_group", ""),
                split=left.get("split", "dev"),
                memory_condition=left.get("memory_condition", "M0"),
                kind=left["kind"],
                prompt=prompt,
                left=left["output"],
                right=right["output"],
                seed=seed,
            )
        )
        mapping[item_id] = {
            "case_id": left["case_id"],
            "left": left["model_id"],
            "right": right["model_id"],
            "memory_condition": left.get("memory_condition", "M0"),
            "turn": left.get("turn_index", 0),
        }
    if not items:
        raise EvalError("BLIND_NO_PAIR", "no pairable units")
    pack = BlindPack(
        pack_id=pack_id or sha256_text(canonical_json([i.item_id for i in items]))[:16],
        seed=seed,
        created_at=created_at,
        items=items,
        duplicate_seed_note=(
            "同一 case 的多次采样不作为独立 case 统计；重复 seed 仅记录在来源运行中。"
        ),
        mode=mode,
    )
    private = {
        "schema": MAP_SCHEMA,
        "fixture_only": True,
        "pack_id": pack.pack_id,
        "seed": seed,
        "private": True,
        "mapping": mapping,
    }
    return pack, private


def save_blind_pack(
    pack: BlindPack,
    private_mapping: dict,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """Write the blind pack (shareable) and the private map (0600)."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pack_path = output_dir / "blind_pack.json"
    map_path = output_dir / "blind-map.json"
    _write_private(pack_path, pack.to_dict(), private=False)
    _write_private(map_path, private_mapping, private=True)
    return pack_path, map_path


def _write_private(path: Path, payload: dict, *, private: bool) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
    fd = os.open(path, flags, 0o600 if private else 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(data)
    if private:
        os.chmod(path, 0o600)


def load_blind_pack(path: Path | str) -> BlindPack:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvalError("BLIND_INVALID", f"cannot read {path.name}") from exc
    if payload.get("schema") != BLIND_SCHEMA:
        raise EvalError("BLIND_INVALID", "schema")
    if payload.get("fixture_only") is not True:
        raise EvalError("FIXTURE_ONLY_REQUIRED", "blind pack")
    items = [
        BlindItem(
            item["item_id"],
            item["case_id"],
            item["event_group"],
            item["split"],
            item["memory_condition"],
            item["kind"],
            item["prompt"],
            item["left"],
            item["right"],
            item["seed"],
        )
        for item in payload["items"]
    ]
    for item in items:
        text = f"{item.prompt}\n{item.left}\n{item.right}"
        for marker in FORBIDDEN_IN_BLIND:
            if marker in text:
                raise EvalError("BLIND_LEAK", f"blind item {item.item_id} contains {marker!r}")
    return BlindPack(
        payload["pack_id"],
        payload["seed"],
        payload["created_at"],
        items,
        payload.get("duplicate_seed_note", ""),
        payload.get("mode", "mock"),
    )


def _review_path(pack: BlindPack, review_dir: Path) -> Path:
    review_dir.mkdir(parents=True, exist_ok=True)
    return review_dir / f"reviews-{pack.pack_id}.jsonl"


def submit_review(
    pack: BlindPack,
    review_dir: Path | str,
    *,
    item_id: str,
    choice: str,
    scores: dict[str, int] | None = None,
    reason: str = "",
    reviewer: str = "anonymous",
    submitted_at: str = "",
) -> ReviewRecord:
    """Append one review to the private review ledger (one JSON per line)."""

    from datetime import UTC, datetime

    if choice not in CHOICES:
        raise EvalError("REVIEW_INVALID", f"choice {choice!r}")
    if not any(item.item_id == item_id for item in pack.items):
        raise EvalError("REVIEW_INVALID", "unknown item")
    scores = scores or {}
    for key, value in scores.items():
        if key not in DIMENSIONS:
            raise EvalError("REVIEW_INVALID", f"unknown dimension {key!r}")
        if type(value) is not int or not 1 <= value <= 5:
            raise EvalError("REVIEW_INVALID", f"score {key}={value!r}")
    record = ReviewRecord(
        item_id,
        choice,
        dict(scores),
        reason[:4000],
        reviewer[:64],
        submitted_at or datetime.now(UTC).isoformat(),
    )
    path = _review_path(pack, Path(review_dir))
    line = json.dumps(
        {"schema": REVIEW_SCHEMA, "fixture_only": True, **record.to_dict()},
        ensure_ascii=False,
    )
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)
    return record


def load_reviews(pack: BlindPack, review_dir: Path | str) -> list[ReviewRecord]:
    path = _review_path(pack, Path(review_dir))
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("schema") != REVIEW_SCHEMA:
            raise EvalError("REVIEW_INVALID", "schema")
        out.append(
            ReviewRecord(
                entry["item_id"],
                entry["choice"],
                dict(entry["scores"]),
                entry.get("reason", ""),
                entry.get("reviewer", ""),
                entry.get("submitted_at", ""),
            )
        )
    return out


@dataclass(slots=True)
class RevealSummary:
    pack_id: str
    seed: int
    item_count: int
    reviewed_count: int
    unreviewed_count: int
    independent_cases: int
    independent_events: int
    wins: dict[str, int]
    counts: dict[str, int]
    dimension_stats: dict[str, dict]
    missing: dict[str, Any]
    duplicate_seed_note: str
    mode: str
    per_item: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def reveal(
    pack: BlindPack,
    mapping: dict,
    reviews: list[ReviewRecord],
) -> RevealSummary:
    """Join reviews with the private mapping and compute reveal statistics."""

    if mapping.get("schema") != MAP_SCHEMA:
        raise EvalError("MAP_INVALID", "schema")
    if mapping.get("pack_id") != pack.pack_id:
        raise EvalError("MAP_INVALID", "pack_id mismatch")
    table = mapping.get("mapping", {})
    reviews_by_item: dict[str, ReviewRecord] = {}
    for review in reviews:
        if review.item_id in reviews_by_item:
            reviews_by_item[review.item_id] = review  # last review wins
        else:
            reviews_by_item[review.item_id] = review
    wins = {}
    counts = {choice: 0 for choice in CHOICES}
    dimension_values: dict[str, list[int]] = {name: [] for name in DIMENSIONS}
    missing = {name: 0 for name in DIMENSIONS}
    per_item = []
    reviewed_items = set()
    case_ids = set()
    event_ids = set()
    for item in pack.items:
        info = table.get(item.item_id)
        if info is None:
            raise EvalError("MAP_INVALID", f"missing mapping for {item.item_id}")
        review = reviews_by_item.get(item.item_id)
        record = {
            "item_id": item.item_id,
            "case_id": item.case_id,
            "event_group": item.event_group,
            "memory_condition": item.memory_condition,
            "left": info["left"],
            "right": info["right"],
        }
        if review is not None:
            counts[review.choice] += 1
            record["choice"] = review.choice
            record["reason"] = review.reason
            record["scores"] = review.scores
            if review.choice == "left":
                wins[info["left"]] = wins.get(info["left"], 0) + 1
                case_ids.add(item.case_id)
                event_ids.add(item.event_group)
            elif review.choice == "right":
                wins[info["right"]] = wins.get(info["right"], 0) + 1
                case_ids.add(item.case_id)
                event_ids.add(item.event_group)
            for name in DIMENSIONS:
                if name in review.scores:
                    dimension_values[name].append(review.scores[name])
                else:
                    missing[name] += 1
            reviewed_items.add(item.item_id)
        else:
            record["choice"] = None
        per_item.append(record)
    dimension_stats = {}
    for name in DIMENSIONS:
        values = dimension_values[name]
        dimension_stats[name] = {
            "count": len(values),
            "missing": missing[name],
            "mean": round(sum(values) / len(values), 3) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "distribution": {str(score): values.count(score) for score in range(1, 6)},
        }
    return RevealSummary(
        pack_id=pack.pack_id,
        seed=pack.seed,
        item_count=len(pack.items),
        reviewed_count=len(reviewed_items),
        unreviewed_count=len(pack.items) - len(reviewed_items),
        independent_cases=len(case_ids),
        independent_events=len(event_ids),
        wins=wins,
        counts=counts,
        dimension_stats=dimension_stats,
        missing=missing,
        duplicate_seed_note=pack.duplicate_seed_note,
        mode=pack.mode,
        per_item=per_item,
    )


def load_reveal_summary(path: Path | str) -> dict:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvalError("BLIND_INVALID", f"cannot read {path.name}") from exc
    if payload.get("schema") != "lm_trainer.eval-reveal.v1":
        raise EvalError("BLIND_INVALID", "schema")
    return payload


def assert_no_leak(payload: dict) -> None:
    """Fail if any value in a blind view contains a model identity hint."""

    def walk(node: Any):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"model_id", "model", "stage", "vram", "adapter", "base"}:
                    raise EvalError("BLIND_LEAK", f"key {key!r}")
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)


def save_reveal_summary(summary: RevealSummary, path: Path | str) -> Path:
    path = Path(path)
    payload = {
        "schema": "lm_trainer.eval-reveal.v1",
        "fixture_only": True,
        **summary.to_dict(),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
