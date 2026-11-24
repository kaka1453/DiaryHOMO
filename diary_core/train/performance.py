"""单卡完整 optimizer update 的同步计时与可审计统计。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata() -> dict:
    from diary_core.config.common import PROJECT_ROOT

    packages = (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "datasets",
        "bitsandbytes",
        "unsloth",
        "unsloth-zoo",
        "triton",
        "xformers",
    )
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "packages": versions,
        "python": sys.version,
        "source_sha256": {
            name: sha256_file(PROJECT_ROOT / name)
            for name in (
                "diary_core/train/engine.py",
                "diary_core/train/modeling.py",
                "diary_core/train/bootstrap.py",
                "diary_core/train/performance.py",
                "diary_core/train/data.py",
                "diary_core/config/train_config.py",
            )
        },
        "lock_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in (
                PROJECT_ROOT / "uv.lock",
                PROJECT_ROOT / "envs/unsloth/uv.lock",
            )
            if path.exists()
        },
    }


def summarize_updates(updates: list[dict], warmup_steps: int) -> dict:
    measured = updates[warmup_steps:]
    if not measured:
        return {
            "measured_updates": 0,
            "seconds_per_update": None,
            "samples_per_second": None,
            "input_tokens_per_second": None,
            "loss_tokens_per_second": None,
        }
    seconds = sum(row["seconds"] for row in measured)
    result = {
        "measured_updates": len(measured),
        "seconds_per_update": seconds / len(measured),
    }
    for key in ("samples", "input_tokens", "loss_tokens"):
        result[f"{key}_per_second"] = sum(row[key] for row in measured) / seconds
    return result


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
