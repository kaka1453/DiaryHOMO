from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from diary_core.config.common import dump_runtime_config, resolve_path


DEFAULT_OUTPUT_ROOT = "generate/output"
DEFAULT_OUTPUT_NAME = "boa256日记"
OUTPUT_MD_FILENAME = "output.md"
PARAMETERS_FILENAME = "parameters.json"


def normalize_output_name(value: str | None) -> str:
    name = str(value or DEFAULT_OUTPUT_NAME).strip()
    if not name:
        name = DEFAULT_OUTPUT_NAME
    return "".join("_" if char in {"/", "\\"} else char for char in name)


def resolve_output_root(value: str | Path | None) -> Path:
    return resolve_path(value or DEFAULT_OUTPUT_ROOT)


def create_run_dir(output_root: str | Path | None, output_name: str | None) -> Path:
    parent = resolve_output_root(output_root) / normalize_output_name(output_name)
    parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for _ in range(20):
        run_dir = parent / f"{timestamp}_{uuid4().hex[:6]}"
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"无法创建唯一输出目录: {parent}")


def prepare_output_bundle(runtime: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(runtime["output_run_dir"]) if runtime.get("output_run_dir") else create_run_dir(
        runtime.get("output_root"),
        runtime.get("output_name"),
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    runtime["output_run_dir"] = str(run_dir)
    runtime["output_file"] = str(run_dir / OUTPUT_MD_FILENAME)
    runtime["parameters_file"] = str(run_dir / PARAMETERS_FILENAME)
    return runtime


def write_parameters(runtime: dict[str, Any], parameters_file: str | Path | None = None) -> Path:
    path = Path(parameters_file or runtime["parameters_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "runtime": json.loads(dump_runtime_config(runtime)),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
