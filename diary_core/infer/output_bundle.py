from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from diary_core.config.assets import check_output_tree, runtime_output
from diary_core.config.common import dump_runtime_config


DEFAULT_OUTPUT_NAME = "diary"
OUTPUT_MD_FILENAME = "output.md"
PARAMETERS_FILENAME = "parameters.json"


def normalize_output_name(value: str | None) -> str:
    name = str(value or DEFAULT_OUTPUT_NAME).strip()
    if not name:
        name = DEFAULT_OUTPUT_NAME
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("output_name must be a single directory name")
    return name


def create_run_dir(parent: Path) -> Path:
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
    root = Path(runtime["asset_root"]) if runtime.get("asset_root") else None
    output_root = Path(runtime.get("output_root") or runtime["output_run_dir"])
    if not output_root.is_absolute():
        raise ValueError("output_root must be resolved before creating output")
    run_dir = runtime.get("output_run_dir")
    if run_dir:
        run_dir = Path(run_dir)
        if root is not None:
            run_dir = runtime_output(root, run_dir, directory=True)
            check_output_tree(root, run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        parent = output_root / normalize_output_name(runtime.get("output_name"))
        if root is not None:
            parent = runtime_output(root, parent, directory=True)
        run_dir = create_run_dir(parent)

    explicit_output = runtime.get("requested_output_file")
    output_file = Path(explicit_output) if explicit_output else run_dir / OUTPUT_MD_FILENAME
    if root is not None:
        output_file = runtime_output(root, output_file)
    if explicit_output:
        runtime["requested_output_file"] = str(output_file)
    runtime["output_run_dir"] = str(run_dir)
    runtime["output_file"] = str(output_file)
    runtime["parameters_file"] = str(run_dir / PARAMETERS_FILENAME)
    return runtime


def write_parameters(runtime: dict[str, Any], parameters_file: str | Path | None = None) -> Path:
    from diary_core.config.infer_config import inference_summary

    path = Path(parameters_file or runtime["parameters_file"])
    if runtime.get("asset_root"):
        path = runtime_output(Path(runtime["asset_root"]), path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "runtime": json.loads(dump_runtime_config(inference_summary(runtime))),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
