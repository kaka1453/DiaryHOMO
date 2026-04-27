from __future__ import annotations

import json
from pathlib import Path

from diary_core.config.common import dump_runtime_config
from diary_core.config.train_config import build_train_parser, build_train_runtime_config


def validate_paths(runtime: dict) -> None:
    model_path = Path(runtime["model_name_or_path"])
    data_path = Path(runtime["data_file_path"])
    checkpoint_dir = Path(runtime["checkpoint_dir"])

    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {data_path}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


def save_runtime_snapshot(runtime: dict) -> Path:
    checkpoint_dir = Path(runtime["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = checkpoint_dir / "resolved_train_runtime.json"
    with snapshot_path.open("w", encoding="utf-8") as fh:
        fh.write(dump_runtime_config(runtime))
    return snapshot_path


def main() -> None:
    parser = build_train_parser()
    parser.add_argument("--check-paths", action="store_true")
    args = parser.parse_args()
    runtime = build_train_runtime_config(args)
    if args.check_paths:
        validate_paths(runtime)

    summary = {
        "model_path": runtime["model_name_or_path"],
        "model_exists": Path(runtime["model_name_or_path"]).exists(),
        "data_path": runtime["data_file_path"],
        "data_exists": Path(runtime["data_file_path"]).exists(),
        "checkpoint_dir": runtime["checkpoint_dir"],
        "checked": args.check_paths,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
