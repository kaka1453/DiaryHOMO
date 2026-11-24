from __future__ import annotations

from pathlib import Path

from lm_trainer.config.common import dump_runtime_config


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
