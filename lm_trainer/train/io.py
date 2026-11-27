from __future__ import annotations

import ctypes
import os
from pathlib import Path

from lm_trainer.config.common import dump_runtime_config


def validate_paths(runtime: dict, *, require_model: bool = True) -> None:
    model_path = Path(runtime["model_name_or_path"])
    data_field = (
        "data_file_path"
        if runtime.get("data_mode", "legacy") == "legacy"
        else "manifest_path"
    )
    data_path = Path(runtime[data_field])
    checkpoint_dir = Path(runtime["checkpoint_dir"])

    if require_model and not model_path.is_dir():
        raise FileNotFoundError("Local model directory is missing.")
    if not data_path.is_file() or (
        data_field == "manifest_path" and data_path.is_symlink()
    ):
        raise FileNotFoundError(f"Missing or unsafe {data_field}.")
    if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
        raise ValueError("checkpoint_dir must be a directory.")


def save_runtime_snapshot(runtime: dict) -> Path:
    checkpoint_dir = Path(runtime["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = checkpoint_dir / "resolved_train_runtime.json"
    with snapshot_path.open("w", encoding="utf-8") as fh:
        fh.write(dump_runtime_config(runtime))
    return snapshot_path


def publish_directory(staging: Path, directory: Path) -> None:
    """Linux atomic directory rename, refusing even a concurrently created empty destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    at_fdcwd, rename_noreplace = -100, 1
    if rename(
        at_fdcwd,
        os.fsencode(staging),
        at_fdcwd,
        os.fsencode(directory),
        rename_noreplace,
    ):
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno), str(directory))
