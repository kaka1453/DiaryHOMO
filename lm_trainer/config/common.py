from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path
from typing import Any

import yaml

ASSET_DIRECTORIES = (
    "config",
    "data/train",
    "data/eval",
    "assets",
    "checkpoints",
    "runs",
)


def open_asset_root(value: str, *, create: bool = False) -> Path:
    path = Path(value).expanduser()
    if ".." in path.parts or path.is_symlink():
        raise ValueError(
            "--asset-root must be a directory path without '..' or a root symlink"
        )
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"--asset-root: {value!r} is missing or is not a directory")
    return path.resolve(strict=True)


def asset_path(
    root: Path,
    value: str,
    label: str,
    *directories: str,
    directory: bool = False,
    required: bool = True,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected a relative asset path")
    relative = Path(value)
    location = f"{label}: {value!r}"
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{location} must be relative and may not contain '..'")
    if not any(
        relative.is_relative_to(base) and (directory or relative != Path(base))
        for base in directories
    ):
        raise ValueError(f"{location} must be under {' or '.join(directories)}")
    path = root / relative
    current = root
    for component in relative.parts:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if required:
                raise FileNotFoundError(f"{location} is missing") from None
            break
        if stat.S_ISLNK(mode):
            raise ValueError(
                f"{location} crosses symlink {str(current.relative_to(root))!r}"
            )
        if current != path or directory:
            if not stat.S_ISDIR(mode):
                raise ValueError(
                    f"{location} requires a directory at {str(current.relative_to(root))!r}"
                )
        elif not stat.S_ISREG(mode):
            raise ValueError(f"{location} must be a regular file")
    return path


def init_asset_root(value: str) -> Path:
    root = open_asset_root(value, create=True)
    directories = [
        asset_path(root, path, "asset directory", path, directory=True, required=False)
        for path in ASSET_DIRECTORIES
    ]
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)
    return root


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def load_yaml_config(config_path: Path, label: str) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{label} 顶层必须是映射对象。")
    return data


def dump_runtime_config(runtime: dict[str, Any]) -> str:
    printable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in runtime.items()
    }
    return json.dumps(printable, ensure_ascii=False, indent=2)


def require_keys(runtime: dict[str, Any], keys: list[str], label: str) -> None:
    missing = [key for key in keys if runtime.get(key) is None]
    if missing:
        raise ValueError(f"{label} 配置缺少字段: {', '.join(missing)}")
