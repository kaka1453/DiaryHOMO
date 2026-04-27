from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Config common helper smoke test")
    parser.add_argument("value", nargs="?", default="true")
    args = parser.parse_args()
    print(json.dumps({"project_root": str(PROJECT_ROOT), "bool": str2bool(args.value)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
