from pathlib import Path
import argparse
import json

from lm_trainer.config.common import (
    dump_runtime_config,
    load_yaml_config,
    require_keys,
    str2bool,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Config common helper smoke test")
    parser.add_argument("value", nargs="?", default="true")
    args = parser.parse_args()
    print(
        json.dumps(
            {"project_root": str(PROJECT_ROOT), "bool": str2bool(args.value)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
