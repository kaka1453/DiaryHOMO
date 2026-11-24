from pathlib import Path
import argparse
import json

from diary_core.config.assets import product_root

from lm_trainer.config.common import (
    asset_path,
    dump_runtime_config,
    load_yaml_config,
    require_keys,
    str2bool,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, *, asset_root: str | Path, directory=False, required=True) -> Path:
    root = product_root(str(asset_root))
    return asset_path(root, str(value), "asset path", "config", "data", "assets", "checkpoints", "runs",
                      directory=directory, required=required)


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
