"""Run from an installed package or a clean source snapshot."""

import argparse
import os
import sys
from pathlib import Path

import yaml

from .config.common import dump_runtime_config, init_asset_root
from .config.train_config import build_train_parser, build_train_runtime_config
from .train.bootstrap import dispatch_unsloth


def main():
    argv = sys.argv[1:]
    if argv[:1] == ["init-assets"]:
        parser = argparse.ArgumentParser(
            prog="python -m lm_trainer init-assets",
            description="Create asset directories",
        )
        parser.add_argument("--asset-root", required=True)
        args = parser.parse_args(argv[1:])
        try:
            print(init_asset_root(args.asset_root))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        return
    parser = build_train_parser()
    args = parser.parse_args(argv)
    try:
        runtime = build_train_runtime_config(args)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        if args.asset_root is None:
            raise
        parser.error(str(exc))
    if args.dry_run:
        print(dump_runtime_config(runtime))
        return
    cache = Path(runtime["checkpoint_dir"]) / ".cache"
    for key, directory in (
        ("HF_HOME", "huggingface"),
        ("HF_DATASETS_CACHE", "datasets"),
        ("MPLCONFIGDIR", "matplotlib"),
    ):
        os.environ.setdefault(key, str(cache / directory))
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    if not args.simulate_only and dispatch_unsloth(
        runtime, ["-m", "lm_trainer", *sys.argv[1:]]
    ):
        return
    from .train.engine import simulate_pipeline, train

    (simulate_pipeline if args.simulate_only else train)(runtime)


if __name__ == "__main__":
    main()
