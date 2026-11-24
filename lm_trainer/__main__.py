"""Run from an installed package or a clean source snapshot."""

import os
from pathlib import Path
import sys

from .config.common import dump_runtime_config
from .config.train_config import build_train_parser, build_train_runtime_config
from .train.bootstrap import dispatch_unsloth


def main():
    args = build_train_parser().parse_args()
    runtime = build_train_runtime_config(args)
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
