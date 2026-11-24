"""Compatibility interface for lm_trainer.train.io."""

from lm_trainer.train.io import (
    validate_paths,
    save_runtime_snapshot,
)

import json
from diary_core.config.train_config import (
    build_train_parser,
    build_train_runtime_config,
)
from pathlib import Path


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
