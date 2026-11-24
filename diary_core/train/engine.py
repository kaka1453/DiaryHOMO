"""Compatibility interface for lm_trainer.train.engine."""

from lm_trainer.train.engine import (
    simulate_pipeline,
    accumulation_windows,
    update_window,
    save_adapter,
    output_directory,
    record_epoch,
)
from diary_core.train.performance import legacy_lock_files
from lm_trainer.train import engine as core


def train(runtime):
    return core.train(runtime, lock_files=legacy_lock_files())


def run_training(runtime, directory, started):
    return core.run_training(
        runtime, directory, started, lock_files=legacy_lock_files()
    )


from diary_core.config.train_config import (
    build_train_parser,
    build_train_runtime_config,
)
from diary_core.config.common import dump_runtime_config


def main() -> None:
    parser = build_train_parser()
    args = parser.parse_args()
    runtime = build_train_runtime_config(args)

    if args.dry_run:
        print("Train 配置检查通过。")
        print(dump_runtime_config(runtime))
        return

    if args.simulate_only:
        simulate_pipeline(runtime)
        return

    train(runtime)


if __name__ == "__main__":
    main()
