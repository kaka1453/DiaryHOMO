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
from diary_core.config.assets import build_or_exit
from diary_core.config.common import dump_runtime_config


def main() -> None:
    parser = build_train_parser()
    args = parser.parse_args()
    runtime = build_or_exit(parser, build_train_runtime_config, args)

    if isinstance(runtime, dict) and (
        "stages" in runtime or bool(runtime.get("init_adapter"))
    ):
        parser.exit(
            2,
            "Legacy diary_core.train.engine only supports single-stage legacy runs. "
            "Use `python -m lm_trainer --asset-root ...` for staged or explicit init_adapter workflows.\n",
        )

    if args.dry_run:
        print("Train 配置检查通过。")
        print(dump_runtime_config(runtime))
        return

    if getattr(args, "validate_release", False):
        import json
        from lm_trainer.template_policy import public_policy
        from lm_trainer.train.io import validate_paths
        from lm_trainer.train.schema import DataContractError, preflight_data

        try:
            validate_paths(runtime, require_model=False)
            plan = preflight_data(runtime).plan
        except DataContractError as exc:
            parser.exit(2, f"{exc}\n")
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {**plan, "template_policy": public_policy(plan["template_policy"])},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.simulate_only:
        simulate_pipeline(runtime)
        return

    train(runtime)


if __name__ == "__main__":
    main()
