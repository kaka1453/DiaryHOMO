"""Run from an installed package or a clean source snapshot."""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from .config.common import dump_runtime_config, init_asset_root
from .config.train_config import build_train_parser, build_train_runtime_config
from .template_policy import public_policy
from .train.bootstrap import dispatch_unsloth
from .train.io import validate_paths
from .train.schema import DataContractError, preflight_data


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
        stages = runtime.get("stages", [runtime])
        explicit = bool(stages[0].get("init_adapter"))
        if explicit:
            from .train.stages import artifact_reference, data_lineage, dev_preflight

            plans = []
            lineage_by_stage = {}
            for item in stages:
                if not (
                    args.dry_run or args.validate_release or args.simulate_only
                ) and os.path.lexists(item["checkpoint_dir"]):
                    raise FileExistsError(
                        "Stage output already exists. Use a fresh output directory."
                    )
                validate_paths(
                    item, require_model=not (args.dry_run or args.validate_release)
                )
                data = preflight_data(item)
                dev = dev_preflight(item, data)
                if item.get("parent_stage"):
                    reference = {"data_lineage": lineage_by_stage[item["parent_stage"]]}
                else:
                    _, reference, _ = artifact_reference(item)
                lineage = data_lineage(data.plan, reference)
                lineage_by_stage[item["name"]] = lineage
                shares = item.get("structured_sft_loss_token_share")
                if shares is not None and set(shares) != set(data.plan["sources"]):
                    raise ValueError(
                        "Sampling share keys must match all selected, approved sources"
                    )
                plan = {
                    **data.plan,
                    "template_policy": public_policy(data.plan["template_policy"]),
                }
                plans.append(
                    {
                        "stage_name": item["name"],
                        "init_adapter": item["init_adapter"],
                        "parent_stage": item.get("parent_stage"),
                        "seed": item["seed"],
                        "backend": item["backend"],
                        "compute_dtype": item["compute_dtype"],
                        "quantization": item["quantization_mode"],
                        "gradient_checkpointing": item["gradient_checkpointing"],
                        "lr": item["lr"],
                        "warmup_steps": item["warmup_steps"],
                        "lora_target_modules": item["lora_target_modules"],
                        "data_plan": plan,
                        "data_lineage": lineage,
                        "sampling_requirements": {
                            key: item.get(key)
                            for key in (
                                "require_exact_token_budget",
                                "sampling_max_share_error",
                                "structured_sft_loss_token_share",
                            )
                        },
                        "dev_manifest_sha256": dev.plan["manifest_sha256"]
                        if dev
                        else None,
                    }
                )
            if args.dry_run or args.validate_release:
                print(
                    json.dumps(
                        {"stages": plans, "token_counts_measured": False},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return
        elif args.dry_run:
            validate_paths(runtime, require_model=False)
            print(dump_runtime_config(runtime))
            return
        elif args.validate_release:
            plan = preflight_data(runtime).plan
            print(
                json.dumps(
                    {**plan, "template_policy": public_policy(plan["template_policy"])},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
    except DataContractError as exc:
        parser.exit(2, f"{exc}\n")
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    cache = Path(stages[0]["checkpoint_dir"]) / ".cache"
    if not (explicit and args.simulate_only):
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

    try:
        if args.simulate_only and explicit:
            for item, plan in zip(stages, plans):
                plan["simulation"] = simulate_pipeline(item, emit=False)
            print(
                json.dumps(
                    {"stages": plans, "token_counts_measured": True},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.simulate_only:
            simulate_pipeline(runtime)
        else:
            for item in stages:
                train(item)
    except DataContractError as exc:
        parser.exit(2, f"{exc}\n")
    except (OSError, ValueError, FloatingPointError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
