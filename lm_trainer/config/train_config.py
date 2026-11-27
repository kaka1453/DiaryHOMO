from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from lm_trainer.config.common import (
    asset_path,
    dump_runtime_config,
    load_yaml_config,
    open_asset_root,
    require_keys,
    str2bool,
)
from lm_trainer.config.stages import (
    BOUNDED_DEFAULTS,
    STAGE_DEFAULTS,
    STAGE_LOCAL,
    check_stage_paths,
    configure_stage,
    reject_placeholders,
)
from lm_trainer.model.quantization import normalize_quantization_mode
from lm_trainer.train.schema import DATA_DEFAULTS, configure_data

TRAIN_CONFIG_FIELDS = [
    "model_name_or_path",
    "checkpoint_dir",
    "data_file_path",
    "max_length",
    "quantization_mode",
    "batch_size",
    "shuffle",
    "auto_pad_batch",
    "epochs",
    "gradient_accumulation_steps",
    "lr",
    "warmup_steps",
    "max_steps",
    "save_steps",
    "seed",
    "num_workers",
    "pin_memory",
]

MODEL_CONFIG_FIELDS = [
    "lora_r",
    "lora_alpha",
    "lora_target_modules",
    "lora_dropout",
    "lora_bias",
]

OPTIONAL_DEFAULTS = {
    **DATA_DEFAULTS,
    "backend": "hf",
    "compute_dtype": "fp16",
    "gradient_checkpointing": "auto",
    "benchmark_warmup_steps": 2,
}

RUNTIME_FIELDS = set(
    TRAIN_CONFIG_FIELDS
    + MODEL_CONFIG_FIELDS
    + list(OPTIONAL_DEFAULTS)
    + list(STAGE_DEFAULTS)
)
PATH_FIELDS = {
    "model_name_or_path",
    "checkpoint_dir",
    "data_file_path",
    "manifest_path",
}
INT_FIELDS = {
    "benchmark_warmup_steps",
    "max_length",
    "batch_size",
    "epochs",
    "gradient_accumulation_steps",
    "warmup_steps",
    "max_steps",
    "save_steps",
    "seed",
    "num_workers",
    "lora_r",
    "lora_alpha",
}
FLOAT_FIELDS = {"lr", "lora_dropout"}
BOOL_FIELDS = {"shuffle", "auto_pad_batch", "pin_memory"}
POSITIVE_FIELDS = {
    "max_length",
    "batch_size",
    "gradient_accumulation_steps",
    "epochs",
    "max_steps",
    "save_steps",
    "lora_r",
    "lora_alpha",
}
NON_NEGATIVE_FIELDS = {"warmup_steps", "num_workers", "benchmark_warmup_steps"}
ALLOWED_QUANTIZATION_MODES = {"4bit", "8bit", "none"}


CLI_OVERRIDE_SPECS = [
    ("--stage-name", "name", str, "Name of an explicit single stage."),
    ("--init-adapter", "init_adapter", str, "new, continue_stage or resume."),
    ("--parent", "parent", str, "Explicit inference_adapter to inherit."),
    ("--resume-from", "resume_from", str, "Explicit training_checkpoint to resume."),
    (
        "--stop-after-updates",
        "stop_after_updates",
        int,
        "HF only: checkpoint and pause at an update boundary.",
    ),
    ("--output-dir", "output_dir", str, "Alias for the single-stage checkpoint_dir."),
    ("--data-mode", "data_mode", str, "Data entrance: legacy, fixture, or release."),
    ("--objective", "objective", str, "Explicit fixture/release training objective."),
    ("--manifest-path", "manifest_path", str, "Path to a fixture or release manifest."),
    (
        "--backend",
        "backend",
        str,
        "Training backend: hf or unsloth (separate processes).",
    ),
    ("--compute-dtype", "compute_dtype", str, "Compute precision: fp16, bf16 or fp32."),
    (
        "--gradient-checkpointing",
        "gradient_checkpointing",
        str,
        "auto, standard, unsloth, or none.",
    ),
    (
        "--benchmark-warmup-steps",
        "benchmark_warmup_steps",
        int,
        "Optimizer updates excluded from steady-state metrics.",
    ),
    (
        "--model-name-or-path",
        "model_name_or_path",
        str,
        "Override model path from config.",
    ),
    (
        "--checkpoint-dir",
        "checkpoint_dir",
        str,
        "Override checkpoint output directory from config.",
    ),
    (
        "--data-file-path",
        "data_file_path",
        str,
        "Override training dataset path from config.",
    ),
    ("--max-length", "max_length", int, "Override max sequence length from config."),
    (
        "--quantization-mode",
        "quantization_mode",
        str,
        "Override quantization mode from config: 4bit, 8bit, or none.",
    ),
    ("--batch-size", "batch_size", int, "Override batch size from config."),
    (
        "--shuffle",
        "shuffle",
        str2bool,
        "Override whether sorted batches are shuffled between epochs.",
    ),
    (
        "--auto-pad-batch",
        "auto_pad_batch",
        str2bool,
        "Override adaptive padding switch from config.",
    ),
    ("--epochs", "epochs", int, "Override max epoch count from config."),
    (
        "--gradient-accumulation-steps",
        "gradient_accumulation_steps",
        int,
        "Override gradient accumulation steps from config.",
    ),
    ("--lr", "lr", float, "Override learning rate from config."),
    ("--warmup-steps", "warmup_steps", int, "Override warmup steps from config."),
    ("--max-steps", "max_steps", int, "Override total optimizer steps from config."),
    (
        "--save-steps",
        "save_steps",
        int,
        "Override checkpoint save interval from config.",
    ),
    ("--seed", "seed", int, "Override random seed from config."),
    (
        "--num-workers",
        "num_workers",
        int,
        "Override dataloader worker count from config.",
    ),
    (
        "--pin-memory",
        "pin_memory",
        str2bool,
        "Override dataloader pin_memory switch from config.",
    ),
    ("--lora-r", "lora_r", int, "Override LoRA rank from model config."),
    ("--lora-alpha", "lora_alpha", int, "Override LoRA alpha from model config."),
    (
        "--lora-target-modules",
        "lora_target_modules",
        str,
        "Override LoRA target modules from model config, comma-separated.",
    ),
    (
        "--lora-dropout",
        "lora_dropout",
        float,
        "Override LoRA dropout from model config.",
    ),
    ("--lora-bias", "lora_bias", str, "Override LoRA bias from model config."),
]


def parse_target_modules(value: Any) -> list[str]:
    if value is None:
        raise ValueError("lora_target_modules 不能为空。")
    if isinstance(value, list):
        modules = [str(item).strip() for item in value if str(item).strip()]
    else:
        modules = [item.strip() for item in str(value).split(",") if item.strip()]
    if not modules:
        raise ValueError("lora_target_modules 不能为空。")
    return modules


class TrainArgumentParser(argparse.ArgumentParser):
    def __init__(self, default_train, default_model):
        super().__init__(description="LoRA causal language model training")
        self.config_defaults = {"config": default_train, "model_config": default_model}

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if parsed.asset_root is not None:
            return parsed
        if parsed.external_base_model is not None:
            self.error("--external-base-model requires --asset-root")
        missing = []
        for field, default in self.config_defaults.items():
            if getattr(parsed, field) is None:
                setattr(parsed, field, default)
                if default is None:
                    missing.append("--" + field.replace("_", "-"))
        if missing:
            self.error("the following arguments are required: " + ", ".join(missing))
        return parsed


def build_train_parser(
    default_train=None, default_model=None
) -> argparse.ArgumentParser:
    parser = TrainArgumentParser(default_train, default_model)
    parser.add_argument(
        "--asset-root",
        help="Use this asset directory. Defaults to config/train.yaml and config/model.yaml within it.",
    )
    parser.add_argument(
        "--config",
        help="Training YAML path. Required unless using an asset root or an entry point with a default.",
    )
    parser.add_argument(
        "--model-config",
        help="Model YAML path. Required unless using an asset root or an entry point with a default.",
    )
    parser.add_argument(
        "--external-base-model",
        help="Absolute path to a shared local model directory in asset-root mode.",
    )
    for flag, dest, value_type, help_text in CLI_OVERRIDE_SPECS:
        parser.add_argument(
            flag, dest=dest, type=value_type, default=None, help=help_text
        )
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument(
        "--validate-release",
        action="store_true",
        help="Check data permissions and integrity and print a stage plan without loading a model.",
    )
    checks.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Only validate config and print resolved values.",
    )
    checks.add_argument(
        "--simulate-only",
        dest="simulate_only",
        action="store_true",
        help="Run a CPU-side simulation without loading the model into VRAM.",
    )
    return parser


def collect_config_values(
    config_data: dict[str, Any], fields: list[str]
) -> dict[str, Any]:
    return {field: config_data.get(field) for field in fields}


def apply_runtime_casts(runtime: dict[str, Any], path_root: Path) -> None:
    for field in PATH_FIELDS:
        if runtime[field] is not None:
            path = path_root / Path(runtime[field]).expanduser()
            runtime[field] = str(
                path.absolute() if field == "manifest_path" else path.resolve()
            )
    cast_runtime_values(runtime)


def cast_runtime_values(runtime: dict[str, Any]) -> None:
    for key, default in OPTIONAL_DEFAULTS.items():
        runtime.setdefault(key, default)
    for field in INT_FIELDS:
        runtime[field] = int(runtime[field])
    for field in FLOAT_FIELDS:
        runtime[field] = float(runtime[field])
    for field in BOOL_FIELDS:
        runtime[field] = str2bool(runtime[field])
    runtime["quantization_mode"] = normalize_quantization_mode(
        runtime["quantization_mode"]
    )
    runtime["lora_target_modules"] = parse_target_modules(
        runtime["lora_target_modules"]
    )
    runtime["lora_bias"] = str(runtime["lora_bias"])


def validate_runtime_values(runtime: dict[str, Any]) -> None:
    for key, default in OPTIONAL_DEFAULTS.items():
        runtime.setdefault(key, default)
    if runtime["backend"] not in {"hf", "unsloth"}:
        raise ValueError("backend 必须是 hf 或 unsloth。")
    if runtime["compute_dtype"] not in {"fp16", "bf16", "fp32"}:
        raise ValueError("compute_dtype 必须是 fp16、bf16 或 fp32。")
    if runtime["compute_dtype"] == "fp32" and (
        runtime["backend"] != "hf" or runtime["quantization_mode"] != "none"
    ):
        raise ValueError("fp32 requires an unquantized HF model.")
    if runtime["lora_bias"] not in {"none", "all", "lora_only"}:
        raise ValueError("Unsupported lora_bias.")
    if not math.isfinite(runtime["lr"]) or runtime["lr"] <= 0:
        raise ValueError("lr must be finite and positive.")
    if runtime["gradient_checkpointing"] == "auto":
        runtime["gradient_checkpointing"] = (
            "unsloth" if runtime["backend"] == "unsloth" else "standard"
        )
    if runtime["gradient_checkpointing"] not in {"standard", "unsloth", "none"}:
        raise ValueError("gradient_checkpointing 必须是 auto/standard/unsloth/none。")
    if runtime["backend"] == "hf" and runtime["gradient_checkpointing"] == "unsloth":
        raise ValueError("HF 后端不能使用 Unsloth 梯度检查点。")
    if runtime["backend"] == "unsloth" and runtime["quantization_mode"] == "8bit":
        raise ValueError("本次 Unsloth 接入仅提供 4bit/none，不支持 8bit。")
    if not 0 <= runtime["lora_dropout"] < 1:
        raise ValueError("lora_dropout 必须在 [0, 1) 内。")
    if runtime["quantization_mode"] not in ALLOWED_QUANTIZATION_MODES:
        raise ValueError(
            f"quantization_mode 必须是 {', '.join(sorted(ALLOWED_QUANTIZATION_MODES))} 之一。"
        )
    for field in POSITIVE_FIELDS:
        if runtime[field] <= 0:
            raise ValueError(f"{field} 必须大于 0。")
    for field in NON_NEGATIVE_FIELDS:
        if runtime[field] < 0:
            raise ValueError(f"{field} 不能小于 0。")


def apply_cli_overrides(runtime: dict[str, Any], args: argparse.Namespace) -> None:
    cli_overrides = {
        key: value
        for key, value in vars(args).items()
        if key in RUNTIME_FIELDS and value is not None
    }
    runtime.update(cli_overrides)


def resolve_asset_runtime_paths(
    runtime: dict[str, Any],
    root: Path,
    *,
    eval_file: str | None,
    external_model: str | None,
    dry_run: bool,
) -> None:
    field = "data_file_path" if runtime["data_mode"] == "legacy" else "manifest_path"
    if any(pattern in runtime[field] for pattern in ("*", "?", "[", "::")):
        raise ValueError(f"{field} must be a literal path")
    runtime[field] = str(asset_path(root, runtime[field], field, "data/train"))
    if eval_file is not None:
        runtime["eval_file_path"] = str(
            asset_path(root, eval_file, "eval_file_path", "data/eval")
        )

    field = "model_name_or_path"
    if external_model is None:
        model = asset_path(
            root, runtime[field], field, "assets", directory=True, required=not dry_run
        )
    else:
        model = Path(external_model)
        if not model.is_absolute() or not model.is_dir():
            raise ValueError(
                "--external-base-model must name an existing absolute model directory"
            )
        model = model.resolve(strict=True)
    runtime[field] = str(model)

    field = "checkpoint_dir"
    output = asset_path(
        root,
        runtime[field],
        field,
        "checkpoints",
        "runs",
        directory=True,
        required=False,
    )
    # The trainer also writes files and caches beneath this directory.
    pending = [output] if output.is_dir() else []
    while pending:
        for child in pending.pop().iterdir():
            if child.is_symlink():
                raise ValueError(
                    f"{field}: output contains symlink {str(child.relative_to(root))!r}"
                )
            if child.is_dir():
                pending.append(child)
    runtime[field] = str(output)


def build_train_runtime_config(
    args: argparse.Namespace, *, path_root: Path | None = None
) -> dict[str, Any]:
    asset_root = getattr(args, "asset_root", None)
    root = None if asset_root is None else open_asset_root(asset_root)
    if root is None:
        train_config_path = Path(args.config).expanduser().resolve()
        model_config_path = Path(args.model_config).expanduser().resolve()
        train_label, model_label = "config/train.yaml", "config/model.yaml"
    else:
        train_label = "config/train.yaml" if args.config is None else args.config
        model_label = (
            "config/model.yaml" if args.model_config is None else args.model_config
        )
        train_config_path = asset_path(root, train_label, "--config", "config")
        model_config_path = asset_path(root, model_label, "--model-config", "config")
    train_cfg = load_yaml_config(train_config_path, train_label)
    model_cfg = load_yaml_config(model_config_path, model_label)

    options = (
        args,
        root,
        path_root or train_config_path.parent,
        train_label,
        model_label,
    )
    stages = train_cfg.get("stages")
    if "stages" in train_cfg:
        if not isinstance(stages, list) or not stages:
            raise ValueError("stages must be a nonempty list of stage mappings.")
        if any(getattr(args, k, None) is not None for k in STAGE_LOCAL):
            raise ValueError("Stage-local CLI overrides require a single-stage config.")
        shared = {
            k: v for k, v in train_cfg.items() if k not in STAGE_LOCAL | {"stages"}
        }
        if set(shared) - RUNTIME_FIELDS:
            raise ValueError("Unknown shared stage configuration field.")
        runtimes = []
        for item in stages:
            if not isinstance(item, dict) or set(item) - RUNTIME_FIELDS:
                raise ValueError("Unknown field or invalid stage mapping.")
            runtimes.append(
                _build_runtime({**shared, **item}, model_cfg, *options, staged=True)
            )
        check_stage_paths(runtimes)
        if len({(r["backend"], r["compute_dtype"]) for r in runtimes}) != 1:
            raise ValueError(
                "One stages plan requires a single backend and compute dtype."
            )
        if any(r["stop_after_updates"] is not None for r in runtimes):
            raise ValueError("Use stop_after_updates with a single-stage config.")
        return {"stages": runtimes, "backend": runtimes[0]["backend"]}

    staged = any(
        k in train_cfg or getattr(args, k, None) is not None for k in STAGE_DEFAULTS
    )
    if staged and set(train_cfg) - RUNTIME_FIELDS - {"load_in_8bit", "eval_file_path"}:
        raise ValueError("Unknown stage configuration field.")
    runtime = _build_runtime(train_cfg, model_cfg, *options, staged=staged)
    if staged:
        check_stage_paths([runtime])
    runtime["train_config_path"] = train_config_path
    runtime["model_config_path"] = model_config_path
    return runtime


def _build_runtime(
    train_cfg, model_cfg, args, root, path_root, train_label, model_label, *, staged
):
    if staged:
        reject_placeholders(dict(model_cfg) | dict(train_cfg))
        train_cfg = {**BOUNDED_DEFAULTS, **train_cfg}
    runtime = collect_config_values(train_cfg, TRAIN_CONFIG_FIELDS)
    runtime.update(collect_config_values(model_cfg, MODEL_CONFIG_FIELDS))
    if staged:
        runtime.update({k: train_cfg[k] for k in MODEL_CONFIG_FIELDS if k in train_cfg})
        runtime.update({k: train_cfg.get(k, v) for k, v in STAGE_DEFAULTS.items()})
    runtime.update(
        {key: train_cfg.get(key, default) for key, default in OPTIONAL_DEFAULTS.items()}
    )
    if runtime.get("quantization_mode") is None and "load_in_8bit" in train_cfg:
        runtime["quantization_mode"] = (
            "8bit" if str2bool(train_cfg["load_in_8bit"]) else "none"
        )
    apply_cli_overrides(runtime, args)
    if staged:
        reject_placeholders(runtime)
        if runtime["output_dir"] is not None:
            if runtime["checkpoint_dir"] is not None:
                raise ValueError("Choose output_dir or checkpoint_dir, not both.")
            runtime["checkpoint_dir"] = runtime["output_dir"]
        if any(
            type(runtime[k]) is not int for k in INT_FIELDS if runtime[k] is not None
        ):
            raise ValueError("Stage integer fields require literal integers.")
    runtime.update(configure_data(runtime))
    required_train = [
        key
        for key in TRAIN_CONFIG_FIELDS
        if key != "data_file_path" or runtime["data_mode"] == "legacy"
    ]
    if root is None:
        require_keys(runtime, required_train + MODEL_CONFIG_FIELDS, "train/model")
        apply_runtime_casts(runtime, path_root)
    else:
        external_model = getattr(args, "external_base_model", None)
        if external_model is not None:
            if getattr(args, "model_name_or_path", None) is not None:
                raise ValueError(
                    "Choose either --model-name-or-path or --external-base-model"
                )
            runtime["model_name_or_path"] = external_model
        require_keys(runtime, required_train, train_label)
        require_keys(runtime, MODEL_CONFIG_FIELDS, model_label)
        resolve_asset_runtime_paths(
            runtime,
            root,
            eval_file=train_cfg.get("eval_file_path"),
            external_model=external_model,
            dry_run=getattr(args, "dry_run", False)
            or getattr(args, "validate_release", False),
        )
        cast_runtime_values(runtime)
    validate_runtime_values(runtime)
    if staged:
        configure_stage(runtime, root, path_root)
    return runtime


build_parser = build_train_parser
build_runtime_config = build_train_runtime_config


def main() -> None:
    parser = build_train_parser()
    args = parser.parse_args()
    runtime = build_train_runtime_config(args)
    print(dump_runtime_config(runtime))


if __name__ == "__main__":
    main()
