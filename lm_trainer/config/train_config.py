from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from lm_trainer.config.common import (
    dump_runtime_config,
    load_yaml_config,
    require_keys,
    str2bool,
)
from lm_trainer.model.quantization import normalize_quantization_mode

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
    "backend": "hf",
    "compute_dtype": "fp16",
    "gradient_checkpointing": "auto",
    "benchmark_warmup_steps": 2,
}

RUNTIME_FIELDS = set(
    TRAIN_CONFIG_FIELDS + MODEL_CONFIG_FIELDS + list(OPTIONAL_DEFAULTS)
)
PATH_FIELDS = {"model_name_or_path", "checkpoint_dir", "data_file_path"}
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
    (
        "--backend",
        "backend",
        str,
        "Training backend: hf or unsloth (separate processes).",
    ),
    ("--compute-dtype", "compute_dtype", str, "Compute precision: fp16 or bf16."),
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


def build_train_parser(
    default_train=None, default_model=None
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA causal language model training")
    parser.add_argument(
        "--config",
        type=str,
        default=default_train,
        required=default_train is None,
        help="Training YAML config file path.",
    )
    parser.add_argument(
        "--model-config",
        dest="model_config",
        type=str,
        default=default_model,
        required=default_model is None,
        help="Model YAML config file path.",
    )
    for flag, dest, value_type, help_text in CLI_OVERRIDE_SPECS:
        parser.add_argument(
            flag, dest=dest, type=value_type, default=None, help=help_text
        )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Only validate config and print resolved values.",
    )
    parser.add_argument(
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
    for key, default in OPTIONAL_DEFAULTS.items():
        runtime.setdefault(key, default)
    for field in PATH_FIELDS:
        runtime[field] = str((path_root / Path(runtime[field]).expanduser()).resolve())
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
    if runtime["compute_dtype"] not in {"fp16", "bf16"}:
        raise ValueError("compute_dtype 必须是 fp16 或 bf16。")
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


def build_train_runtime_config(
    args: argparse.Namespace, *, path_root: Path | None = None
) -> dict[str, Any]:
    train_config_path = Path(args.config).expanduser().resolve()
    model_config_path = Path(args.model_config).expanduser().resolve()
    train_cfg = load_yaml_config(train_config_path, "config/train.yaml")
    model_cfg = load_yaml_config(model_config_path, "config/model.yaml")

    runtime = collect_config_values(train_cfg, TRAIN_CONFIG_FIELDS)
    runtime.update(collect_config_values(model_cfg, MODEL_CONFIG_FIELDS))
    runtime.update(
        {key: train_cfg.get(key, default) for key, default in OPTIONAL_DEFAULTS.items()}
    )
    if runtime.get("quantization_mode") is None and "load_in_8bit" in train_cfg:
        runtime["quantization_mode"] = (
            "8bit" if str2bool(train_cfg["load_in_8bit"]) else "none"
        )

    apply_cli_overrides(runtime, args)
    require_keys(runtime, TRAIN_CONFIG_FIELDS + MODEL_CONFIG_FIELDS, "train/model")
    apply_runtime_casts(
        runtime, path_root if path_root is not None else train_config_path.parent
    )
    validate_runtime_values(runtime)
    runtime["train_config_path"] = train_config_path
    runtime["model_config_path"] = model_config_path
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
