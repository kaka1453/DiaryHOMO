from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from diary_core.config.assets import (
    add_asset_root,
    build_or_exit,
    external_base_model,
    load_product_config,
    product_root,
)
from diary_core.config.common import dump_runtime_config, require_keys, str2bool
from diary_core.infer.product_content import load_product_content
from diary_core.infer.prompt_debug import normalize_prompt_debug_config
from lm_trainer.config.common import asset_path


DEFAULT_OUTPUT_ROOT = "runs/generation"
DEFAULT_OUTPUT_NAME = "diary"


GENERATION_KEYS = [
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "num_beams",
]


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", dest="temperature", type=float, default=None)
    parser.add_argument("--top-p", dest="top_p", type=float, default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=None)
    parser.add_argument("--num-beams", dest="num_beams", type=int, default=None)
    parser.add_argument("--stop-sequences", dest="stop_sequences", type=str, default=None)
    parser.add_argument("--use-chat-template", dest="use_chat_template", type=str2bool, default=None)
    parser.add_argument("--prompt-debug", dest="prompt_debug_enabled", action="store_true", default=None)
    parser.add_argument("--prompt-debug-output-dir", dest="prompt_debug_output_dir", type=str, default=None)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="YAML path relative to the asset root.")
    parser.add_argument("--external-base-model", help="Absolute path to a shared local base model.")


def build_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="8bit HuggingFace batch generation")
    add_asset_root(parser)
    add_config_args(parser)
    parser.add_argument("--model-name-or-path", dest="model_name_or_path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", type=str, default=None)
    parser.add_argument("--input-file", dest="input_file", type=str, default=None)
    parser.add_argument("--output-file", dest="output_file", type=str, default=None)
    parser.add_argument("--output-root", dest="output_root", type=str, default=None)
    parser.add_argument("--output-name", dest="output_name", type=str, default=None)
    parser.add_argument("--device", dest="device", type=str, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--print-prompts", dest="print_prompts", type=str2bool, default=None)
    parser.add_argument("--quantization-mode", dest="quantization_mode", type=str, default=None)
    add_generation_args(parser)
    return parser


def build_webui_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="8bit HuggingFace WebUI")
    add_asset_root(parser)
    add_config_args(parser)
    parser.add_argument("--model-name-or-path", dest="model_name_or_path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", type=str, default=None)
    parser.add_argument("--device", dest="device", type=str, default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--output-root", dest="output_root", type=str, default=None)
    parser.add_argument("--output-name", dest="output_name", type=str, default=None)
    parser.add_argument("--save-md", dest="save_md", type=str2bool, default=None)
    parser.add_argument("--server-name", dest="server_name", type=str, default=None)
    parser.add_argument("--server-port", dest="server_port", type=int, default=None)
    parser.add_argument("--share", dest="share", type=str2bool, default=None)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--quantization-mode", dest="quantization_mode", type=str, default=None)
    add_generation_args(parser)
    return parser


def normalize_generation_fields(runtime: dict[str, Any]) -> None:
    runtime["max_new_tokens"] = int(runtime["max_new_tokens"])
    runtime["temperature"] = float(runtime["temperature"])
    runtime["top_p"] = float(runtime["top_p"])
    runtime["top_k"] = int(runtime["top_k"])
    runtime["repetition_penalty"] = float(runtime["repetition_penalty"])
    runtime["num_beams"] = int(runtime["num_beams"])
    runtime["stop_sequences"] = normalize_stop_sequences(runtime.get("stop_sequences"))
    runtime["use_chat_template"] = str2bool(runtime.get("use_chat_template", True))


def normalize_stop_sequences(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item for item in value.split(",") if item]
    return [str(item) for item in value if str(item)]


def deep_merge_dict(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if not isinstance(overrides, dict):
        return merged
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_diary_profile_runtime_config(raw_config: dict[str, Any], root: Path) -> dict[str, Any]:
    config = {"enabled": True, **(raw_config.get("diary_profile") or {})}
    config["enabled"] = str2bool(config["enabled"])
    if config["enabled"]:
        name = config.get("profile_path", "assets/persona/diary_profile.yaml")
        path = asset_path(root, name, "diary_profile.profile_path", "assets/persona")
        config["profile_path"] = str(path)
        config["profile"] = load_product_config(path, "diary_profile.profile_path")
    else:
        config["profile"] = {}
    return config


def apply_cli_overrides(runtime: dict[str, Any], args: argparse.Namespace, excluded: set[str]) -> None:
    cli_overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded and value is not None
    }
    runtime.update(cli_overrides)


def build_runtime_config(args: argparse.Namespace, kind: str, *, evaluation: bool = False) -> dict[str, Any]:
    root = product_root(args.asset_root)
    config_path = asset_path(root, args.config or f"config/{kind}.yaml", "--config", "config")
    raw = load_product_config(config_path, "--config")
    generation = raw.get("generation") or {}
    guard_path = asset_path(root, raw.get("guard_path", "assets/guard/guard.yaml"), "guard_path", "assets/guard")
    guard = deep_merge_dict(load_product_config(guard_path, "guard_path"), raw.get("guard"))
    base_audit = guard.get("audit") if isinstance(guard.get("audit"), dict) else {}
    webui = kind == "webui"
    fields = ["model_name_or_path", "checkpoint_dir", "device"]
    fields.extend(["save_md", "server_name", "server_port", "share"] if webui else ["input_file", "batch_size"])
    runtime = {
        **{field: raw.get(field) for field in fields},
        "asset_root": str(root),
        "config_path": config_path,
        "output_root": raw.get("output_root", raw.get("output_dir") or ("runs/webui" if webui else DEFAULT_OUTPUT_ROOT)),
        "output_name": raw.get("output_name", DEFAULT_OUTPUT_NAME),
        "print_prompts": raw.get("print_prompts", False),
        "quantization_mode": raw.get("quantization_mode", "8bit"),
        "use_chat_template": raw.get("use_chat_template", True),
        "generation_profiles": raw.get("generation_profiles") or {},
        "diary_profile": load_diary_profile_runtime_config(raw, root),
        "guard": guard,
        "audit": deep_merge_dict(base_audit, raw.get("audit")),
        "stop_sequences": generation.get("stop_sequences", []),
        **{key: generation.get(key) for key in GENERATION_KEYS},
    }
    apply_cli_overrides(runtime, args, {
        "asset_root", "config", "dry_run", "external_base_model",
        "prompt_debug_enabled", "prompt_debug_output_dir",
    })
    if args.external_base_model is not None:
        if args.model_name_or_path is not None:
            raise ValueError("Choose either --model-name-or-path or --external-base-model")
        runtime["model_name_or_path"] = str(external_base_model(args.external_base_model))
    require_keys(runtime, [*fields, *GENERATION_KEYS], kind)
    if args.external_base_model is None:
        runtime["model_name_or_path"] = str(asset_path(
            root, runtime["model_name_or_path"], "model_name_or_path", "assets", directory=True,
        ))
    adapter = asset_path(
        root, runtime["checkpoint_dir"], "checkpoint_dir", "checkpoints", "runs", directory=True,
    )
    runtime["checkpoint_dir"] = str(adapter)
    adapter_parent = adapter.relative_to(root)
    for filename in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        asset_path(root, str(adapter_parent / filename), filename, "checkpoints", "runs",
                   required=filename == "adapter_config.json" and not args.dry_run)
    if getattr(args, "output_dir", None):
        runtime["output_root"] = args.output_dir
    runtime["output_root"] = str(asset_path(
        root, runtime["output_root"], "output_root", "runs", directory=True, required=False,
    ))
    from diary_core.infer.output_bundle import normalize_output_name

    runtime["output_name"] = normalize_output_name(runtime["output_name"])
    output_parent = Path(runtime["output_root"]).relative_to(root) / runtime["output_name"]
    asset_path(root, str(output_parent), "output parent", "runs", directory=True, required=False)
    if webui:
        runtime["save_md"] = str2bool(runtime["save_md"])
        runtime["share"] = str2bool(runtime["share"])
        runtime["server_port"] = int(runtime["server_port"])
    else:
        inputs = ("data/eval",) if evaluation else ("assets/prompts", "data/eval")
        runtime["input_file"] = str(asset_path(root, runtime["input_file"], "input_file", *inputs))
        output_file = runtime.get("output_file") or raw.get("output_file")
        if output_file:
            runtime["requested_output_file"] = str(asset_path(root, output_file, "output_file", "runs", required=False))
        runtime["batch_size"] = int(runtime["batch_size"])
        if runtime["batch_size"] <= 0:
            raise ValueError("batch_size must be positive")
        runtime["print_prompts"] = str2bool(runtime["print_prompts"])
    runtime["quantization_mode"] = str(runtime["quantization_mode"] or "8bit")
    normalize_generation_fields(runtime)
    prompt_debug = dict(raw.get("prompt_debug") or {})
    if args.prompt_debug_enabled is not None:
        prompt_debug["enabled"] = args.prompt_debug_enabled
    if args.prompt_debug_output_dir is not None:
        prompt_debug["output_dir"] = args.prompt_debug_output_dir
    if "output_dir" in prompt_debug:
        prompt_debug["output_dir"] = str(asset_path(
            root, prompt_debug["output_dir"], "prompt_debug.output_dir", "runs", directory=True, required=False,
        ))
    runtime["prompt_debug"] = normalize_prompt_debug_config(prompt_debug, root)
    runtime["product_content"] = load_product_content(root)
    return runtime


def build_batch_runtime_config(args: argparse.Namespace, *, evaluation: bool = False) -> dict[str, Any]:
    return build_runtime_config(args, "generate", evaluation=evaluation)


def build_webui_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    return build_runtime_config(args, "webui")


def inference_summary(runtime: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "asset_root", "config_path", "model_name_or_path", "checkpoint_dir", "input_file", "device",
        "output_root", "output_name", "output_run_dir", "output_file", "requested_output_file", "parameters_file", "quantization_mode",
        "batch_size", "server_name", "server_port", "share", "save_md", *GENERATION_KEYS,
    )
    summary = {key: str(runtime[key]) if isinstance(runtime[key], Path) else runtime[key] for key in keys if key in runtime}
    if isinstance(runtime.get("prompt_debug"), dict):
        summary["prompt_debug"] = {key: runtime["prompt_debug"].get(key) for key in ("enabled", "output_dir")}
    if isinstance(runtime.get("diary_profile"), dict):
        summary["diary_profile"] = {"enabled": runtime["diary_profile"]["enabled"]}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate product inference config", add_help=False)
    parser.add_argument("--kind", choices=("batch", "webui"), default="batch")
    args, argv = parser.parse_known_args()
    if args.kind == "batch":
        entry, builder = build_batch_parser(), build_batch_runtime_config
    else:
        entry, builder = build_webui_parser(), build_webui_runtime_config
    runtime = build_or_exit(entry, builder, entry.parse_args(argv))
    print(dump_runtime_config(inference_summary(runtime)))


if __name__ == "__main__":
    main()
