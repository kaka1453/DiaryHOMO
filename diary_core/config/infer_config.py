from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from diary_core.config.common import (
    PROJECT_ROOT,
    dump_runtime_config,
    load_yaml_config,
    require_keys,
    resolve_path,
    str2bool,
)
from diary_core.infer.prompt_debug import normalize_prompt_debug_config


DEFAULT_GENERATE_CONFIG_PATH = PROJECT_ROOT / "config" / "generate.yaml"
DEFAULT_WEBUI_CONFIG_PATH = PROJECT_ROOT / "config" / "webui.yaml"


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


def build_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="8bit HuggingFace batch generation")
    parser.add_argument("--config", type=str, default=str(DEFAULT_GENERATE_CONFIG_PATH), help="YAML config file path.")
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
    parser.add_argument("--config", type=str, default=str(DEFAULT_WEBUI_CONFIG_PATH), help="YAML config file path.")
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


def apply_prompt_debug_overrides(runtime: dict[str, Any], args: argparse.Namespace) -> None:
    prompt_debug = normalize_prompt_debug_config(runtime.get("prompt_debug"))
    if getattr(args, "prompt_debug_enabled", None) is not None:
        prompt_debug["enabled"] = bool(args.prompt_debug_enabled)
    if getattr(args, "prompt_debug_output_dir", None):
        prompt_debug["output_dir"] = str(resolve_path(args.prompt_debug_output_dir))
    runtime["prompt_debug"] = normalize_prompt_debug_config(prompt_debug)


def apply_cli_overrides(runtime: dict[str, Any], args: argparse.Namespace, excluded: set[str]) -> None:
    cli_overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded and value is not None
    }
    runtime.update(cli_overrides)


def build_batch_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve()
    raw_config = load_yaml_config(config_path, "config/generate.yaml")
    generation_cfg = raw_config.get("generation") or {}

    runtime = {
        "model_name_or_path": raw_config.get("model_name_or_path"),
        "checkpoint_dir": raw_config.get("checkpoint_dir"),
        "input_file": raw_config.get("input_file"),
        "output_file": raw_config.get("output_file"),
        "output_root": raw_config.get("output_root", "output"),
        "output_name": raw_config.get("output_name", "boa256日记"),
        "device": raw_config.get("device"),
        "batch_size": raw_config.get("batch_size"),
        "print_prompts": raw_config.get("print_prompts", True),
        "quantization_mode": raw_config.get("quantization_mode", "8bit"),
        "use_chat_template": raw_config.get("use_chat_template", True),
        "prompt_debug": raw_config.get("prompt_debug") or {},
        "stop_sequences": generation_cfg.get("stop_sequences", []),
        **{key: generation_cfg.get(key) for key in GENERATION_KEYS},
    }
    apply_cli_overrides(
        runtime,
        args,
        {"config", "dry_run", "prompt_debug_enabled", "prompt_debug_output_dir"},
    )

    require_keys(
        runtime,
        [
            "model_name_or_path",
            "checkpoint_dir",
            "input_file",
            "device",
            "batch_size",
            "print_prompts",
            *GENERATION_KEYS,
        ],
        "generate",
    )

    runtime["config_path"] = config_path
    runtime["model_name_or_path"] = str(resolve_path(runtime["model_name_or_path"]))
    runtime["checkpoint_dir"] = str(resolve_path(runtime["checkpoint_dir"]))
    runtime["input_file"] = str(resolve_path(runtime["input_file"]))
    if runtime.get("output_file"):
        runtime["legacy_output_file"] = str(resolve_path(runtime["output_file"]))
    runtime.pop("output_file", None)
    runtime["output_root"] = str(resolve_path(runtime.get("output_root") or "output"))
    runtime["output_name"] = str(runtime.get("output_name") or "boa256日记")
    runtime["batch_size"] = int(runtime["batch_size"])
    runtime["print_prompts"] = str2bool(runtime["print_prompts"])
    runtime["quantization_mode"] = str(runtime.get("quantization_mode") or "8bit")
    normalize_generation_fields(runtime)
    apply_prompt_debug_overrides(runtime, args)
    return runtime


def build_webui_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve()
    raw_config = load_yaml_config(config_path, "config/webui.yaml")
    generation_cfg = raw_config.get("generation") or {}

    runtime = {
        "model_name_or_path": raw_config.get("model_name_or_path"),
        "checkpoint_dir": raw_config.get("checkpoint_dir"),
        "device": raw_config.get("device"),
        "output_dir": raw_config.get("output_dir"),
        "output_root": raw_config.get("output_root", raw_config.get("output_dir") or "output"),
        "output_name": raw_config.get("output_name", "boa256日记"),
        "save_md": raw_config.get("save_md"),
        "server_name": raw_config.get("server_name"),
        "server_port": raw_config.get("server_port"),
        "share": raw_config.get("share"),
        "quantization_mode": raw_config.get("quantization_mode", "8bit"),
        "use_chat_template": raw_config.get("use_chat_template", True),
        "prompt_debug": raw_config.get("prompt_debug") or {},
        "stop_sequences": generation_cfg.get("stop_sequences", []),
        **{key: generation_cfg.get(key) for key in GENERATION_KEYS},
    }
    apply_cli_overrides(
        runtime,
        args,
        {"config", "dry_run", "prompt_debug_enabled", "prompt_debug_output_dir"},
    )

    require_keys(
        runtime,
        [
            "model_name_or_path",
            "checkpoint_dir",
            "device",
            "save_md",
            "server_name",
            "server_port",
            "share",
            *GENERATION_KEYS,
        ],
        "webui",
    )

    runtime["config_path"] = config_path
    runtime["model_name_or_path"] = str(resolve_path(runtime["model_name_or_path"]))
    runtime["checkpoint_dir"] = str(resolve_path(runtime["checkpoint_dir"]))
    if runtime.get("output_dir"):
        runtime["legacy_output_dir"] = str(resolve_path(runtime["output_dir"]))
    runtime.pop("output_dir", None)
    runtime["output_root"] = str(resolve_path(runtime.get("output_root") or "output"))
    runtime["output_name"] = str(runtime.get("output_name") or "boa256日记")
    runtime["save_md"] = str2bool(runtime["save_md"])
    runtime["share"] = str2bool(runtime["share"])
    runtime["server_port"] = int(runtime["server_port"])
    runtime["quantization_mode"] = str(runtime.get("quantization_mode") or "8bit")
    normalize_generation_fields(runtime)
    apply_prompt_debug_overrides(runtime, args)
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference config smoke test")
    parser.add_argument("--kind", choices=("batch", "webui"), default="batch")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if args.kind == "batch":
        batch_args = build_batch_parser().parse_args(["--config", args.config or str(DEFAULT_GENERATE_CONFIG_PATH), "--dry-run"])
        runtime = build_batch_runtime_config(batch_args)
    else:
        webui_args = build_webui_parser().parse_args(["--config", args.config or str(DEFAULT_WEBUI_CONFIG_PATH), "--dry-run"])
        runtime = build_webui_runtime_config(webui_args)
    print(dump_runtime_config(runtime))


if __name__ == "__main__":
    main()
