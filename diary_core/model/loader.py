from __future__ import annotations

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from diary_core.config.infer_config import build_batch_parser, build_batch_runtime_config
from diary_core.model.quantization import build_quantization_config


def load_tokenizer(model_name_or_path: str, use_fast: bool | None = None):
    load_kwargs = {
        "trust_remote_code": True,
    }
    if use_fast is not None:
        load_kwargs["use_fast"] = use_fast

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        **load_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name_or_path: str, quantization_mode: str = "8bit"):
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    quantization_config = build_quantization_config(quantization_mode)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["dtype"] = torch.float16

    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)


def load_model_and_tokenizer(runtime: dict):
    tokenizer = load_tokenizer(runtime["model_name_or_path"])
    base_model = load_base_model(
        runtime["model_name_or_path"],
        quantization_mode=runtime.get("quantization_mode", "8bit"),
    )
    model = PeftModel.from_pretrained(base_model, runtime["checkpoint_dir"])
    model.eval()
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Model loader smoke test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    argv = ["--dry-run"]
    if args.config:
        argv.extend(["--config", args.config])
    runtime = build_batch_runtime_config(build_batch_parser().parse_args(argv))
    summary = {
        "model_name_or_path": runtime["model_name_or_path"],
        "checkpoint_dir": runtime["checkpoint_dir"],
        "quantization_mode": runtime.get("quantization_mode", "8bit"),
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    tokenizer, model = load_model_and_tokenizer(runtime)
    print(json.dumps({"tokenizer_vocab": len(tokenizer), "model_class": model.__class__.__name__}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
