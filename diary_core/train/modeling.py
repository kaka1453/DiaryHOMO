from __future__ import annotations

import argparse
import json

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM

from diary_core.config.train_config import build_train_parser, build_train_runtime_config
from diary_core.model.quantization import build_quantization_config


def create_lora_config(runtime: dict) -> LoraConfig:
    return LoraConfig(
        r=runtime["lora_r"],
        lora_alpha=runtime["lora_alpha"],
        target_modules=runtime["lora_target_modules"],
        lora_dropout=runtime["lora_dropout"],
        bias=runtime["lora_bias"],
        task_type="CAUSAL_LM",
    )


def load_training_model(runtime: dict, lora_config: LoraConfig):
    quantization_config = build_quantization_config(runtime["quantization_mode"])
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "dtype": torch.float16,
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(runtime["model_name_or_path"], **load_kwargs)
    if runtime["quantization_mode"] != "none":
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return get_peft_model(model, lora_config)


def main() -> None:
    parser = build_train_parser()
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()
    runtime = build_train_runtime_config(args)
    lora_config = create_lora_config(runtime)
    summary = {
        "quantization_mode": runtime["quantization_mode"],
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_target_modules": sorted(list(lora_config.target_modules)),
        "load_model": args.load_model,
    }
    if args.load_model:
        model = load_training_model(runtime, lora_config)
        summary["model_class"] = model.__class__.__name__
        summary["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
