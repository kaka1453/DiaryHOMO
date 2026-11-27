from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_trainer.model.quantization import build_quantization_config


def load_tokenizer(
    model_name_or_path: str,
    use_fast: bool | None = None,
    *,
    local_files_only: bool = False,
):
    load_kwargs = {
        "trust_remote_code": not local_files_only,
        "local_files_only": local_files_only,
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
