from __future__ import annotations

import warnings

from lm_trainer.train.bootstrap import initialize_backend


def create_lora_config(runtime: dict):
    from peft import LoraConfig

    return LoraConfig(
        r=runtime["lora_r"],
        lora_alpha=runtime["lora_alpha"],
        target_modules=runtime["lora_target_modules"],
        lora_dropout=runtime["lora_dropout"],
        bias=runtime["lora_bias"],
        task_type="CAUSAL_LM",
    )


def load_training_model(runtime: dict, lora_config=None):
    """兼容 HF 模型加载接口；Unsloth 必须成对加载 tokenizer/model。"""
    if runtime.get("backend", "hf") != "hf":
        raise ValueError(
            "Unsloth 请使用 load_training_components，不能重复 PEFT 包装。"
        )
    initialize_backend(runtime)
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    from lm_trainer.model.quantization import build_quantization_config

    dtype = torch.bfloat16 if runtime.get("compute_dtype") == "bf16" else torch.float16
    quantization_config = build_quantization_config(runtime["quantization_mode"])
    if runtime["quantization_mode"] == "4bit":
        quantization_config.bnb_4bit_compute_dtype = dtype
    load_kwargs = {
        "device_map": {"": torch.cuda.current_device()}
        if torch.cuda.is_available()
        else "auto",
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(
        runtime["model_name_or_path"], **load_kwargs
    )
    checkpointing = runtime.get("gradient_checkpointing", "standard") != "none"
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=checkpointing
        )
    model.config.use_cache = False
    if checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return get_peft_model(model, lora_config or create_lora_config(runtime))


def load_training_components(runtime: dict, *, initialize=None):
    if runtime.get("backend", "hf") == "hf":
        model = load_training_model(runtime)
        from lm_trainer.model.loader import load_tokenizer

        return model, load_tokenizer(runtime["model_name_or_path"], use_fast=False)

    (initialize or initialize_backend)(runtime)
    import torch
    from unsloth import FastLanguageModel

    if runtime["lora_dropout"] != 0 or runtime["lora_bias"] != "none":
        warnings.warn(
            "Unsloth 快速 LoRA 路径建议 dropout=0、bias=none；不自动修改用户参数。",
            stacklevel=2,
        )
    checkpointing = {"standard": True, "unsloth": "unsloth", "none": False}[
        runtime["gradient_checkpointing"]
    ]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=runtime["model_name_or_path"],
        max_seq_length=runtime["max_length"],
        dtype=torch.bfloat16 if runtime["compute_dtype"] == "bf16" else torch.float16,
        load_in_4bit=runtime["quantization_mode"] == "4bit",
        load_in_8bit=False,
        device_map={"": torch.cuda.current_device()},
        trust_remote_code=False,
        local_files_only=True,
        use_exact_model_name=True,
        use_gradient_checkpointing=checkpointing,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=runtime["lora_r"],
        lora_alpha=runtime["lora_alpha"],
        target_modules=runtime["lora_target_modules"],
        lora_dropout=runtime["lora_dropout"],
        bias=runtime["lora_bias"],
        use_gradient_checkpointing=checkpointing,
        random_state=runtime["seed"],
        max_seq_length=runtime["max_length"],
    )
    FastLanguageModel.for_training(model, use_gradient_checkpointing=checkpointing)
    model.config.use_cache = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer
