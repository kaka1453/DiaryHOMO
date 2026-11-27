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


def load_training_model(runtime: dict, lora_config=None, *, adapter_path=None):
    """兼容 HF 模型加载接口；Unsloth 必须成对加载 tokenizer/model。"""
    if runtime.get("backend", "hf") != "hf":
        raise ValueError(
            "Unsloth 请使用 load_training_components，不能重复 PEFT 包装。"
        )
    initialize_backend(runtime)
    import torch
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    from lm_trainer.model.quantization import build_quantization_config

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}.get(
        runtime.get("compute_dtype"), torch.float16
    )
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
    if runtime.get("data_mode", "legacy") != "legacy":
        load_kwargs.update(local_files_only=True, trust_remote_code=False)
    if runtime.get("init_adapter"):
        load_kwargs.update(use_safetensors=True, output_loading_info=True)
        if quantization_config is not None and not torch.cuda.is_available():
            raise ValueError("Quantized stage training requires CUDA.")
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(
        runtime["model_name_or_path"], **load_kwargs
    )
    if runtime.get("init_adapter"):
        model, loading_info = model
        if any(
            loading_info.get(k)
            for k in (
                "missing_keys",
                "mismatched_keys",
                "unexpected_keys",
                "error_msgs",
            )
        ):
            raise ValueError(
                "Local base weight state does not match the configured model architecture."
            )
    checkpointing = runtime.get("gradient_checkpointing", "standard") != "none"
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )
        if runtime.get("init_adapter") and runtime["quantization_mode"] == "4bit":
            from bitsandbytes.nn import Linear4bit

            # k-bit preparation promotes biases to FP32. BnB 0.50 otherwise
            # mutates a quantized linear's frozen bias on its first forward.
            # Apply exactly that dtype conversion as part of base preparation,
            # before taking the immutable-base fingerprint or installing LoRA.
            with torch.no_grad():
                for layer in model.modules():
                    if isinstance(layer, Linear4bit) and layer.bias is not None:
                        if (
                            layer.compute_dtype != dtype
                            or not layer.compute_type_is_set
                        ):
                            raise ValueError(
                                "4-bit base must have a fixed compute dtype."
                            )
                        layer.bias.data = layer.bias.detach().to(dtype)
    model.config.use_cache = False
    if checkpointing and quantization_config is None:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if runtime.get("init_adapter"):
        validate_targets(model, runtime["lora_target_modules"])
    if adapter_path is not None:
        # HF new adapters use PEFT's FP32 adapter policy. Match it *before*
        # copying the state so a BF16/FP16 base cannot round the parent weights.
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=True,
            autocast_adapter_dtype=True,
        )
        verify_loaded_weights(model, adapter_path)
        return model
    return get_peft_model(model, lora_config or create_lora_config(runtime))


def load_stage_tokenizer(runtime):
    from lm_trainer.model.loader import load_tokenizer

    explicit = runtime.get("tokenizer_name_or_path")
    if explicit:
        from pathlib import Path

        from transformers import PreTrainedTokenizerFast

        if not (Path(explicit) / "tokenizer.json").is_file():
            raise ValueError("Explicit tokenizer directory requires tokenizer.json.")
        # A serialized fast backend is the ABI for an explicitly pinned
        # tokenizer; don't rebuild it from architecture/version defaults.
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            explicit, local_files_only=True
        )
    else:
        tokenizer = load_tokenizer(
            runtime["model_name_or_path"], use_fast=True, local_files_only=True
        )
    tokenizer.padding_side = "right"
    return tokenizer


def load_training_components(
    runtime: dict, *, initialize=None, adapter_path=None, tokenizer=None
):
    if runtime.get("backend", "hf") == "hf":
        model = load_training_model(runtime, adapter_path=adapter_path)
        if tokenizer is not None:
            return model, tokenizer
        from lm_trainer.model.loader import load_tokenizer

        structured = runtime.get("data_mode", "legacy") != "legacy"
        return model, load_tokenizer(
            runtime["model_name_or_path"],
            use_fast=structured,
            local_files_only=structured,
        )

    if runtime.get("init_adapter") not in (None, "new") or adapter_path is not None:
        raise ValueError(
            "Unsloth continue_stage/resume unsupported; use HF for both stages."
        )
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
    if runtime.get("init_adapter"):
        validate_targets(model, runtime["lora_target_modules"])
        if runtime.get("tokenizer_name_or_path"):
            tokenizer = load_stage_tokenizer(runtime)
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


def validate_targets(model, targets):
    names = [name for name, _ in model.named_modules()]
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate LoRA target modules.")
    missing = [
        t for t in targets if not any(n == t or n.endswith(f".{t}") for n in names)
    ]
    if missing:
        raise ValueError(
            "One or more LoRA target_modules do not exist on the base model."
        )


def adapter_tensors(model):
    from peft import get_peft_model_state_dict

    return get_peft_model_state_dict(
        model, adapter_name="default", save_embedding_layers=False
    )


def verify_loaded_weights(model, directory):
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    from lm_trainer.train.artifacts import WEIGHTS

    expected = load_file(Path(directory) / WEIGHTS, device="cpu")
    actual = adapter_tensors(model)
    if actual.keys() != expected.keys() or any(
        t.shape != expected[k].shape
        or t.dtype != expected[k].dtype
        or not torch.equal(t.detach().cpu(), expected[k])
        for k, t in actual.items()
    ):
        raise ValueError(
            "Loaded LoRA tensor names, shapes, dtypes or bytes differ from parent."
        )


def tensor_fingerprint(tensors):
    import hashlib

    import torch

    from lm_trainer.train.artifacts import digest

    fingerprint = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        fingerprint.update(digest([name, str(value.dtype), list(value.shape)]).encode())
        fingerprint.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return fingerprint.hexdigest()


def check_trainable(model, runtime):
    from lm_trainer.train.artifacts import lora_identity

    config = model.peft_config["default"]
    if lora_identity(runtime) != {
        "r": config.r,
        "alpha": config.lora_alpha,
        "dropout": config.lora_dropout,
        "bias": config.bias,
        "target_modules": sorted(config.target_modules),
    }:
        raise ValueError(
            "Loaded LoRA structure does not match target modules, rank and alpha."
        )
    active = model.active_adapters
    if active != ["default"] or set(model.peft_config) != {"default"}:
        raise ValueError("Stage requires exactly one active, unmerged default adapter.")
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if (
        not trainable
        or any(
            not any(f".{part}.default." in n for part in ("lora_A", "lora_B"))
            for n in trainable
        )
        or len(trainable) != len(adapter_tensors(model))
    ):
        raise ValueError(
            "Only LoRA A/B tensors may be trainable; the base must be frozen."
        )
    if any(getattr(m, "merged", False) for m in model.modules()):
        raise ValueError("Merged adapters are prohibited in stage training.")
    dtypes = sorted({str(p.dtype) for p in trainable.values()})
    if runtime["backend"] == "hf" and dtypes != ["torch.float32"]:
        raise ValueError("HF staged LoRA requires FP32 trainable adapter tensors.")
    total = sum(p.numel() for p in model.parameters())
    count = sum(p.numel() for p in trainable.values())
    return {
        "trainable": count,
        "adapter_dtypes": dtypes,
        "total_parameter_elements": total,
        "fraction": count / total,
        "target_modules": sorted(runtime["lora_target_modules"]),
    }


def frozen_fingerprint(model):
    frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    quant_state = {}
    for name, param in frozen.items():
        if getattr(param, "quant_state", None) is not None:
            quant_state.update(
                {
                    f"{name}.quant_state.{k}": v
                    for k, v in param.quant_state.as_dict(packed=True).items()
                }
            )
    return tensor_fingerprint({**frozen, **quant_state})
