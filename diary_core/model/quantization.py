from __future__ import annotations

import argparse
import json

import torch
from transformers import BitsAndBytesConfig


def normalize_quantization_mode(value: str | None) -> str:
    normalized = str(value or "8bit").strip().lower()
    aliases = {
        "4": "4bit",
        "4bit": "4bit",
        "nf4": "4bit",
        "8": "8bit",
        "8bit": "8bit",
        "int8": "8bit",
        "none": "none",
        "off": "none",
        "false": "none",
        "fp16": "none",
        "bf16": "none",
        "full": "none",
        "16": "none",
    }
    if normalized not in aliases:
        raise ValueError(f"不支持的 quantization_mode: {value}")
    return aliases[normalized]


def build_quantization_config(mode: str | None = "8bit"):
    normalized = normalize_quantization_mode(mode)
    if normalized == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if normalized == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantization config smoke test")
    parser.add_argument("--mode", default="8bit")
    args = parser.parse_args()
    config = build_quantization_config(args.mode)
    print(json.dumps({"mode": normalize_quantization_mode(args.mode), "enabled": config is not None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
