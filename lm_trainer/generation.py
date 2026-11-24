from __future__ import annotations

import torch


def normalize_stop_sequences(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item for item in value.split(",") if item]
    return [str(item) for item in value if str(item)]


def trim_stop_sequences(text: str, stop_sequences: list[str] | None = None) -> str:
    cut = len(text)
    for sequence in normalize_stop_sequences(stop_sequences):
        position = text.find(sequence)
        if position != -1:
            cut = min(cut, position)
    return text[:cut].strip()


def generation_kwargs(runtime: dict, tokenizer) -> dict:
    return {
        "max_new_tokens": runtime["max_new_tokens"],
        "temperature": runtime["temperature"],
        "top_p": runtime["top_p"],
        "top_k": runtime["top_k"],
        "repetition_penalty": runtime["repetition_penalty"],
        "do_sample": True,
        "num_beams": runtime["num_beams"],
        "pad_token_id": tokenizer.pad_token_id,
    }


def decode_batch(prompts: list[str], tokenizer, model, runtime: dict) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(runtime["device"])

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs(runtime, tokenizer))

    input_len = inputs["input_ids"].shape[-1]
    new_tokens = outputs[:, input_len:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def generate_batch(prompts: list[str], tokenizer, model, runtime: dict) -> list[str]:
    texts = decode_batch(prompts, tokenizer, model, runtime)
    return [
        trim_stop_sequences(text, runtime.get("stop_sequences", [])) for text in texts
    ]
