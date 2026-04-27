from __future__ import annotations

import argparse
import json

import torch


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


def generate_batch(prompts: list[str], tokenizer, model, runtime: dict) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(runtime["device"])

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs(runtime, tokenizer))

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def extract_ai_reply(text: str) -> str:
    return text.split("AI:")[-1].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation helper smoke test")
    parser.add_argument("--prompt", default="AI: hello")
    args = parser.parse_args()
    print(json.dumps({"reply": extract_ai_reply(args.prompt)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

