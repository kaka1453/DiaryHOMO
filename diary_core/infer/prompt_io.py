from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_prompts(input_file: str) -> list[str]:
    with open(input_file, "r", encoding="utf-8") as fh:
        prompts = [line.strip() for line in fh if line.strip()]
    if not prompts:
        raise ValueError(f"输入文件中未读取到有效 prompt: {input_file}")
    return prompts


def format_markdown_block(index: int, prompt: str, text: str) -> str:
    return f"## 第{index}篇\n\n引言: {prompt}\n\n{text}\n\n---\n"


def write_results(results: list[str], output_file: str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.writelines(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt IO smoke test")
    parser.add_argument("--input-file", required=True)
    args = parser.parse_args()
    prompts = load_prompts(args.input_file)
    print(json.dumps({"count": len(prompts), "first": prompts[0]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

