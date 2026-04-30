from __future__ import annotations

import argparse

import torch

from diary_core.config.common import dump_runtime_config
from diary_core.config.infer_config import build_batch_parser, build_batch_runtime_config
from diary_core.infer.diary_runtime import DiaryRuntime
from diary_core.infer.output_bundle import prepare_output_bundle, write_parameters
from diary_core.infer.prompt_io import format_markdown_block, load_prompts, write_results
from diary_core.model.loader import load_model_and_tokenizer


def run_generation(runtime: dict) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    prepare_output_bundle(runtime)
    write_parameters(runtime)

    prompts = load_prompts(runtime["input_file"])
    if runtime["print_prompts"]:
        print(f"读取到 {len(prompts)} 条 prompts。")

    tokenizer, model = load_model_and_tokenizer(runtime)
    diary_runtime = DiaryRuntime(runtime, tokenizer, model)

    results = []
    batch_size = runtime["batch_size"]
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]

        for prompt in batch:
            result = diary_runtime.generate(prompt)
            block = format_markdown_block(
                len(results) + 1,
                prompt,
                result.final_text,
                guard=result.guard,
                debug_dir=result.debug_dir,
                audit_config=runtime.get("audit"),
            )
            if runtime["print_prompts"]:
                print(block)
            results.append(block)

    write_results(results, runtime["output_file"])
    print(f"\n8bit 推理完成，共 {len(results)} 篇，输出目录: {runtime['output_run_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference module smoke test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    argv = ["--dry-run"]
    if args.config:
        argv.extend(["--config", args.config])
    runtime = build_batch_runtime_config(build_batch_parser().parse_args(argv))
    print(dump_runtime_config(runtime))
    if not args.dry_run:
        run_generation(runtime)


if __name__ == "__main__":
    main()
