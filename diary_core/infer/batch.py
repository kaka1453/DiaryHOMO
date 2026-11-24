from __future__ import annotations

from pathlib import Path

from diary_core.config.assets import build_or_exit, configure_caches, runtime_output
from diary_core.config.common import dump_runtime_config
from diary_core.config.infer_config import build_batch_parser, build_batch_runtime_config, inference_summary
from diary_core.infer.output_bundle import prepare_output_bundle, write_parameters
from diary_core.infer.prompt_io import format_markdown_block, load_prompts, write_results


def run_generation(runtime: dict) -> None:
    import torch

    from diary_core.infer.audit import build_audit_record, append_audit_record, write_audit_artifacts
    from diary_core.infer.diary_runtime import DiaryRuntime
    from diary_core.model.loader import load_model_and_tokenizer

    torch.backends.cuda.matmul.allow_tf32 = True
    prepare_output_bundle(runtime)
    write_parameters(runtime)

    prompts = load_prompts(runtime["input_file"])
    if runtime["print_prompts"]:
        print(f"读取到 {len(prompts)} 条 prompts。")

    tokenizer, model = load_model_and_tokenizer(runtime)
    diary_runtime = DiaryRuntime(runtime, tokenizer, model)

    results = []
    audit_records = []
    batch_size = runtime["batch_size"]
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]

        for prompt in batch:
            result = diary_runtime.generate(prompt)
            audit_record = build_audit_record(
                index=len(results) + 1,
                result=result,
                audit_config=runtime.get("audit"),
            )
            append_audit_record(runtime["output_run_dir"], audit_record, runtime.get("audit"))
            audit_records.append(audit_record)
            block = format_markdown_block(
                len(results) + 1,
                prompt,
                result.final_text,
                guard=result.guard,
                debug_dir=result.debug_dir,
                audit_config=runtime.get("audit"),
            )
            results.append(block)

    write_results(results, runtime_output(Path(runtime["asset_root"]), runtime["output_file"]))
    write_audit_artifacts(runtime["output_run_dir"], audit_records, runtime.get("audit"))
    print(f"\n8bit 推理完成，共 {len(results)} 篇，输出目录: {runtime['output_run_dir']}")


def main() -> None:
    parser = build_batch_parser()
    args = parser.parse_args()
    runtime = build_or_exit(parser, build_batch_runtime_config, args)
    if args.dry_run:
        print(dump_runtime_config(inference_summary(runtime)))
        return
    configure_caches(Path(runtime["asset_root"]))
    run_generation(runtime)


if __name__ == "__main__":
    main()
