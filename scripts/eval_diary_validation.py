from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diary_core.config.common import dump_runtime_config, resolve_path, str2bool
from diary_core.config.infer_config import build_batch_parser, build_batch_runtime_config
from diary_core.eval.metrics import build_eval_result, summarize_eval_results, write_eval_artifacts
from diary_core.eval.validation import load_validation_samples, summarize_samples
from diary_core.infer.audit import append_audit_record, build_audit_record, write_audit_artifacts
from diary_core.infer.diary_runtime import DiaryRuntime
from diary_core.infer.output_bundle import OUTPUT_MD_FILENAME, PARAMETERS_FILENAME, write_parameters
from diary_core.infer.prompt_io import format_markdown_block, write_results
from diary_core.model.loader import load_model_and_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run diary generation regression validation.")
    parser.add_argument("--val-file", default="data/val/diary_validation_prompts.jsonl")
    parser.add_argument("--config", default="config/generate.yaml")
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default="generate/validation")
    parser.add_argument("--output-run-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-debug", default="false")
    parser.add_argument("--prompt-debug-output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_eval_runtime(args: argparse.Namespace) -> dict:
    batch_args = build_batch_parser().parse_args(
        [
            "--config",
            args.config,
            "--input-file",
            args.val_file,
            "--output-root",
            args.output_root,
            "--output-name",
            "validation",
            "--print-prompts",
            "false",
        ]
    )
    runtime = build_batch_runtime_config(batch_args)
    if args.model_name_or_path:
        runtime["model_name_or_path"] = str(resolve_path(args.model_name_or_path))
    if args.checkpoint_dir:
        runtime["checkpoint_dir"] = str(resolve_path(args.checkpoint_dir))
    if args.device:
        runtime["device"] = args.device
    runtime["print_prompts"] = False
    runtime["eval"] = {
        "enabled": True,
        "val_file": str(resolve_path(args.val_file)),
        "limit": args.limit,
        "note": "Generation regression validation only. Do not use data/val as training data.",
    }
    prompt_debug = dict(runtime.get("prompt_debug") or {})
    prompt_debug["enabled"] = str2bool(args.prompt_debug)
    if args.prompt_debug_output_dir:
        prompt_debug["output_dir"] = str(resolve_path(args.prompt_debug_output_dir))
    runtime["prompt_debug"] = prompt_debug

    run_dir = resolve_path(args.output_run_dir) if args.output_run_dir else create_eval_run_dir(args.output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime["output_run_dir"] = str(run_dir)
    runtime["output_file"] = str(run_dir / OUTPUT_MD_FILENAME)
    runtime["parameters_file"] = str(run_dir / PARAMETERS_FILENAME)
    return runtime


def create_eval_run_dir(output_root: str | Path) -> Path:
    parent = resolve_path(output_root)
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for _ in range(20):
        run_dir = parent / f"{timestamp}_{uuid4().hex[:6]}"
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"无法创建验证输出目录: {parent}")


def run_validation(args: argparse.Namespace) -> dict:
    samples = load_validation_samples(args.val_file, limit=args.limit)
    runtime = build_eval_runtime(args)
    write_parameters(runtime)

    tokenizer, model = load_model_and_tokenizer(runtime)
    diary_runtime = DiaryRuntime(runtime, tokenizer, model)

    markdown_blocks = []
    audit_records = []
    eval_results = []
    for index, sample in enumerate(samples, start=1):
        result = diary_runtime.generate(sample.prompt)
        sample_dict = sample.to_dict()
        audit_record = build_audit_record(
            index=index,
            result=result,
            sample=sample_dict,
            audit_config=runtime.get("audit"),
        )
        append_audit_record(runtime["output_run_dir"], audit_record, runtime.get("audit"))
        audit_records.append(audit_record)

        eval_result = build_eval_result(sample, result.final_text, result.guard)
        eval_results.append(eval_result)

        markdown_blocks.append(
            format_markdown_block(
                index,
                f"[{sample.id}/{sample.category}] {sample.prompt}",
                result.final_text,
                guard=result.guard,
                debug_dir=result.debug_dir,
                audit_config=runtime.get("audit"),
            )
        )

    write_results(markdown_blocks, runtime["output_file"])
    audit_summary = write_audit_artifacts(runtime["output_run_dir"], audit_records, runtime.get("audit"))
    eval_summary = summarize_eval_results(eval_results)
    write_eval_artifacts(runtime["output_run_dir"], eval_results, eval_summary)

    payload = {
        "output_run_dir": runtime["output_run_dir"],
        "count": len(samples),
        "audit_summary": audit_summary,
        "eval_summary": eval_summary,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    samples = load_validation_samples(args.val_file, limit=args.limit)
    if args.dry_run:
        runtime = build_eval_runtime(args)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "validation": summarize_samples(samples),
                    "runtime": json.loads(dump_runtime_config(runtime)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    run_validation(args)


if __name__ == "__main__":
    main()
