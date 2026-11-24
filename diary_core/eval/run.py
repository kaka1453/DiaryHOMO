"""Run product generation regression against an asset-root validation set."""

from pathlib import Path

from diary_core.config.assets import build_or_exit, configure_caches, runtime_output
from diary_core.config.common import dump_runtime_config
from diary_core.config.infer_config import build_batch_parser, build_batch_runtime_config, inference_summary
from diary_core.eval.validation import load_validation_samples
from diary_core.infer.output_bundle import prepare_output_bundle, write_parameters
from lm_trainer.config.common import asset_path


def build_parser():
    parser = build_batch_parser()
    parser.description = "Run product generation regression validation"
    parser.add_argument("--val-file", default="data/eval/diary_validation_prompts.jsonl")
    parser.add_argument("--output-run-dir", help="Run directory below the asset root's runs/ directory")
    parser.add_argument("--limit", type=int)
    return parser


def build_eval_runtime(args):
    if args.input_file is not None:
        raise ValueError("Use --val-file for evaluation input")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    limit, run_dir = args.limit, args.output_run_dir
    fields = {key: value for key, value in vars(args).items() if key not in {"val_file", "limit", "output_run_dir"}}
    fields.update(
        input_file=args.val_file,
        output_root=args.output_root or "runs/evaluation",
        output_name=args.output_name or "validation",
        print_prompts=False,
    )
    args = type(args)(**fields)
    runtime = build_batch_runtime_config(args, evaluation=True)
    if run_dir is not None:
        runtime["output_run_dir"] = str(asset_path(
            Path(runtime["asset_root"]), run_dir, "--output-run-dir", "runs", directory=True, required=False,
        ))
    samples = load_validation_samples(runtime["input_file"], limit=limit)
    return runtime, samples


def run_validation(runtime, samples):
    from diary_core.eval.metrics import build_eval_result, summarize_eval_results, write_eval_artifacts
    from diary_core.infer.audit import append_audit_record, build_audit_record, write_audit_artifacts
    from diary_core.infer.diary_runtime import DiaryRuntime
    from diary_core.infer.prompt_io import format_markdown_block, write_results
    from diary_core.model.loader import load_model_and_tokenizer

    prepare_output_bundle(runtime)
    write_parameters(runtime)
    tokenizer, model = load_model_and_tokenizer(runtime)
    diary = DiaryRuntime(runtime, tokenizer, model)
    markdown, audits, results = [], [], []
    for index, sample in enumerate(samples, start=1):
        result = diary.generate(sample.prompt)
        record = build_audit_record(index=index, result=result, sample=sample.to_dict(), audit_config=runtime["audit"])
        append_audit_record(runtime["output_run_dir"], record, runtime["audit"])
        audits.append(record)
        results.append(build_eval_result(sample, result.final_text, result.guard))
        markdown.append(format_markdown_block(
            index, f"[{sample.id}/{sample.category}] {sample.prompt}", result.final_text,
            guard=result.guard, debug_dir=result.debug_dir, audit_config=runtime["audit"],
        ))
    root = Path(runtime["asset_root"])
    write_results(markdown, runtime_output(root, runtime["output_file"]))
    write_audit_artifacts(runtime["output_run_dir"], audits, runtime["audit"])
    write_eval_artifacts(runtime["output_run_dir"], results, summarize_eval_results(results))


def main():
    parser = build_parser()
    args = parser.parse_args()
    runtime, samples = build_or_exit(parser, build_eval_runtime, args)
    if not args.dry_run:
        configure_caches(Path(runtime["asset_root"]))
        run_validation(runtime, samples)
    print(dump_runtime_config({"count": len(samples), "runtime": inference_summary(runtime)}))


if __name__ == "__main__":
    main()
