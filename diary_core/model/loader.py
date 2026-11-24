"""Compatibility interface for lm_trainer.model.loader."""

from lm_trainer.model.loader import (
    load_tokenizer,
    load_base_model,
    load_model_and_tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    PeftModel,
)

import argparse
import json
from diary_core.config.infer_config import (
    build_batch_parser,
    build_batch_runtime_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model loader smoke test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    argv = ["--dry-run"]
    if args.config:
        argv.extend(["--config", args.config])
    runtime = build_batch_runtime_config(build_batch_parser().parse_args(argv))
    summary = {
        "model_name_or_path": runtime["model_name_or_path"],
        "checkpoint_dir": runtime["checkpoint_dir"],
        "quantization_mode": runtime.get("quantization_mode", "8bit"),
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    tokenizer, model = load_model_and_tokenizer(runtime)
    print(
        json.dumps(
            {
                "tokenizer_vocab": len(tokenizer),
                "model_class": model.__class__.__name__,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
