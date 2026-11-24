"""Compatibility interface for lm_trainer.model.loader."""

import json
from diary_core.config.infer_config import (
    build_batch_parser,
    build_batch_runtime_config,
)


__all__ = [
    "load_tokenizer", "load_base_model", "load_model_and_tokenizer",
    "AutoModelForCausalLM", "AutoTokenizer", "PeftModel",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from lm_trainer.model import loader

    return getattr(loader, name)


def main() -> None:
    from pathlib import Path

    from diary_core.config.assets import build_or_exit, configure_caches

    parser = build_batch_parser()
    args = parser.parse_args()
    runtime = build_or_exit(parser, build_batch_runtime_config, args)
    summary = {key: runtime[key] for key in ("model_name_or_path", "checkpoint_dir", "quantization_mode")}
    if not args.dry_run:
        configure_caches(Path(runtime["asset_root"]))
        from lm_trainer.model.loader import load_model_and_tokenizer

        tokenizer, model = load_model_and_tokenizer(runtime)
        summary.update(tokenizer_vocab=len(tokenizer), model_class=model.__class__.__name__)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
