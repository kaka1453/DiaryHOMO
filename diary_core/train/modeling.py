"""Compatibility interface for lm_trainer.train.modeling."""

from lm_trainer.train.modeling import (
    create_lora_config,
    load_training_model,
)

import json
from diary_core.config.train_config import (
    build_train_parser,
    build_train_runtime_config,
)


from lm_trainer.train import modeling as core
from diary_core.train.bootstrap import initialize_backend


def load_training_components(runtime):
    return core.load_training_components(runtime, initialize=initialize_backend)


def main() -> None:
    parser = build_train_parser()
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()
    runtime = build_train_runtime_config(args)
    summary = {
        key: runtime[key]
        for key in (
            "backend",
            "quantization_mode",
            "lora_r",
            "lora_alpha",
            "lora_target_modules",
        )
    }
    summary["load_model"] = args.load_model
    summary["lora_target_modules"] = sorted(runtime["lora_target_modules"])
    if args.load_model:
        model, _ = load_training_components(runtime)
        summary["model_class"] = model.__class__.__name__
        summary["trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
