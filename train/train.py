import sys

sys.dont_write_bytecode = True
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diary_core.config.common import (
    dump_runtime_config,
    load_yaml_config,
    resolve_path,
    str2bool,
)
from diary_core.config.assets import build_or_exit, configure_caches, product_root
from diary_core.config.train_config import (
    ALLOWED_QUANTIZATION_MODES,
    BOOL_FIELDS,
    CLI_OVERRIDE_SPECS,
    DEFAULT_MODEL_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
    FLOAT_FIELDS,
    INT_FIELDS,
    MODEL_CONFIG_FIELDS,
    NON_NEGATIVE_FIELDS,
    PATH_FIELDS,
    POSITIVE_FIELDS,
    TRAIN_CONFIG_FIELDS,
    apply_runtime_casts,
    collect_config_values,
    parse_target_modules,
    validate_runtime_values,
)
from diary_core.config.train_config import (
    build_train_parser as build_parser,
)
from diary_core.config.train_config import (
    build_train_runtime_config as build_runtime_config,
)
from diary_core.train.bootstrap import dispatch_unsloth as _dispatch_unsloth

# Legacy exports are resolved on demand so backend selection runs before HF imports.
_LAZY_EXPORTS = {
    "load_tokenizer": ("diary_core.model.loader", "load_tokenizer"),
    "normalize_quantization_mode": (
        "diary_core.model.quantization",
        "normalize_quantization_mode",
    ),
    "train": ("diary_core.train.engine", "train"),
    "simulate_pipeline": ("diary_core.train.engine", "simulate_pipeline"),
    "load_training_model": ("diary_core.train.modeling", "load_training_model"),
    "create_lora_config": ("diary_core.train.modeling", "create_lora_config"),
    "plot_metrics": ("diary_core.train.metrics", "plot_metrics"),
    "set_reproducibility": ("diary_core.train.reproducibility", "set_reproducibility"),
    "compute_schedule": ("diary_core.train.schedule", "compute_schedule"),
    "DataCollatorForCausalLMWith8xPadding": (
        "diary_core.train.data",
        "DataCollatorForCausalLMWith8xPadding",
    ),
    "SortedBatchSampler": ("diary_core.train.data", "SortedBatchSampler"),
    "build_data_loader": ("diary_core.train.data", "build_data_loader"),
    "load_training_dataset": ("diary_core.train.data", "load_training_dataset"),
    "save_runtime_snapshot": ("diary_core.train.io", "save_runtime_snapshot"),
    "validate_paths": ("diary_core.train.io", "validate_paths"),
    "load_model": ("diary_core.train.modeling", "load_training_model"),
}

__all__ = [
    "ALLOWED_QUANTIZATION_MODES",
    "BOOL_FIELDS",
    "CLI_OVERRIDE_SPECS",
    "DEFAULT_MODEL_CONFIG_PATH",
    "DEFAULT_TRAIN_CONFIG_PATH",
    "FLOAT_FIELDS",
    "INT_FIELDS",
    "MODEL_CONFIG_FIELDS",
    "NON_NEGATIVE_FIELDS",
    "PATH_FIELDS",
    "POSITIVE_FIELDS",
    "PROJECT_ROOT",
    "TRAIN_CONFIG_FIELDS",
    "Path",
    "apply_runtime_casts",
    "build_parser",
    "build_quantization_config",
    "build_runtime_config",
    "collect_config_values",
    "dump_runtime_config",
    "load_yaml_config",
    "main",
    "parse_target_modules",
    "resolve_path",
    "str2bool",
    "sys",
    "validate_runtime_values",
]


__all__.extend(_LAZY_EXPORTS)


def __getattr__(name):
    from importlib import import_module

    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module, attribute = _LAZY_EXPORTS[name]
    return getattr(import_module(module), attribute)


def build_quantization_config(runtime):
    from diary_core.model.quantization import build_quantization_config as build

    return build(
        runtime.get("quantization_mode") if isinstance(runtime, dict) else runtime
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    runtime = build_or_exit(parser, build_runtime_config, args)

    if args.dry_run:
        print("Train 配置检查通过。")
        print(dump_runtime_config(runtime))
        return

    configure_caches(product_root(runtime["asset_root"]))
    if not args.simulate_only and _dispatch_unsloth(runtime, [__file__, *sys.argv[1:]]):
        return

    from diary_core.train.engine import simulate_pipeline, train

    if args.simulate_only:
        simulate_pipeline(runtime)
        return

    train(runtime)


if __name__ == "__main__":
    main()
