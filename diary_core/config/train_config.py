"""Product training entry point for the shared training configuration."""

from pathlib import Path

from diary_core.config.common import (
    PROJECT_ROOT,
    dump_runtime_config,
    load_yaml_config,
    require_keys,
    resolve_path,
    str2bool,
)
from diary_core.config.assets import build_or_exit, external_base_model, product_root
from lm_trainer.model.quantization import normalize_quantization_mode
from lm_trainer.config import train_config as core
from lm_trainer.config.common import asset_path
from lm_trainer.config.train_config import (
    ALLOWED_QUANTIZATION_MODES,
    BOOL_FIELDS,
    CLI_OVERRIDE_SPECS,
    FLOAT_FIELDS,
    INT_FIELDS,
    MODEL_CONFIG_FIELDS,
    NON_NEGATIVE_FIELDS,
    OPTIONAL_DEFAULTS,
    PATH_FIELDS,
    POSITIVE_FIELDS,
    RUNTIME_FIELDS,
    TRAIN_CONFIG_FIELDS,
    apply_cli_overrides,
    collect_config_values,
    parse_target_modules,
    validate_runtime_values,
)

DEFAULT_TRAIN_CONFIG_PATH = "config/train.yaml"
DEFAULT_MODEL_CONFIG_PATH = "config/model.yaml"


def build_train_parser():
    parser = core.build_train_parser()
    next(action for action in parser._actions if action.dest == "asset_root").required = True
    return parser


def apply_runtime_casts(runtime, asset_root):
    core.resolve_asset_runtime_paths(
        runtime, product_root(asset_root), eval_file=runtime.get("eval_file_path"),
        external_model=None, dry_run=False,
    )
    core.cast_runtime_values(runtime)


def build_train_runtime_config(args):
    root = product_root(args.asset_root)
    args.asset_root = str(root)
    if args.external_base_model is not None:
        external_base_model(args.external_base_model)
    runtime = core.build_train_runtime_config(args)
    if args.external_base_model is None:
        model = Path(runtime["model_name_or_path"]).relative_to(root)
        asset_path(root, str(model), "model_name_or_path", "assets", directory=True)
    return {**runtime, "asset_root": str(root)}


build_parser = build_train_parser
build_runtime_config = build_train_runtime_config


def main():
    parser = build_parser()
    print(dump_runtime_config(build_or_exit(parser, build_runtime_config, parser.parse_args())))


if __name__ == "__main__":
    main()
