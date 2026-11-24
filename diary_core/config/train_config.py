"""Legacy configuration defaults and project-relative asset paths."""

from diary_core.config.common import (
    PROJECT_ROOT,
    dump_runtime_config,
    load_yaml_config,
    require_keys,
    resolve_path,
    str2bool,
)
from lm_trainer.model.quantization import normalize_quantization_mode
from lm_trainer.config import train_config as core
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

DEFAULT_TRAIN_CONFIG_PATH = PROJECT_ROOT / "config/train.yaml"
DEFAULT_MODEL_CONFIG_PATH = PROJECT_ROOT / "config/model.yaml"


def build_train_parser():
    return core.build_train_parser(
        str(DEFAULT_TRAIN_CONFIG_PATH), str(DEFAULT_MODEL_CONFIG_PATH)
    )


def apply_runtime_casts(runtime):
    return core.apply_runtime_casts(runtime, PROJECT_ROOT)


def build_train_runtime_config(args):
    return core.build_train_runtime_config(args, path_root=PROJECT_ROOT)


build_parser = build_train_parser
build_runtime_config = build_train_runtime_config


def main():
    print(dump_runtime_config(build_runtime_config(build_parser().parse_args())))


if __name__ == "__main__":
    main()
