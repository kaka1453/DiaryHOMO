import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diary_core.config.common import dump_runtime_config, load_yaml_config, resolve_path, str2bool
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
    build_train_parser as build_parser,
    build_train_runtime_config as build_runtime_config,
    collect_config_values,
    parse_target_modules,
    validate_runtime_values,
)
from diary_core.model.loader import load_tokenizer
from diary_core.model.quantization import (
    build_quantization_config as _build_quantization_config,
    normalize_quantization_mode,
)
from diary_core.train.data import (
    DataCollatorForCausalLMWith8xPadding,
    SortedBatchSampler,
    build_data_loader,
    load_training_dataset,
)
from diary_core.train.engine import simulate_pipeline, train
from diary_core.train.io import save_runtime_snapshot, validate_paths
from diary_core.train.metrics import plot_metrics
from diary_core.train.modeling import create_lora_config, load_training_model
from diary_core.train.reproducibility import set_reproducibility
from diary_core.train.schedule import compute_schedule


def build_quantization_config(runtime):
    mode = runtime.get("quantization_mode") if isinstance(runtime, dict) else runtime
    return _build_quantization_config(mode)


load_model = load_training_model


def main():
    parser = build_parser()
    args = parser.parse_args()
    runtime = build_runtime_config(args)

    if args.dry_run:
        print("Train 配置检查通过。")
        print(dump_runtime_config(runtime))
        return

    if args.simulate_only:
        simulate_pipeline(runtime)
        return

    train(runtime)


if __name__ == "__main__":
    main()
