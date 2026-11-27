# Causal LM trainer 1.0.0

Run `python -m lm_trainer --help` from the directory containing this package. For training, supply `--config TRAIN.yaml --model-config MODEL.yaml`. `--dry-run` checks configuration and paths, `--validate-release` validates corpus integrity and permissions without a model, and `--simulate-only` tokenizes input and constructs a loader without loading model weights. Relative model, JSON/JSONL data and output paths are anchored at the training YAML directory. The CLI defaults its library caches to `.cache` inside the explicit output directory. A fresh output directory is required. Supply a local base model. The legacy entrance accepts one JSON/JSONL file with a `text` string per record. Structured `release` and `fixture` modes require a manifest and explicit `causal_text` (CPT) or `assistant_only_messages` (SFT) objective. See [Task A data and batch contracts](TASK_A_CONTRACT.md) and the [fictional vision-support validation](../examples/vision_support/TASK_A_VALIDATION.md).

`requirements.txt` records the reference HF dependencies. The default backend is `hf`, precision is `fp16`, and automatic checkpointing selects `standard`. The optional `unsloth` backend uses its own compatible interpreter. Set `LM_TRAINER_UNSLOTH_PYTHON` before running the same CLI, or start that dedicated environment directly. Its version check is in `train/bootstrap.py`.

Legacy `step_*` and `final` exports keep their original inference-only contract. Opt-in stages (`init_adapter: new`, `continue_stage`, or `resume`) add verified parent inheritance, distinct HF `checkpoint_N` optimizer-boundary checkpoints, and a versioned `final` inference adapter. Fixture stages are never production adapters. See the [Task B contract](TASK_B_CONTRACT.md) and [offline reproduction guide](../examples/vision_support/BENCHMARK_REPORT.md).

For an offline CPU round trip, use a Python environment with the HF packages and pytest. From the snapshot root run `CUDA_VISIBLE_DEVICES="" ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 python -B -m pytest -p no:cacheprovider -q lm_trainer/tests`. The test creates a local random tiny model, checks a partial accumulation window and reloads the final adapter in a fresh process.

## Asset directories

Pass `--asset-root` to use a self-contained asset directory. The same layout works for any root. Initialize the folders with `python -m lm_trainer init-assets --asset-root PATH`. This creates `config/`, `data/train/`, `data/eval/`, `assets/`, `checkpoints/`, and `runs/` and preserves existing files.

From the directory containing the package, check the bundled synthetic example with Python and PyYAML. The check needs no model files, network access, or GPU.

```sh
python -B -m lm_trainer --asset-root "$PWD/product_assets/example" --dry-run
```

In asset-root mode, `--config` and `--model-config` default to `config/train.yaml` and `config/model.yaml`. Overrides for those flags are relative to the selected root. Set `data_file_path` (legacy) or `manifest_path` (fixture/release) under `data/train/`, optionally set `eval_file_path` under `data/eval/`, and choose `checkpoint_dir` under `checkpoints/` or `runs/`. `eval_file_path` is checked and included in the resolved config for evaluation tooling. The trainer never reads `eval_file_path` into the training loader. Use literal relative paths throughout. Parent traversal and symlinks in configured asset paths and output directories are rejected.

Put a local base model under `assets/` and set `model_name_or_path`, or pass `--external-base-model /absolute/path/to/local-model` to use an existing shared model directory. `--dry-run` checks the configured local model path without requiring the model directory to exist. Training and simulation require the model to be present. The shared model option applies only to the base model.
