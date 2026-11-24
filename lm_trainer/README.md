# Causal LM trainer 1.0.0

Run `python -m lm_trainer --help` from the directory containing this package. For training, supply `--config TRAIN.yaml --model-config MODEL.yaml`. `--dry-run` validates the configuration. `--simulate-only` tokenizes input and constructs a loader on the CPU when CUDA is hidden. Relative model, JSON/JSONL data and output paths are anchored at the training YAML directory. The CLI defaults its library caches to `.cache` inside the explicit output directory. A fresh output directory is required. Supply a local base model and a `text` string per training record. Labels cover the input text, with padding masked and next-token shifting handled by the model.

`requirements.txt` records the reference HF dependencies. The default backend is `hf`, precision is `fp16`, and automatic checkpointing selects `standard`. The optional `unsloth` backend uses its own compatible interpreter. Set `LM_TRAINER_UNSLOTH_PYTHON` before running the same CLI, or start that dedicated environment directly. Its version check is in `train/bootstrap.py`.

Periodic `step_*` directories and `final` contain inference adapters plus tokenizers. `training_manifest.json` records the completed optimizer updates. Resume state for the optimizer, scheduler and RNG is unavailable. Store performance reports and runtime snapshots with each run, outside the source distribution.

For an offline CPU round trip, use a Python environment with the HF packages and pytest. From the snapshot root run `CUDA_VISIBLE_DEVICES="" ACCELERATE_USE_CPU=true HF_HUB_OFFLINE=1 python -B -m pytest -p no:cacheprovider -q lm_trainer/tests`. The test creates a local random tiny model, checks a partial accumulation window and reloads the final adapter in a fresh process.
