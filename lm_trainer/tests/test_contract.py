import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from lm_trainer.config.train_config import (
    build_train_parser,
    build_train_runtime_config,
)


def test_cli_requires_both_configs(tmp_path):
    with pytest.raises(SystemExit):
        build_train_parser().parse_args([])
    args = build_train_parser().parse_args(
        [
            "--config",
            str(tmp_path / "missing.yaml"),
            "--model-config",
            str(tmp_path / "m.yaml"),
        ]
    )
    with pytest.raises(FileNotFoundError):
        build_train_runtime_config(args)


def test_lightweight_cli_and_module_origins():
    root = Path(__file__).resolve().parents[2]
    code = """
import importlib, pathlib, sys
root = pathlib.Path.cwd()
for name in ('lm_trainer', 'lm_trainer.__main__', 'lm_trainer.train.engine'):
    importlib.import_module(name)
assert not {'torch','transformers','peft','unsloth'} & set(sys.modules)
for name in ('lm_trainer.model.loader', 'lm_trainer.generation', 'lm_trainer.train.data'):
    importlib.import_module(name)
assert 'unsloth' not in sys.modules
for name, module in sys.modules.items():
    if name == 'lm_trainer' or name.startswith('lm_trainer.'):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(root / 'lm_trainer'), name
"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-B", "-c", code], cwd=root, env=env, check=True)
    subprocess.run(
        [sys.executable, "-B", "-m", "lm_trainer", "--help"],
        cwd=root,
        env=env,
        check=True,
    )


def test_config_relative_paths_legacy_quantization_and_backend(tmp_path, monkeypatch):
    train = dict(
        model_name_or_path="model",
        checkpoint_dir="output",
        data_file_path="synthetic.jsonl",
        max_length=16,
        load_in_8bit=True,
        batch_size=1,
        shuffle=False,
        auto_pad_batch=True,
        epochs=1,
        gradient_accumulation_steps=2,
        lr=0.001,
        warmup_steps=0,
        max_steps=2,
        save_steps=9,
        seed=1,
        num_workers=0,
        pin_memory=False,
    )
    train_path, model_path = tmp_path / "train.yaml", tmp_path / "model.yaml"
    train_path.write_text(json.dumps(train))
    model_path.write_text(
        json.dumps(
            dict(
                lora_r=2,
                lora_alpha=4,
                lora_target_modules=["q_proj"],
                lora_dropout=0,
                lora_bias="none",
            )
        )
    )
    args = build_train_parser().parse_args(
        ["--config", str(train_path), "--model-config", str(model_path)]
    )
    monkeypatch.chdir(tmp_path.parent)
    cfg = build_train_runtime_config(args)
    assert (cfg["backend"], cfg["compute_dtype"], cfg["quantization_mode"]) == (
        "hf",
        "fp16",
        "8bit",
    )
    assert cfg["gradient_checkpointing"] == "standard"
    assert Path(cfg["data_file_path"]) == tmp_path / "synthetic.jsonl"
