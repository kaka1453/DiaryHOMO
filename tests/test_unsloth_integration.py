from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from diary_core.config.train_config import (
    build_train_parser,
    build_train_runtime_config,
)
from diary_core.train.performance import summarize_updates

ROOT = Path(__file__).resolve().parents[1]


def runtime(*args):
    return build_train_runtime_config(build_train_parser().parse_args(list(args)))


def test_defaults_preserve_hf():
    cfg = runtime()
    assert (cfg["backend"], cfg["compute_dtype"], cfg["gradient_checkpointing"]) == (
        "hf",
        "fp16",
        "standard",
    )


def test_unsloth_config_and_cli():
    cfg = runtime(
        "--backend",
        "unsloth",
        "--quantization-mode",
        "none",
        "--compute-dtype",
        "bf16",
        "--lora-dropout",
        "0",
    )
    assert cfg["backend"] == "unsloth" and cfg["compute_dtype"] == "bf16"
    assert cfg["lora_dropout"] == 0 and cfg["gradient_checkpointing"] == "unsloth"
    assert (
        runtime("--backend", "unsloth", "--quantization-mode", "none")[
            "gradient_checkpointing"
        ]
        == "unsloth"
    )


@pytest.mark.parametrize(
    "args",
    [
        ("--backend", "bad"),
        ("--compute-dtype", "fp32"),
        ("--backend", "unsloth", "--quantization-mode", "8bit"),
        ("--gradient-checkpointing", "unsloth"),
        ("--benchmark-warmup-steps", "-1"),
        ("--lora-dropout", "1"),
    ],
)
def test_bad_options(args):
    with pytest.raises(ValueError):
        runtime(*args)


def test_bootstrap_import_order_and_dry_run():
    code = """
import sys
from train.train import build_parser, build_runtime_config
from diary_core.train.engine import train
build_runtime_config(build_parser().parse_args(['--backend', 'unsloth', '--quantization-mode', 'none']))
assert not any(name in sys.modules for name in ('torch', 'transformers', 'peft', 'trl', 'unsloth'))
from diary_core.train.bootstrap import initialize_backend
sys.modules['transformers'] = object()
try:
    initialize_backend({'backend': 'unsloth'})
except RuntimeError as exc:
    assert '必须先导入' in str(exc)
else:
    raise AssertionError('late import accepted')
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "train/train.py"),
            "--backend",
            "unsloth",
            "--quantization-mode",
            "none",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout.split("\n", 1)[1])["backend"] == "unsloth"


def test_multirank_rejected(monkeypatch):
    from diary_core.train.bootstrap import initialize_backend

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="单进程单卡"):
        initialize_backend({"backend": "hf"})


def test_performance_accounting():
    rows = [
        {"seconds": s, "samples": 4, "input_tokens": 100, "loss_tokens": 96}
        for s in (100, 1, 3)
    ]
    summary = summarize_updates(rows, 1)
    assert summary["seconds_per_update"] == 2
    assert summary["input_tokens_per_second"] == 50
    assert summary["samples_per_second"] == 2
    assert summarize_updates(rows, 3)["seconds_per_update"] is None


@pytest.mark.parametrize(
    "quantization,checkpointing",
    [("4bit", "unsloth"), ("none", "standard"), ("4bit", "none")],
)
def test_unsloth_factory_receives_config_once(monkeypatch, quantization, checkpointing):
    from types import SimpleNamespace

    from diary_core.train import modeling

    calls = []
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    tokenizer = SimpleNamespace(pad_token=None, eos_token="eos", padding_side="left")

    class Factory:
        @staticmethod
        def from_pretrained(**kwargs):
            calls.append(("load", kwargs))
            return model, tokenizer

        @staticmethod
        def get_peft_model(value, **kwargs):
            assert value is model
            calls.append(("peft", kwargs))
            return model

        @staticmethod
        def for_training(value, **kwargs):
            calls.append(("train", kwargs))

    monkeypatch.setitem(
        sys.modules, "unsloth", SimpleNamespace(FastLanguageModel=Factory)
    )
    monkeypatch.setattr(modeling, "initialize_backend", lambda cfg: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    cfg = runtime(
        "--backend",
        "unsloth",
        "--quantization-mode",
        quantization,
        "--gradient-checkpointing",
        checkpointing,
        "--lora-dropout",
        "0",
    )
    result = modeling.load_training_components(cfg)
    assert result == (model, tokenizer)
    assert [call[0] for call in calls] == ["load", "peft", "train"]
    assert calls[0][1]["load_in_4bit"] == (quantization == "4bit")
    assert (
        calls[1][1]["use_gradient_checkpointing"]
        == {"standard": True, "unsloth": "unsloth", "none": False}[checkpointing]
    )
    assert not model.config.use_cache and tokenizer.pad_token == "eos"


def test_separate_interpreter(monkeypatch, tmp_path):
    import os

    from diary_core.train.bootstrap import dispatch_unsloth, initialize_backend

    monkeypatch.setenv("BOA_UNSLOTH_PYTHON", sys.executable)
    pid_file = tmp_path / "pid"
    code = """import os, sys
from pathlib import Path
assert 'BOA_UNSLOTH_PYTHON' not in os.environ
assert 'torch' not in sys.modules
Path(sys.argv[1]).write_text(str(os.getpid()))"""
    assert not dispatch_unsloth({"backend": "hf"}, ["-c", code, str(pid_file)])
    assert dispatch_unsloth({"backend": "unsloth"}, ["-c", code, str(pid_file)])
    assert int(pid_file.read_text()) != os.getpid()
    monkeypatch.setitem(sys.modules, "unsloth", object())
    with pytest.raises(RuntimeError, match="独立进程"):
        initialize_backend({"backend": "hf"})


def test_legacy_imports_and_quantization(monkeypatch):
    from types import SimpleNamespace

    from diary_core.model import loader
    from train.train import build_quantization_config, load_model, load_training_model

    assert load_model is load_training_model
    assert build_quantization_config("none") is None
    assert build_quantization_config({"quantization_mode": "8bit"}).load_in_8bit
    assert build_quantization_config("4bit").bnb_4bit_quant_type == "nf4"
    monkeypatch.setattr(
        loader.AutoModelForCausalLM,
        "from_pretrained",
        lambda _path, **kwargs: SimpleNamespace(**kwargs),
    )
    base = loader.load_base_model("synthetic-base", "8bit")
    assert base.trust_remote_code and base.quantization_config.load_in_8bit
