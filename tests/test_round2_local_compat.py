"""Local entrypoint safety tests for Round 2 compatibility (M2)."""

import json
import sys
from pathlib import Path

import pytest

import diary_core.train.engine as diary_engine
import train.train as legacy_train


def _make_asset_root(tmp_path: Path) -> Path:
    root = tmp_path / "assets"
    for sub in ("config", "data/train", "data/eval", "assets/base", "checkpoints", "runs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "config" / "model.yaml").write_text(
        "model_name_or_path: assets/base\n", encoding="utf-8"
    )
    (root / "config" / "train.yaml").write_text(
        "data_mode: release\n"
        "manifest_path: data/train/missing_manifest.json\n"
        "checkpoint_dir: checkpoints/run1\n"
        "max_length: 128\n"
        "batch_size: 1\n"
        "gradient_accumulation_steps: 1\n"
        "epochs: 1\n"
        "lr: 0.0001\n"
        "lora_r: 4\n"
        "lora_alpha: 8\n"
        "lora_dropout: 0.0\n"
        "lora_target_modules: q_proj\n"
        "quantization_mode: none\n"
        "backend: hf\n"
        "seed: 42\n",
        encoding="utf-8",
    )
    return root


def test_legacy_train_validate_release_never_trains_and_exits_nonzero_on_bad_manifest(
    tmp_path, monkeypatch
):
    root = _make_asset_root(tmp_path)
    called = []

    def forbid(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("training/dispatch/model loading must not be called")

    monkeypatch.setattr(diary_engine, "train", forbid)
    monkeypatch.setattr(legacy_train, "_dispatch_unsloth", forbid)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train/train.py",
            "--asset-root",
            str(root),
            "--validate-release",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        legacy_train.main()
    assert exc.value.code != 0
    assert not called


def test_diary_engine_validate_release_never_trains_and_exits_nonzero_on_bad_manifest(
    tmp_path, monkeypatch
):
    root = _make_asset_root(tmp_path)
    called = []

    def forbid(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("training must not be called")

    monkeypatch.setattr(diary_engine, "train", forbid)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diary_core.train.engine",
            "--asset-root",
            str(root),
            "--validate-release",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        diary_engine.main()
    assert exc.value.code != 0
    assert not called


def test_legacy_train_rejects_staged_runtime_without_training(tmp_path, monkeypatch):
    root = _make_asset_root(tmp_path)
    (root / "config" / "staged.yaml").write_text(
        "stages:\n"
        "  - name: s1\n"
        "    init_adapter: new\n"
        "    data_mode: release\n"
        "    objective: causal_text\n"
        "    manifest_path: data/train/missing_manifest.json\n"
        "    checkpoint_dir: checkpoints/s1\n"
        "    max_length: 128\n"
        "    batch_size: 1\n"
        "    gradient_accumulation_steps: 1\n"
        "    epochs: 1\n"
        "    lr: 0.0001\n"
        "    lora_r: 4\n"
        "    lora_alpha: 8\n"
        "    lora_dropout: 0.0\n"
        "    lora_target_modules: q_proj\n"
        "    quantization_mode: none\n"
        "    backend: hf\n"
        "    seed: 42\n",
        encoding="utf-8",
    )
    called = []
    monkeypatch.setattr(diary_engine, "train", lambda *a, **k: called.append("train"))
    monkeypatch.setattr(legacy_train, "_dispatch_unsloth", lambda *a, **k: called.append("unsloth"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train/train.py",
            "--asset-root",
            str(root),
            "--config",
            "config/staged.yaml",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        legacy_train.main()
    assert exc.value.code == 2
    assert not called
