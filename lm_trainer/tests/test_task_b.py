"""Task B correctness matrix using only locally generated fictional fixtures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from examples.vision_support.task_b_workload import stage_configs, tiny_base, workload
from lm_trainer.config.train_config import (
    build_train_parser,
    build_train_runtime_config,
)
from lm_trainer.train.artifacts import (
    ADAPTER,
    CHECKPOINT,
    MANIFEST,
    STATE,
    WEIGHTS,
    inventory,
    verify_artifact,
)
from lm_trainer.train.engine import save_adapter, train, update_window
from lm_trainer.train.performance import sha256_file, write_json

LORA = {
    "lora_r": 4,
    "lora_alpha": 8,
    "lora_target_modules": ["q_proj", "v_proj"],
    "lora_bias": "none",
    "lora_dropout": 0.0,
}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    base = tmp_path / "base"
    tokenizer = tiny_base(base)
    tr = workload(tmp_path / "train", tokenizer, length=256, count=9)
    dv = workload(tmp_path / "dev", tokenizer, split="dev", length=256, count=2)
    cpt, sft = stage_configs(tmp_path, base, tr, dv, steps=4)
    model = tmp_path / "model.yaml"
    write_json(model, LORA)

    def config(value, **kwargs):
        location = tmp_path / "run.yaml"
        write_json(location, {**value, **kwargs})
        args = build_train_parser().parse_args(
            ["--config", str(location), "--model-config", str(model)]
        )
        return build_train_runtime_config(args)

    return SimpleNamespace(
        cpt=cpt, sft=sft, root=tmp_path, config=config, data=tr, dev=dv, base=base
    )


def manifest(path, kind=ADAPTER):
    return verify_artifact(path, kind=kind, allow_fixture=True)[0]


def artifact(config):
    return Path(config["output_dir"]) / "final"


def test_b01_b03_cpt_and_direct_sft_and_dev(setup):
    initial = sha256_file(setup.base / "model.safetensors")
    for cfg in (setup.cpt, {**setup.sft, "dev_probe": True, "dev_every": 2}):
        rows = train(setup.config(cfg))
        m = manifest(artifact(cfg))
        assert len(rows) == m["optimizer_updates"] == 4
        assert all(
            torch.isfinite(torch.tensor(r["loss"])) and r["loss_tokens"] > 0
            for r in rows
        )
        assert m["fixture_only"] and m["init_adapter"] == "new"
        assert m["adapter_weights_sha256"] != m["start_adapter_sha256"]
        assert 0 < m["trainable_parameters"]["fraction"] < 1
        assert m["planned_vs_actual_tokens"]["actual"]["unused_budget_loss_tokens"] >= 0
        assert m["dev"]["split"] == "dev"
        assert m["dev"]["sources"].keys() == set(cfg["sources"])
        assert m["dev"].get("probe", {}).get("text") is None
        if cfg.get("dev_every"):
            assert [r["optimizer_updates"] for r in m["dev_history"]] == [2, 4]
        assert str(setup.root) not in json.dumps(m)
    assert sha256_file(setup.base / "model.safetensors") == initial
    assert (
        manifest(artifact(setup.cpt))["frozen_parameter_sha256"]
        == manifest(artifact(setup.sft))["frozen_parameter_sha256"]
    )


def test_b02_inheritance_config_and_fresh_scheduler(setup):
    sft = {
        **setup.sft,
        "init_adapter": "continue_stage",
        "parent": "stage:mixed_cpt",
        "lr": 0.008,
        "seed": 17,
        "save_steps": 1,
    }
    runtime = setup.config(
        {"epochs": 999999, "max_steps": 999999, "stages": [setup.cpt, sft]}
    )
    first, second = runtime["stages"]
    assert first["epochs"] == second["epochs"] == 1
    train(first)
    parent = artifact(setup.cpt)
    before = inventory(parent), sha256_file(parent / MANIFEST)
    rows = train(second)
    m, p = manifest(artifact(sft)), manifest(parent)
    assert before == (inventory(parent), sha256_file(parent / MANIFEST))
    assert m["parent_adapter_sha256"] == before[1]
    assert (
        m["start_adapter_sha256"]
        == m["loaded_adapter_sha256"]
        == p["adapter_weights_sha256"]
    )
    assert m["adapter_weights_sha256"] != m["start_adapter_sha256"]
    assert m["frozen_parameter_sha256"] == p["frozen_parameter_sha256"]
    assert m["seed"] == 17 and m["run_id"] != p["run_id"] and rows[0]["step"] == 1
    first_state = torch.load(
        artifact(sft).parent / "checkpoint_1" / STATE, weights_only=True
    )
    assert first_state["scheduler"]["last_epoch"] == 1
    assert all(
        value["step"].item() == 1
        for value in first_state["optimizer"]["state"].values()
    )


def _assert_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left:
            _assert_tree(left[k], right[k])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tree(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("pause_step", [1, 5])
def test_b04_resume_with_worker_prefetch_dropout_and_epoch_tail(setup, pause_step):
    cfg = {
        **setup.sft,
        "epochs": 2,
        "max_steps": 7,
        "token_budget": None,
        "lora_dropout": 0.15,
        "num_workers": 2,
        "warmup_steps": 2,
        "save_steps": 1,
        "dev_manifest_path": None,
        "output_dir": str(setup.root / "continuous"),
    }
    continuous = train(setup.config(cfg))
    partial_dir = setup.root / "partial"
    train(setup.config(cfg, output_dir=str(partial_dir), stop_after_updates=pause_step))
    assert not (partial_dir / "final").exists()
    ckpt = partial_dir / f"checkpoint_{pause_step}"
    checkpoint = manifest(ckpt, CHECKPOINT)
    saved = torch.load(ckpt / STATE, weights_only=True)
    assert saved["sampler"]["epoch"] == (0 if pause_step == 1 else 1)
    assert saved["sampler"]["next_batch"] == (2 if pause_step == 1 else 0)
    completed_dir = setup.root / "resumed"
    resumed = train(
        setup.config(
            cfg,
            output_dir=str(completed_dir),
            init_adapter="resume",
            resume_from=str(ckpt),
        )
    )
    for a, b in zip(continuous, resumed, strict=True):
        assert {k: v for k, v in a.items() if k != "seconds"} == {
            k: v for k, v in b.items() if k != "seconds"
        }
    assert [r["microbatches"] for r in resumed] == [2, 2, 2, 2, 1, 2, 2]
    a, b = (
        torch.load(p / "checkpoint_7" / STATE, weights_only=True)
        for p in (artifact(cfg).parent, completed_dir)
    )
    for key in ("optimizer", "scheduler", "grad_scaler", "rng", "sampler", "exposure"):
        _assert_tree(a[key], b[key])
    _assert_tree(
        load_file(artifact(cfg) / WEIGHTS), load_file(completed_dir / "final" / WEIGHTS)
    )
    end = manifest(completed_dir / "final")
    assert end["run_id"] == checkpoint["run_id"]
    assert end["resume_checkpoint_sha256"] == sha256_file(ckpt / MANIFEST)
    assert end["loaded_adapter_sha256"] == checkpoint["adapter_weights_sha256"]
    assert manifest(ckpt, CHECKPOINT) == checkpoint


def test_b05_reload_inference_and_failed_save(setup, monkeypatch):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from lm_trainer.train import engine
    from lm_trainer.train.artifacts import tokenizer_hash
    from lm_trainer.train.modeling import load_stage_tokenizer

    old = engine.save_adapter
    live = {}
    ids = torch.tensor([[3, 4, 5]])

    def inspect(path, model, tokenizer, *args, **kwargs):
        model.eval()
        with torch.no_grad():
            live["logits"] = model(ids).logits.detach().clone()
        old(path, model, tokenizer, *args, **kwargs)

    monkeypatch.setattr(engine, "save_adapter", inspect)
    train(setup.config(setup.cpt))
    parent = artifact(setup.cpt)
    before = inventory(parent), sha256_file(parent / MANIFEST)
    base = AutoModelForCausalLM.from_pretrained(setup.base, local_files_only=True)
    reloaded = PeftModel.from_pretrained(base, parent).eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(ids).logits, live["logits"], rtol=0, atol=0)
    assert (
        tokenizer_hash(load_stage_tokenizer({"model_name_or_path": str(parent)}))
        == manifest(parent)["tokenizer_hash"]
    )
    with pytest.raises(ValueError, match="fixture"):
        verify_artifact(parent)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_artifact(parent, allow_fixture=True, expected_sha256="0" * 64)

    def interrupt(*_):
        raise OSError("interrupted after staging all files")

    output = setup.root / "unsaved"
    with pytest.raises(OSError, match="interrupted"):
        save_adapter(
            output,
            reloaded,
            load_stage_tokenizer({"model_name_or_path": str(parent)}),
            4,
            artifact_data=manifest(parent),
            ready=interrupt,
        )
    assert not output.exists() and not list(setup.root.glob(".adapter-*"))
    assert before == (inventory(parent), sha256_file(parent / MANIFEST))


def test_b06_reject_incompatible_parent_and_invalid_cfg(setup):
    train(setup.config(setup.cpt, save_steps=1, stop_after_updates=1))
    ckpt = artifact(setup.cpt).parent / "checkpoint_1"
    bad_resume = {
        **setup.cpt,
        "init_adapter": "resume",
        "resume_from": str(ckpt),
        "lr": 0.9,
        "save_steps": 1,
        "output_dir": str(setup.root / "bad-resume"),
    }
    with pytest.raises(ValueError, match="Resume configuration"):
        train(setup.config(bad_resume))
    with pytest.raises(ValueError, match="artifact schema"):
        verify_artifact(ckpt, allow_fixture=True)
    with pytest.raises(FileExistsError):
        train(setup.config(setup.cpt))
    cpt = {**setup.cpt, "output_dir": str(setup.root / "valid")}
    train(setup.config(cpt))
    parent = artifact(cpt)
    sft = {**setup.sft, "init_adapter": "continue_stage", "parent": str(parent)}
    for key, patch, error in (
        ("rank", {"lora_r": 8}, "lora_config"),
        ("targets", {"lora_target_modules": ["k_proj"]}, "lora_config"),
        (
            "template",
            {"template_policy": {"stop_tokens": ["<|im_end|>"], "version": "other"}},
            "template_policy",
        ),
        (
            "quantization",
            {"quantization_mode": "4bit", "compute_dtype": "fp16"},
            "quantization",
        ),
    ):
        with pytest.raises(ValueError, match=error):
            train(setup.config(sft, **patch, output_dir=str(setup.root / f"bad-{key}")))
    with pytest.raises(ValueError, match="target_modules"):
        train(setup.config(setup.sft, lora_target_modules=["no_such_projection"]))
    base2 = setup.root / "base2"
    shutil.copytree(setup.base, base2)
    with (base2 / "model.safetensors").open("ab") as f:
        f.write(b"X")  # same config, different weight bytes: reject before load
    with pytest.raises(ValueError, match="base_model"):
        train(
            setup.config(
                sft,
                output_dir=str(setup.root / "bad-base"),
                model_name_or_path=str(base2),
            )
        )
    tok = setup.root / "other-tokenizer"
    shutil.copytree(setup.base, tok)
    from lm_trainer.train.modeling import load_stage_tokenizer

    tokenizer = load_stage_tokenizer({"model_name_or_path": str(tok)})
    tokenizer.add_tokens(["FICTIONAL_ALTERNATE_TOKEN"])
    tokenizer.save_pretrained(tok)
    with pytest.raises(ValueError, match="tokenizer_hash"):
        train(
            setup.config(
                sft,
                output_dir=str(setup.root / "bad-tokenizer"),
                tokenizer_name_or_path=str(tok),
            )
        )
    with (parent / WEIGHTS).open("ab") as f:
        f.write(b"X")
    with pytest.raises(ValueError, match="inventory or SHA-256"):
        train(setup.config(sft, output_dir=str(setup.root / "bad-parent")))
    for patch, error in [
        ({"packing": True}, "packing"),
        ({"max_steps": 100001}, "budget"),
        ({"manifest_path": "SET_DATA"}, "placeholder"),
        ({"lora_bias": "all"}, "bias=none"),
    ]:
        with pytest.raises(ValueError, match=error):
            setup.config(setup.sft, **patch)
    with pytest.raises(ValueError, match="warmup_steps"):
        train(
            setup.config(
                setup.sft, output_dir=str(setup.root / "bad-warmup"), warmup_steps=5
            )
        )
    with pytest.raises(ValueError, match="budget"):
        train(
            setup.config(
                setup.sft, output_dir=str(setup.root / "bad-budget"), token_budget=1
            )
        )
    raw = json.loads(setup.data.read_text())
    raw["review_status"] = "pending_owner"
    write_json(setup.data, raw)
    with pytest.raises(ValueError, match="authorization"):
        train(setup.config(setup.cpt, output_dir=str(setup.root / "bad-data")))


def test_b06_no_test_payload_read(setup, monkeypatch):
    from lm_trainer.train import schema
    from lm_trainer.train.stages import dev_preflight, stage_settings

    cfg = setup.config(setup.sft)
    assert "eval_file_path" not in stage_settings(
        {**cfg, "eval_file_path": "/private/eval"}
    )
    prepared = schema.preflight_data(cfg)
    raw = json.loads(setup.dev.read_text())
    raw["files"][0]["split"] = "test"
    write_json(setup.dev, raw)
    monkeypatch.setattr(
        schema, "_verified_file", lambda *_: pytest.fail("Test payload was touched")
    )
    with pytest.raises(ValueError, match="dev required"):
        dev_preflight(cfg, prepared)


@pytest.mark.parametrize("mode", ["resume", "continue_stage"])
def test_b07_unsloth_lifecycle_unsupported_before_backend_import(setup, mode):
    with pytest.raises(ValueError, match="unsupported;.*HF"):
        setup.config(
            setup.cpt, backend="unsloth", compute_dtype="bf16", init_adapter=mode
        )


def test_b06_nonfinite_gradient_fails_before_optimizer_step():
    p = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.SGD([p], lr=1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        autocast=__import__("contextlib").nullcontext,
        backward=lambda loss: loss.backward(),
        optimizer_step_was_skipped=False,
    )
    model = lambda **_: SimpleNamespace(
        loss=p.sqrt()
    )  # finite scalar, infinite derivative
    batch = {"input_ids": torch.tensor([[0, 0]]), "labels": torch.tensor([[0, 1]])}
    with pytest.raises(FloatingPointError, match="gradient"):
        update_window(model, opt, sched, accelerator, [batch])
    assert p.item() == 0 and sched.last_epoch == 0


def test_b02_parent_fp32_weights_not_rounded_by_bf16_base(setup):
    from lm_trainer.train.modeling import adapter_tensors, load_training_model

    cfg = setup.config(setup.cpt, compute_dtype="bf16")
    model = load_training_model(cfg)
    assert any(
        p.dtype == torch.bfloat16 for p in model.parameters() if not p.requires_grad
    )
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.normal_()
    adapter = setup.root / "bf16_parent"
    model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
    reloaded = load_training_model(cfg, adapter_path=adapter)
    assert all(t.dtype == torch.float32 for t in adapter_tensors(reloaded).values())
    _assert_tree(adapter_tensors(reloaded), adapter_tensors(model))


def test_b08_multistage_cli_preflight_and_simulate_do_not_create_outputs(setup):
    import os
    import subprocess
    import sys

    setup.config(
        {
            "stages": [
                setup.cpt,
                {
                    **setup.sft,
                    "init_adapter": "continue_stage",
                    "parent": "stage:mixed_cpt",
                },
            ]
        }
    )
    command = [
        sys.executable,
        "-m",
        "lm_trainer",
        "--config",
        str(setup.root / "run.yaml"),
        "--model-config",
        str(setup.root / "model.yaml"),
    ]
    for check in ("--dry-run", "--validate-release", "--simulate-only"):
        out = subprocess.check_output(
            [*command, check], env={**os.environ, "TOKENIZERS_PARALLELISM": "false"}
        )
        plan = json.loads(out)
        assert [x["stage_name"] for x in plan["stages"]] == [
            "mixed_cpt",
            "dialogue_sft",
        ]
        assert plan["stages"][1]["parent_stage"] == "mixed_cpt"
        assert plan["token_counts_measured"] is (check == "--simulate-only")
        assert not any(artifact(cfg).parent.exists() for cfg in (setup.cpt, setup.sft))


@pytest.mark.skipif(
    __import__("os").environ.get("LM_TRAINER_TASK_B_CUDA") != "1",
    reason="opt-in CUDA/AMP checkpoint test",
)
@pytest.mark.parametrize("quantization_mode", ["none", "4bit"])
def test_b04_cuda_fp16_scaler_resume(setup, quantization_mode):
    assert torch.cuda.is_available()
    cfg = {
        **setup.cpt,
        "compute_dtype": "fp16",
        "quantization_mode": quantization_mode,
        "lora_dropout": 0.1,
        "save_steps": 1,
        "dev_manifest_path": None,
    }
    train(setup.config(cfg))
    paused_dir = setup.root / "paused"
    train(setup.config(cfg, output_dir=str(paused_dir), stop_after_updates=1))
    checkpoint = paused_dir / "checkpoint_1"
    state = torch.load(checkpoint / STATE, weights_only=True)
    assert state["grad_scaler"] and state["rng"]["cuda"]
    output = setup.root / "resumed"
    train(
        setup.config(
            cfg,
            init_adapter="resume",
            resume_from=str(checkpoint),
            output_dir=str(output),
        )
    )
    before = manifest(artifact(cfg))
    resumed = manifest(output / "final")
    assert resumed["adapter_weights_sha256"] == before["adapter_weights_sha256"]
    a, b = (
        torch.load(p / "checkpoint_4" / STATE, weights_only=True)
        for p in (artifact(cfg).parent, output)
    )
    for key in ("optimizer", "scheduler", "grad_scaler", "rng", "sampler", "exposure"):
        _assert_tree(a[key], b[key])
    if quantization_mode == "4bit":
        sft = {
            **setup.sft,
            "compute_dtype": "fp16",
            "quantization_mode": "4bit",
            "lora_dropout": 0.1,
            "init_adapter": "continue_stage",
            "parent": str(output / "final"),
            "dev_manifest_path": None,
        }
        train(setup.config(sft))
        final = manifest(artifact(sft))
        assert final["start_adapter_sha256"] == resumed["adapter_weights_sha256"]
        assert final["frozen_parameter_sha256"] == resumed["frozen_parameter_sha256"]
        assert final["adapter_weights_sha256"] != final["start_adapter_sha256"]
