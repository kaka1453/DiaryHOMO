from __future__ import annotations

import fcntl
import json
import math
import os
import time
from contextlib import closing, contextmanager
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

from lm_trainer.template_policy import public_policy
from lm_trainer.train.bootstrap import initialize_backend
from lm_trainer.train.io import publish_directory, save_runtime_snapshot, validate_paths
from lm_trainer.train.performance import (
    batch_tensor_digest,
    environment_metadata,
    sha256_file,
    summarize_updates,
    write_json,
)
from lm_trainer.train.schedule import compute_schedule
from lm_trainer.train.schema import preflight_data


def simulate_pipeline(runtime: dict, *, emit=True) -> dict:
    validated = preflight_data(runtime)
    validate_paths(runtime)

    from lm_trainer.model.loader import load_tokenizer
    from lm_trainer.train.data import build_data_loader, load_training_dataset
    from lm_trainer.train.modeling import create_lora_config, load_stage_tokenizer
    from lm_trainer.train.reproducibility import set_reproducibility
    from lm_trainer.train.sampling import projected_exposure

    set_reproducibility(runtime["seed"])
    structured = runtime.get("data_mode", "legacy") != "legacy"
    tokenizer = (
        load_stage_tokenizer(runtime)
        if runtime.get("init_adapter")
        else load_tokenizer(
            runtime["model_name_or_path"],
            use_fast=structured,
            local_files_only=structured,
        )
    )
    if runtime.get("init_adapter"):
        from lm_trainer.train.stages import dev_preflight, policy_identity

        policy_identity(tokenizer, runtime)
        dev = dev_preflight(runtime, validated)
        if dev is not None:
            load_training_dataset(runtime, tokenizer, validated=dev)
    dataset = load_training_dataset(runtime, tokenizer, validated=validated)
    train_loader = build_data_loader(runtime, dataset, tokenizer)
    lora_config = create_lora_config(runtime)
    schedule = compute_schedule(runtime, train_loader)
    first_batch = next(iter(train_loader))

    summary = {
        "backend": runtime.get("backend", "hf"),
        "backend_model_validated": False,
        "dataset_size": len(dataset),
        "quantization_mode": runtime["quantization_mode"],
        "batch_input_shape": list(first_batch["input_ids"].shape),
        "batch_labels_shape": list(first_batch["labels"].shape),
        "first_batch_max_token_id": int(first_batch["input_ids"].max().item()),
        "lora_target_modules": sorted(lora_config.target_modules),
        "num_update_steps_per_epoch": schedule["num_update_steps_per_epoch"],
        "planned_training_steps": schedule["planned_training_steps"],
        "fixture_only": dataset.plan["fixture_only"],
        "data_plan": stage_plan_report(dataset),
        "sampling": projected_exposure(train_loader, runtime, schedule),
    }
    if emit:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("Train 模拟检查通过。")
    return summary


def stage_plan_report(dataset):
    return {
        **dataset.plan,
        "template_policy": dataset.template_policy
        or public_policy(dataset.plan["template_policy"]),
    }


def accumulation_windows(loader, size):
    iterator = iter(loader)
    while window := list(islice(iterator, size)):
        yield window


def update_window(model, optimizer, scheduler, accelerator, window):
    import torch

    if getattr(accelerator, "gradient_accumulation_steps", 1) != 1:
        raise RuntimeError(
            "Update loss is already normalized; Accelerate accumulation must be one."
        )
    counts = [int(batch["labels"][:, 1:].ne(-100).sum()) for batch in window]
    targets = sum(counts)
    if not targets:
        raise ValueError("Accumulation window has no next-token targets.")
    loss_sum = torch.zeros((), device=accelerator.device)
    for batch, count in zip(window, counts):
        if count == 0:
            continue
        batch = {
            name: value.to(accelerator.device, non_blocking=True)
            for name, value in batch.items()
            if name in ("input_ids", "attention_mask", "labels")
        }
        with accelerator.autocast():
            # HF and Unsloth use mean next-token NLL when called without
            # num_items_in_batch; Accelerator must not divide it again.
            loss = model(**batch).loss
        if loss is None or loss.ndim != 0:
            optimizer.zero_grad(set_to_none=True)
            raise ValueError("Expected scalar mean NLL from the model.")
        accelerator.backward(loss * (count / targets))
        loss_sum += loss.detach().float() * count
    loss_value = float(loss_sum / targets)
    if not math.isfinite(loss_value):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite loss before optimizer update.")
    gradients = [
        p.grad.detach()
        for group in optimizer.param_groups
        for p in group["params"]
        if p.grad is not None
    ]
    if (
        not gradients
        or not torch.isfinite(torch.stack(torch._foreach_norm(gradients, float("inf"))))
        .all()
        .item()
    ):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            "Missing or non-finite gradient before optimizer update."
        )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if accelerator.optimizer_step_was_skipped:
        raise FloatingPointError("AMP skipped the optimizer update.")
    scheduler.step()
    return loss_value, targets


def save_adapter(
    directory,
    model,
    tokenizer,
    updates,
    *,
    artifact_data=None,
    payload=None,
    ready=None,
):
    """Publish only after all files and hashes have been validated in staging."""
    from lm_trainer.train.artifacts import (
        ADAPTER,
        MANIFEST,
        inventory,
        read_json,
        verify_artifact,
    )

    if os.path.lexists(directory):
        raise FileExistsError("Artifact output already exists.")
    with TemporaryDirectory(prefix=".adapter-", dir=directory.parent) as temporary:
        staging = Path(temporary) / "adapter"
        managed = artifact_data is not None and "schema" in artifact_data
        if managed:
            model.save_pretrained(
                staging, safe_serialization=True, save_embedding_layers=False
            )
            config_path = staging / "adapter_config.json"
            config = read_json(config_path)
            # Resolve the base through the manifest fingerprint, not a saved host path.
            write_json(
                config_path,
                {
                    **config,
                    "base_model_name_or_path": None,
                    "revision": artifact_data["base_model"]["revision"],
                },
            )
            # PEFT's generated model card can contain the local model path.
            write_json(
                staging / "README.md",
                {
                    "artifact_kind": artifact_data["artifact_kind"],
                    "fixture_only": artifact_data["fixture_only"],
                },
            )
        else:
            model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        for name in (
            "adapter_model.safetensors",
            "adapter_config.json",
            "tokenizer_config.json",
        ):
            if not (staging / name).stat().st_size:
                raise ValueError(f"Empty adapter export file: {name}")
        if payload is not None:
            payload(staging)
        manifest = {
            "artifact_kind": ADAPTER,
            "optimizer_updates": updates,
            **(artifact_data or {}),
        }
        if managed:
            # Some tokenizer implementations serialize the original path.
            cfg = staging / "tokenizer_config.json"
            tok_cfg = read_json(cfg)
            tok_cfg.pop("name_or_path", None)
            write_json(cfg, tok_cfg)
            manifest["artifacts"] = inventory(staging)
        write_json(staging / MANIFEST, manifest)
        if managed:
            from lm_trainer.train.modeling import verify_loaded_weights

            verify_loaded_weights(model, staging)
            verify_artifact(
                staging,
                kind=manifest["artifact_kind"],
                allow_fixture=manifest["fixture_only"],
            )
        if ready is not None:
            ready(staging, manifest)
        if os.path.lexists(directory):
            raise FileExistsError("Artifact output already exists.")
        publish_directory(staging, directory)


@contextmanager
def output_directory(runtime):
    validate_paths(runtime)
    directory = Path(runtime["checkpoint_dir"])
    directory.mkdir(parents=True, exist_ok=not runtime.get("init_adapter"))
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (
            any(directory.glob("step_*"))
            or any(directory.glob("checkpoint_*"))
            or any(
                os.path.lexists(directory / name)
                for name in (
                    "final",
                    "performance.json",
                    "resolved_train_runtime.json",
                    "metrics_images",
                )
            )
        ):
            raise FileExistsError(
                "Adapter output already exists. Choose a fresh --checkpoint-dir."
            )
        yield directory
    finally:
        os.close(descriptor)


def record_epoch(metrics, epoch_rows):
    epoch_loss = sum(r["loss"] * r["loss_tokens"] for r in epoch_rows) / sum(
        r["loss_tokens"] for r in epoch_rows
    )
    seconds = sum(r["seconds"] for r in epoch_rows)
    metrics["loss"].append(epoch_loss)
    metrics["perplexity"].append(math.exp(min(epoch_loss, 20)))
    metrics["lr"].append(epoch_rows[-1]["lr"])
    metrics["step_time"].append(seconds / len(epoch_rows))
    metrics["throughput"].append(sum(r["samples"] for r in epoch_rows) / seconds)


def train(runtime: dict, *, lock_files=None):
    started = time.perf_counter()
    validated = preflight_data(runtime)
    if runtime.get("init_adapter"):
        from lm_trainer.config.stages import check_stage_paths
        from lm_trainer.train.stages import artifact_reference

        check_stage_paths([runtime])
        # Lightweight structural checks (including precision) before Accelerator
        # claims its process-wide state. The model binding is checked later.
        reference = artifact_reference(runtime)
    else:
        reference = None
    with output_directory(runtime) as directory:
        initialize_backend(runtime)
        return run_training(
            runtime,
            directory,
            started,
            lock_files=lock_files,
            validated=validated,
            reference=reference,
        )


def run_training(
    runtime, directory, started, *, lock_files=None, validated=None, reference=None
):
    if validated is None:
        validated = preflight_data(runtime)

    import torch
    from accelerate import Accelerator
    from transformers import get_scheduler

    from lm_trainer.train.data import build_data_loader, load_training_dataset
    from lm_trainer.train.modeling import load_training_components
    from lm_trainer.train.reproducibility import set_reproducibility
    from lm_trainer.train.sampling import Exposure, projected_exposure

    try:
        from lm_trainer.train.metrics import empty_metrics, plot_metrics
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        plot_metrics = None
    staged = bool(runtime.get("init_adapter"))
    precision = runtime.get("compute_dtype", "fp16")
    if (
        precision == "bf16"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("This GPU does not support bf16.")
    set_reproducibility(
        runtime["seed"], deterministic=staged and runtime["backend"] == "hf"
    )
    accelerator = Accelerator(
        mixed_precision="no" if precision == "fp32" else precision,
        gradient_accumulation_steps=1,
    )
    if accelerator.num_processes != 1 or accelerator.gradient_accumulation_steps != 1:
        raise RuntimeError(
            "The trainer requires one process and manages accumulation itself."
        )
    gpu = accelerator.device.type == "cuda"
    if gpu:
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    def sync():
        if gpu:
            torch.cuda.synchronize(accelerator.device)

    environment = environment_metadata(lock_files)
    stage, tokenizer = None, None
    preprocess_seconds = 0.0
    stage_started = time.perf_counter()
    if staged:
        import subprocess

        from lm_trainer.train.artifacts import (
            ADAPTER,
            CHECKPOINT,
            MANIFEST,
            STATE,
            digest,
        )
        from lm_trainer.train.checkpoints import (
            capture_rng,
            consumed_cursor,
            restore_rng,
            restore_training,
            training_state,
        )
        from lm_trainer.train.dev import evaluate_dev
        from lm_trainer.train.modeling import load_stage_tokenizer
        from lm_trainer.train.stages import Stage, dev_preflight

        repo = Path(__file__).resolve().parents[2]
        locks = (
            lock_files
            if lock_files is not None
            else {
                "hf": repo / "uv.lock",
                "unsloth": repo / "envs/unsloth/uv.lock",
            }
        )
        environment = environment_metadata(locks)
        driver = None
        if gpu:
            try:
                driver = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    timeout=3,
                    text=True,
                ).splitlines()[0]
            except (OSError, subprocess.SubprocessError, IndexError):
                pass
        environment.update(
            device=torch.cuda.get_device_name(accelerator.device)
            if gpu
            else str(accelerator.device),
            cuda=torch.version.cuda if gpu else None,
            driver=driver,
            cpu_threads=torch.get_num_threads(),
            deterministic=runtime["backend"] == "hf",
        )
        stage = Stage(
            runtime,
            validated,
            dev_preflight(runtime, validated),
            environment,
            reference=reference,
        )
        if runtime["backend"] == "hf":
            tokenizer = load_stage_tokenizer(runtime)
            stage.identify_tokenizer(tokenizer)
    preflight_seconds = time.perf_counter() - stage_started

    def plan_batches(tokenizer):
        dataset = load_training_dataset(runtime, tokenizer, validated=validated)
        loader = build_data_loader(runtime, dataset, tokenizer)
        if not len(loader):
            raise ValueError("Empty training dataset.")
        schedule = compute_schedule(runtime, loader)
        if stage is not None:
            stage.identify_plan(loader, schedule)
            stop_after = runtime.get("stop_after_updates")
            if (
                stop_after is not None
                and not (stage.reference["optimizer_updates"] if stage.resume else 0)
                < stop_after
                < schedule["planned_training_steps"]
            ):
                raise ValueError(
                    "stop_after_updates must be after the cursor and before the actual planned end."
                )
        return dataset, loader, schedule, projected_exposure(loader, runtime, schedule)

    if tokenizer is not None:
        prep_start = time.perf_counter()
        dataset, loader, schedule, projected = plan_batches(tokenizer)
        preprocess_seconds = time.perf_counter() - prep_start
    load_start = time.perf_counter()
    if stage is not None and runtime["backend"] == "hf":
        model, tokenizer = load_training_components(
            runtime, adapter_path=stage.path, tokenizer=tokenizer
        )
    else:
        model, tokenizer = load_training_components(runtime)
    sync()
    load_seconds = time.perf_counter() - load_start
    if runtime["quantization_mode"] in ("4bit", "8bit"):
        bit = runtime["quantization_mode"]
        if not getattr(model, f"is_loaded_in_{bit}", False):
            raise RuntimeError(f"Requested {bit} model was not loaded in {bit}.")
        if staged and bit == "4bit":
            config = model.config.quantization_config
            get = (
                config.get
                if isinstance(config, dict)
                else lambda key: getattr(config, key, None)
            )
            if (
                get("bnb_4bit_quant_type") != "nf4"
                or get("bnb_4bit_use_double_quant") is not True
            ):
                raise ValueError(
                    "Requested NF4 with double quantization did not match the loaded model."
                )
    if staged and runtime["backend"] == "unsloth":
        stage.identify_tokenizer(tokenizer)
    if stage is None or runtime["backend"] == "unsloth":
        prep_start = time.perf_counter()
        dataset, loader, schedule, projected = plan_batches(tokenizer)
        preprocess_seconds = time.perf_counter() - prep_start
    audit_start = time.perf_counter()
    if stage is not None:
        stage.identify_model(model)
    initial_audit_seconds = time.perf_counter() - audit_start
    exposure = Exposure(loader.batch_sampler, track_unique=staged)
    plan_report = stage_plan_report(dataset)
    artifact_data = None
    if dataset.plan["data_mode"] != "legacy" and not staged:
        artifact_data = {
            key: plan_report[key]
            for key in (
                "data_mode",
                "objective",
                "fixture_only",
                "manifest_sha256",
                "template_policy",
            )
        }
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=runtime["lr"]
    )
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=runtime["warmup_steps"],
        num_training_steps=schedule["planned_training_steps"],
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    optimizer.zero_grad(set_to_none=True)
    updates, dev_history = [], []
    accumulation = runtime["gradient_accumulation_steps"]
    if stage is not None:
        if stage.resume:
            updates, dev_history = restore_training(
                stage.path,
                optimizer,
                scheduler,
                accelerator,
                loader,
                exposure,
                stage.identity,
                stage.reference,
                schedule,
                accumulation,
            )
        write_json(
            directory / "resolved_train_runtime.json",
            {
                "run_id": stage.manifest["run_id"],
                "stage_name": runtime["name"],
                "identity": stage.identity,
                "schedule": schedule,
                "init_adapter": runtime["init_adapter"],
                "training_settings": stage.manifest["training_settings"],
                "planned_sampling": projected,
            },
        )
    else:
        save_runtime_snapshot(runtime)
    metrics = (
        empty_metrics()
        if plot_metrics
        else {
            key: [] for key in ("loss", "perplexity", "lr", "step_time", "throughput")
        }
    )
    first_step = len(updates)
    save_seconds = 0.0
    eval_seconds = 0.0
    target = schedule["planned_training_steps"]
    stop_after = runtime.get("stop_after_updates") or target
    model.train()
    sync()
    unwrapped = accelerator.unwrap_model(model)

    def report(reason):
        timing = {
            "preflight_seconds": preflight_seconds,
            "preprocess_seconds": preprocess_seconds,
            "model_load_seconds": load_seconds,
            "initial_model_audit_seconds": initial_audit_seconds,
            "save_seconds": save_seconds,
            "dev_seconds": eval_seconds,
            "run_wall_seconds_including_setup_and_save": time.perf_counter() - started,
        }
        steady = summarize_updates(
            updates[max(first_step, runtime.get("benchmark_warmup_steps", 2)) :], 0
        )
        result = {
            "backend": runtime.get("backend", "hf"),
            "optimizer_updates": len(updates),
            "compute_dtype": precision,
            "quantization_mode": runtime["quantization_mode"],
            "base_config_sha256": sha256_file(
                Path(runtime["model_name_or_path"]) / "config.json"
            ),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "device": torch.cuda.get_device_name(accelerator.device)
            if gpu
            else str(accelerator.device),
            "data_sha256": dataset.plan.get(
                "manifest_sha256", dataset.plan.get("data_sha256")
            ),
            "fixture_only": dataset.plan["fixture_only"],
            "data_plan": plan_report,
            "planned_sampling": projected,
            "actual_sampling": {**exposure.report(), "run_stop_reason": reason},
            **timing,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(accelerator.device)
            if gpu
            else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(accelerator.device)
            if gpu
            else None,
            "steady_state": steady,
            "measured_attempt_start_update": first_step,
            "updates": updates,
            **environment,
        }
        if stage is not None:
            result.update(
                run_id=stage.manifest["run_id"],
                stage_name=runtime["name"],
                objective=runtime["objective"],
                init_adapter=runtime["init_adapter"],
                stop_reason=reason,
                dev=dev_history[-1] if dev_history else None,
                dev_history=dev_history,
                base_model=stage.manifest["base_model"],
                tokenizer_hash=stage.manifest["tokenizer_hash"],
                template_policy=stage.manifest["template_policy"],
                quantization=stage.manifest["quantization"],
                lora_config=stage.manifest["lora_config"],
                stage_config_sha256=stage.manifest["stage_config_sha256"],
            )
        return timing, steady, result

    def run_dev(step):
        nonlocal eval_seconds
        if stage.dev is None and not runtime["dev_probe"]:
            return
        if dev_history and dev_history[-1]["optimizer_updates"] == step:
            return
        start = time.perf_counter()
        rng = capture_rng()
        try:
            result = evaluate_dev(stage, unwrapped, tokenizer, accelerator)
            dev_history.append({"optimizer_updates": step, **result})
            sync()
        finally:
            restore_rng(rng)
            model.train()
        eval_seconds += time.perf_counter() - start

    def publish(path, reason, *, checkpoint=False, terminal=False):
        nonlocal save_seconds
        save_start = time.perf_counter()
        data = artifact_data
        payload = None
        ready = None
        if stage is not None:
            data = stage.artifact_data(
                unwrapped, updates, exposure, projected, reason, full_check=terminal
            )
            timing, steady, _ = report(reason)
            data.update(
                artifact_kind=CHECKPOINT if checkpoint else ADAPTER,
                timing=timing,
                performance_summary=steady,
                dev=dev_history[-1] if dev_history else None,
                dev_history=dev_history,
            )
            if checkpoint:
                state = training_state(
                    optimizer,
                    scheduler,
                    accelerator,
                    consumed_cursor(loader, schedule, accumulation, len(updates)),
                    exposure,
                    updates,
                    stage.identity,
                    data,
                    dev_history,
                )

                def payload(staging):
                    torch.save(state, staging / STATE)

            def ready(staging, manifest):
                nonlocal save_seconds
                save_seconds += time.perf_counter() - save_start
                timing, steady, result = report(reason)
                manifest.update(
                    timing=timing,
                    performance_summary={
                        **steady,
                        "peak_allocated_bytes": result["peak_allocated_bytes"],
                        "peak_reserved_bytes": result["peak_reserved_bytes"],
                    },
                )
                write_json(staging / MANIFEST, manifest)
                if terminal:
                    write_json(directory / "performance.json", result)

        save_adapter(
            path,
            unwrapped,
            tokenizer,
            len(updates),
            artifact_data=data,
            payload=payload,
            ready=ready,
        )
        if stage is None:
            save_seconds += time.perf_counter() - save_start

    epoch_start = first_step // schedule["num_update_steps_per_epoch"]
    for epoch in range(epoch_start, schedule["planned_epochs"]):
        epoch_rows = []
        if staged:
            cursor = consumed_cursor(loader, schedule, accumulation, len(updates))
            loader.batch_sampler.load_state_dict(cursor)
            order = loader.batch_sampler.batches_at(epoch)
            next_batch = cursor["next_batch"]
        start = time.perf_counter()
        with closing(accumulation_windows(loader, accumulation)) as windows:
            for window in windows:
                loss, tokens = update_window(
                    model, optimizer, scheduler, accelerator, window
                )
                sync()
                elapsed = time.perf_counter() - start
                exposure.observe(trace for batch in window for trace in batch.trace)
                row = {
                    "loss": loss,
                    "loss_tokens": tokens,
                    "seconds": elapsed,
                    "step": len(updates) + 1,
                    "epoch": epoch + 1,
                    "microbatches": len(window),
                    "samples": sum(batch["labels"].shape[0] for batch in window),
                    "input_tokens": sum(
                        int(batch["attention_mask"].sum()) for batch in window
                    ),
                    "lr": scheduler.get_last_lr()[0],
                }
                if staged:
                    expected = [
                        [dataset[i]["trace"]["sample_id"] for i in indices]
                        for indices in order[next_batch : next_batch + len(window)]
                    ]
                    observed = [
                        [t["sample_id"] for t in batch.trace] for batch in window
                    ]
                    if expected != observed:
                        raise RuntimeError(
                            "Consumed batches diverged from the committed sampling plan."
                        )
                    row["batch_order_sha256"] = digest(observed)
                    row["batch_tensors_sha256"] = batch_tensor_digest(window)
                    next_batch += len(window)
                updates.append(row)
                epoch_rows.append(row)
                step = len(updates)
                accelerator.print(f"Epoch {epoch + 1} | step {step} | loss {loss:.4f}")
                if staged:
                    if runtime.get("dev_every") and step % runtime["dev_every"] == 0:
                        run_dev(step)
                    paused = step == stop_after < target
                    if runtime["backend"] == "hf" and (
                        paused or step % runtime["save_steps"] == 0
                    ):
                        publish(
                            directory / f"checkpoint_{step}",
                            "paused" if paused else "checkpoint_interval",
                            checkpoint=True,
                            terminal=paused,
                        )
                elif step % runtime["save_steps"] == 0:
                    publish(directory / f"step_{step}", "checkpoint_interval")
                    if plot_metrics:
                        partial = {
                            key: values.copy() for key, values in metrics.items()
                        }
                        record_epoch(partial, epoch_rows)
                        plot_metrics(
                            partial,
                            directory
                            / "metrics_images"
                            / f"metrics_up_to_step_{step}.png",
                        )
                if step == stop_after:
                    break
                start = time.perf_counter()
        record_epoch(metrics, epoch_rows)
        if len(updates) == stop_after:
            break

    if len(updates) != stop_after:
        raise RuntimeError("Training ended before the planned update boundary.")
    if stop_after < target:
        accelerator.print(
            "Training paused at a complete optimizer boundary; no final adapter published."
        )
        return updates
    reason = projected["run_stop_reason"]
    if staged:
        if exposure.report() != {
            k: v for k, v in projected.items() if k != "run_stop_reason"
        }:
            raise RuntimeError(
                "Actual consumed token plan differs from the projection."
            )
        run_dev(len(updates))
    if plot_metrics and not staged:
        plot_metrics(metrics, directory / "metrics_images" / "final.png")
    publish(directory / "final", reason, terminal=True)
    if not staged:
        _, _, result = report(reason)
        result["model_name"] = Path(runtime["model_name_or_path"]).name
        write_json(directory / "performance.json", result)
    accelerator.print("全部训练完成，推理 adapter 已保存至 checkpoint_dir/final。")
    return updates
