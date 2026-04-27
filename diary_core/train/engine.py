from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import get_scheduler

from diary_core.config.common import dump_runtime_config
from diary_core.config.train_config import build_train_parser, build_train_runtime_config
from diary_core.model.loader import load_tokenizer
from diary_core.train.data import build_data_loader, load_training_dataset
from diary_core.train.io import save_runtime_snapshot, validate_paths
from diary_core.train.metrics import empty_metrics, plot_metrics
from diary_core.train.modeling import create_lora_config, load_training_model
from diary_core.train.reproducibility import set_reproducibility
from diary_core.train.schedule import compute_schedule


def simulate_pipeline(runtime: dict) -> None:
    validate_paths(runtime)
    set_reproducibility(runtime["seed"])
    tokenizer = load_tokenizer(runtime["model_name_or_path"], use_fast=False)
    dataset = load_training_dataset(runtime, tokenizer)
    train_loader = build_data_loader(runtime, dataset, tokenizer)
    lora_config = create_lora_config(runtime)
    schedule = compute_schedule(runtime, train_loader)
    first_batch = next(iter(train_loader))

    summary = {
        "dataset_size": len(dataset),
        "quantization_mode": runtime["quantization_mode"],
        "batch_input_shape": list(first_batch["input_ids"].shape),
        "batch_labels_shape": list(first_batch["labels"].shape),
        "first_batch_max_token_id": int(first_batch["input_ids"].max().item()),
        "lora_target_modules": sorted(list(lora_config.target_modules)),
        "num_update_steps_per_epoch": schedule["num_update_steps_per_epoch"],
        "planned_training_steps": schedule["planned_training_steps"],
        "checkpoint_dir": runtime["checkpoint_dir"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Train 模拟检查通过。")


def train(runtime: dict) -> None:
    validate_paths(runtime)
    set_reproducibility(runtime["seed"])
    snapshot_path = save_runtime_snapshot(runtime)

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=runtime["gradient_accumulation_steps"],
    )

    tokenizer = load_tokenizer(runtime["model_name_or_path"], use_fast=False)
    dataset = load_training_dataset(runtime, tokenizer)
    train_loader = build_data_loader(runtime, dataset, tokenizer)
    lora_config = create_lora_config(runtime)
    schedule = compute_schedule(runtime, train_loader)

    model = load_training_model(runtime, lora_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=runtime["lr"])
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=runtime["warmup_steps"],
        num_training_steps=schedule["planned_training_steps"],
    )

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        lr_scheduler,
    )

    metrics = empty_metrics()

    accelerator.print(f"已写入实验快照: {snapshot_path}")
    global_step = 0
    epoch = 0

    while global_step < runtime["max_steps"] and epoch < runtime["epochs"]:
        model.train()
        epoch_loss = 0.0
        epoch_lr = []
        epoch_step_time = []
        epoch_throughput = []
        optimizer_updates = 0
        start_time = time.time()
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}",
            leave=False,
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in enumerate(progress):
            if global_step >= runtime["max_steps"]:
                break

            step_start = time.time()
            outputs = model(**batch)
            loss = outputs.loss / runtime["gradient_accumulation_steps"]
            accelerator.backward(loss)

            is_last_batch = (step + 1) == len(train_loader)
            should_step = ((step + 1) % runtime["gradient_accumulation_steps"] == 0) or is_last_batch
            if not should_step:
                continue

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            optimizer_updates += 1
            epoch_loss += loss.item() * runtime["gradient_accumulation_steps"]

            step_time = time.time() - step_start
            epoch_lr.append(lr_scheduler.get_last_lr()[0])
            epoch_step_time.append(step_time)
            epoch_throughput.append(runtime["batch_size"] / max(step_time, 1e-6))

            if global_step % runtime["save_steps"] == 0:
                save_dir = Path(runtime["checkpoint_dir"]) / f"step_{global_step}"
                save_dir.mkdir(parents=True, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(save_dir, safe_serialization=True)
                tokenizer.save_pretrained(save_dir)
                accelerator.print(f"模型已保存至 {save_dir}")

                metrics_path = Path(runtime["checkpoint_dir"]) / "metrics_images" / f"metrics_up_to_step_{global_step}.png"
                plot_metrics(metrics, metrics_path)
                accelerator.print(f"指标图已保存至 {metrics_path}")

        if optimizer_updates == 0:
            accelerator.print("当前 epoch 未产生优化步，训练提前结束。")
            break

        avg_loss = epoch_loss / optimizer_updates
        metrics["loss"].append(avg_loss)
        metrics["perplexity"].append(math.exp(min(avg_loss, 20)))
        metrics["lr"].append(sum(epoch_lr) / len(epoch_lr))
        metrics["step_time"].append(sum(epoch_step_time) / len(epoch_step_time))
        metrics["throughput"].append(sum(epoch_throughput) / len(epoch_throughput))

        epoch += 1
        epoch_elapsed = time.time() - start_time
        accelerator.print(
            f"Epoch {epoch} | avg_loss: {metrics['loss'][-1]:.4f} | "
            f"time: {epoch_elapsed:.1f}s | speed: {epoch_elapsed / max(len(train_loader), 1):.2f}s/step"
        )

    accelerator.print("全部训练完成。")


def main() -> None:
    parser = build_train_parser()
    args = parser.parse_args()
    runtime = build_train_runtime_config(args)

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
