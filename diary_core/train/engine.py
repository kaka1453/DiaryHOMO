from __future__ import annotations

import json
import math
import time
from itertools import islice
from tempfile import TemporaryDirectory
from contextlib import contextmanager
import os
import fcntl
from pathlib import Path

import torch
from accelerate import Accelerator
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


def accumulation_windows(loader, size):
    iterator = iter(loader)
    while window := list(islice(iterator, size)):
        yield window


def update_window(model, optimizer, scheduler, accelerator, window):
    counts = [int(batch['labels'][:, 1:].ne(-100).sum()) for batch in window]
    targets = sum(counts)
    if not targets:
        raise ValueError('Accumulation window has no next-token targets.')
    loss_sum = torch.zeros((), device=accelerator.device)
    for batch, count in zip(window, counts):
        if count == 0:
            continue
        batch = {name: value.to(accelerator.device, non_blocking=True) for name, value in batch.items()}
        with accelerator.autocast():
            loss = model(**batch).loss
        accelerator.backward(loss * (count / targets))
        loss_sum += loss.detach().float() * count
    loss_value = float(loss_sum / targets)
    if not math.isfinite(loss_value):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError('Non-finite loss before optimizer update.')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if accelerator.optimizer_step_was_skipped:
        raise FloatingPointError('AMP skipped the optimizer update.')
    scheduler.step()
    return loss_value, targets


def save_adapter(directory, model, tokenizer, updates):
    with TemporaryDirectory(prefix='.adapter-', dir=directory.parent) as temporary:
        staging = Path(temporary) / 'adapter'
        model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        for name in ('adapter_model.safetensors', 'adapter_config.json', 'tokenizer_config.json'):
            if not (staging / name).stat().st_size:
                raise ValueError(f'Empty adapter export file: {name}')
        manifest = {'artifact_kind': 'inference_adapter', 'optimizer_updates': updates}
        (staging / 'training_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        staging.rename(directory)


@contextmanager
def output_directory(runtime):
    validate_paths(runtime)
    directory = Path(runtime['checkpoint_dir'])
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(directory.glob('step_*')) or os.path.lexists(directory / 'final'):
            raise FileExistsError('Adapter output already exists. Choose a fresh --checkpoint-dir.')
        yield directory
    finally:
        os.close(descriptor)


def train(runtime: dict):
    with output_directory(runtime) as directory:
        return run_training(runtime, directory)


def run_training(runtime, directory):
    set_reproducibility(runtime['seed'])
    accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=1)
    if accelerator.num_processes != 1 or accelerator.gradient_accumulation_steps != 1:
        raise RuntimeError('The trainer requires one process and manages accumulation itself.')
    tokenizer = load_tokenizer(runtime['model_name_or_path'], use_fast=False)
    model = load_training_model(runtime, create_lora_config(runtime))
    dataset = load_training_dataset(runtime, tokenizer)
    loader = build_data_loader(runtime, dataset, tokenizer)
    if not len(loader):
        raise ValueError('Empty training dataset.')
    schedule = compute_schedule(runtime, loader)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=runtime['lr'])
    scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=runtime['warmup_steps'],
                              num_training_steps=schedule['planned_training_steps'])
    model, optimizer = accelerator.prepare(model, optimizer)
    optimizer.zero_grad(set_to_none=True)
    save_runtime_snapshot(runtime)
    metrics = empty_metrics()
    updates = []
    for epoch in range(runtime['epochs']):
        model.train()
        epoch_rows = []
        start = time.perf_counter()
        for window in accumulation_windows(loader, runtime['gradient_accumulation_steps']):
            loss, tokens = update_window(model, optimizer, scheduler, accelerator, window)
            if accelerator.device.type == 'cuda':
                torch.cuda.synchronize(accelerator.device)
            elapsed = time.perf_counter() - start
            row = {'loss': loss, 'loss_tokens': tokens, 'seconds': elapsed,
                   'samples': sum(batch['labels'].shape[0] for batch in window),
                   'input_tokens': sum(int(batch['attention_mask'].sum()) for batch in window),
                   'lr': scheduler.get_last_lr()[0]}
            updates.append(row)
            epoch_rows.append(row)
            step = len(updates)
            accelerator.print(f'Epoch {epoch + 1} | step {step} | loss {loss:.4f}')
            if step % runtime['save_steps'] == 0:
                save_adapter(directory / f'step_{step}', accelerator.unwrap_model(model), tokenizer, step)
                plot_metrics(metrics, directory / 'metrics_images' / f'metrics_up_to_step_{step}.png')
            if step == schedule['planned_training_steps']:
                break
            start = time.perf_counter()
        epoch_loss = sum(r['loss'] * r['loss_tokens'] for r in epoch_rows) / sum(r['loss_tokens'] for r in epoch_rows)
        seconds = sum(r['seconds'] for r in epoch_rows)
        metrics['loss'].append(epoch_loss)
        metrics['perplexity'].append(math.exp(min(epoch_loss, 20)))
        metrics['lr'].append(epoch_rows[-1]['lr'])
        metrics['step_time'].append(seconds / len(epoch_rows))
        metrics['throughput'].append(sum(r['samples'] for r in epoch_rows) / seconds)
        if len(updates) == schedule['planned_training_steps']:
            break
    save_adapter(directory / 'final', accelerator.unwrap_model(model), tokenizer, len(updates))
    accelerator.print(f"全部训练完成，推理 adapter 已保存到 {directory / 'final'}")
    return updates


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
