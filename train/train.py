import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CONFIG_PATH = PROJECT_ROOT / "config" / "train.yaml"
DEFAULT_MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"

TRAIN_CONFIG_FIELDS = [
    "model_name_or_path",
    "checkpoint_dir",
    "data_file_path",
    "max_length",
    "load_in_8bit",
    "batch_size",
    "shuffle",
    "auto_pad_batch",
    "epochs",
    "gradient_accumulation_steps",
    "lr",
    "warmup_steps",
    "max_steps",
    "save_steps",
    "seed",
    "num_workers",
    "pin_memory",
]

MODEL_CONFIG_FIELDS = [
    "lora_r",
    "lora_alpha",
    "lora_target_modules",
    "lora_dropout",
    "lora_bias",
]

PATH_FIELDS = {"model_name_or_path", "checkpoint_dir", "data_file_path"}
INT_FIELDS = {
    "max_length",
    "batch_size",
    "epochs",
    "gradient_accumulation_steps",
    "warmup_steps",
    "max_steps",
    "save_steps",
    "seed",
    "num_workers",
    "lora_r",
    "lora_alpha",
}
FLOAT_FIELDS = {"lr", "lora_dropout"}
BOOL_FIELDS = {"load_in_8bit", "shuffle", "auto_pad_batch", "pin_memory"}
POSITIVE_FIELDS = {
    "max_length",
    "batch_size",
    "gradient_accumulation_steps",
    "epochs",
    "max_steps",
    "save_steps",
    "lora_r",
    "lora_alpha",
}
NON_NEGATIVE_FIELDS = {"warmup_steps", "num_workers"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


CLI_OVERRIDE_SPECS = [
    ("--model-name-or-path", "model_name_or_path", str, "Override model path from config."),
    ("--checkpoint-dir", "checkpoint_dir", str, "Override checkpoint output directory from config."),
    ("--data-file-path", "data_file_path", str, "Override training dataset path from config."),
    ("--max-length", "max_length", int, "Override max sequence length from config."),
    ("--load-in-8bit", "load_in_8bit", str2bool, "Override 8bit loading switch from config."),
    ("--batch-size", "batch_size", int, "Override batch size from config."),
    ("--shuffle", "shuffle", str2bool, "Override whether sorted batches are shuffled between epochs."),
    ("--auto-pad-batch", "auto_pad_batch", str2bool, "Override adaptive padding switch from config."),
    ("--epochs", "epochs", int, "Override max epoch count from config."),
    (
        "--gradient-accumulation-steps",
        "gradient_accumulation_steps",
        int,
        "Override gradient accumulation steps from config.",
    ),
    ("--lr", "lr", float, "Override learning rate from config."),
    ("--warmup-steps", "warmup_steps", int, "Override warmup steps from config."),
    ("--max-steps", "max_steps", int, "Override total optimizer steps from config."),
    ("--save-steps", "save_steps", int, "Override checkpoint save interval from config."),
    ("--seed", "seed", int, "Override random seed from config."),
    ("--num-workers", "num_workers", int, "Override dataloader worker count from config."),
    ("--pin-memory", "pin_memory", str2bool, "Override dataloader pin_memory switch from config."),
    ("--lora-r", "lora_r", int, "Override LoRA rank from model config."),
    ("--lora-alpha", "lora_alpha", int, "Override LoRA alpha from model config."),
    (
        "--lora-target-modules",
        "lora_target_modules",
        str,
        "Override LoRA target modules from model config, comma-separated.",
    ),
    ("--lora-dropout", "lora_dropout", float, "Override LoRA dropout from model config."),
    ("--lora-bias", "lora_bias", str, "Override LoRA bias from model config."),
]


def parse_target_modules(value):
    if value is None:
        return None
    if isinstance(value, list):
        modules = [str(item).strip() for item in value if str(item).strip()]
    else:
        modules = [item.strip() for item in str(value).split(",") if item.strip()]
    if not modules:
        raise ValueError("lora_target_modules 不能为空。")
    return modules


def build_parser():
    parser = argparse.ArgumentParser(description="Qwen LoRA training entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_TRAIN_CONFIG_PATH),
        help="Training YAML config file path.",
    )
    parser.add_argument(
        "--model-config",
        dest="model_config",
        type=str,
        default=str(DEFAULT_MODEL_CONFIG_PATH),
        help="Model YAML config file path.",
    )
    for flag, dest, value_type, help_text in CLI_OVERRIDE_SPECS:
        parser.add_argument(
            flag,
            dest=dest,
            type=value_type,
            default=None,
            help=help_text,
        )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Only validate config and print resolved values.",
    )
    parser.add_argument(
        "--simulate-only",
        dest="simulate_only",
        action="store_true",
        help="Run a CPU-side simulation without loading the model into VRAM.",
    )
    return parser


def load_yaml_config(config_path: Path, label: str):
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{label} 顶层必须是映射对象。")
    return data


def resolve_path(value: str):
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def collect_config_values(config_data, fields):
    return {field: config_data.get(field) for field in fields}


def apply_runtime_casts(runtime):
    for field in PATH_FIELDS:
        runtime[field] = str(resolve_path(runtime[field]))
    for field in INT_FIELDS:
        runtime[field] = int(runtime[field])
    for field in FLOAT_FIELDS:
        runtime[field] = float(runtime[field])
    for field in BOOL_FIELDS:
        runtime[field] = bool(runtime[field])
    runtime["lora_target_modules"] = parse_target_modules(runtime["lora_target_modules"])
    runtime["lora_bias"] = str(runtime["lora_bias"])


def validate_runtime_values(runtime):
    missing = [
        field
        for field in TRAIN_CONFIG_FIELDS + MODEL_CONFIG_FIELDS
        if runtime.get(field) is None
    ]
    if missing:
        raise ValueError(f"train/model 配置缺少字段: {', '.join(missing)}")

    for field in POSITIVE_FIELDS:
        if runtime[field] <= 0:
            raise ValueError(f"{field} 必须大于 0。")
    for field in NON_NEGATIVE_FIELDS:
        if runtime[field] < 0:
            raise ValueError(f"{field} 不能小于 0。")


def build_runtime_config(args):
    train_config_path = Path(args.config).expanduser().resolve()
    model_config_path = Path(args.model_config).expanduser().resolve()
    train_cfg = load_yaml_config(train_config_path, "config/train.yaml")
    model_cfg = load_yaml_config(model_config_path, "config/model.yaml")

    runtime = collect_config_values(train_cfg, TRAIN_CONFIG_FIELDS)
    runtime.update(collect_config_values(model_cfg, MODEL_CONFIG_FIELDS))

    cli_overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in {"config", "model_config", "dry_run", "simulate_only"} and value is not None
    }
    runtime.update(cli_overrides)

    validate_runtime_values(runtime)

    apply_runtime_casts(runtime)
    runtime["train_config_path"] = train_config_path
    runtime["model_config_path"] = model_config_path
    return runtime


def dump_runtime_config(runtime):
    printable = dict(runtime)
    printable["train_config_path"] = str(printable["train_config_path"])
    printable["model_config_path"] = str(printable["model_config_path"])
    return json.dumps(printable, ensure_ascii=False, indent=2)


def set_reproducibility(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_paths(runtime):
    model_path = Path(runtime["model_name_or_path"])
    data_path = Path(runtime["data_file_path"])
    checkpoint_dir = Path(runtime["checkpoint_dir"])

    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {data_path}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


class DataCollatorForCausalLMWith8xPadding:
    def __init__(self, tokenizer, max_length=None, pad_to_max_length=False, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        batch_max_len = self.max_length if self.pad_to_max_length else max(len(f["input_ids"]) for f in features)
        padded_len = int(math.ceil(batch_max_len / 8) * 8)
        batch = self.tokenizer.pad(
            features,
            padding="max_length",
            max_length=padded_len,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = self.label_pad_token_id
        batch["labels"] = labels
        return batch


class SortedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.iteration = 0
        self.lengths = [len(item["input_ids"]) for item in dataset]

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices.sort(key=lambda idx: self.lengths[idx], reverse=True)
        if self.shuffle:
            batch_indices = [
                indices[i : i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)
            ]
            rng = random.Random(self.seed + self.iteration)
            rng.shuffle(batch_indices)
            indices = [idx for batch in batch_indices for idx in batch]
        self.iteration += 1
        return iter(indices)

    def __len__(self):
        return len(self.dataset)


def load_tokenizer(runtime):
    tokenizer = AutoTokenizer.from_pretrained(
        runtime["model_name_or_path"],
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_training_dataset(runtime, tokenizer):
    dataset = load_dataset("json", data_files={"train": runtime["data_file_path"]})["train"]
    if "text" not in dataset.column_names:
        raise ValueError(f"训练数据缺少 text 字段: {dataset.column_names}")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=runtime["max_length"],
            return_attention_mask=True,
        )

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset


def build_data_loader(runtime, dataset, tokenizer):
    data_collator = DataCollatorForCausalLMWith8xPadding(
        tokenizer=tokenizer,
        max_length=runtime["max_length"],
        pad_to_max_length=not runtime["auto_pad_batch"],
        label_pad_token_id=-100,
    )
    sampler = SortedBatchSampler(
        dataset=dataset,
        batch_size=runtime["batch_size"],
        shuffle=runtime["shuffle"],
        seed=runtime["seed"],
    )
    train_loader = DataLoader(
        dataset,
        batch_size=runtime["batch_size"],
        sampler=sampler,
        collate_fn=data_collator,
        num_workers=runtime["num_workers"],
        pin_memory=runtime["pin_memory"],
    )
    return train_loader


def create_lora_config(runtime):
    return LoraConfig(
        r=runtime["lora_r"],
        lora_alpha=runtime["lora_alpha"],
        target_modules=runtime["lora_target_modules"],
        lora_dropout=runtime["lora_dropout"],
        bias=runtime["lora_bias"],
        task_type="CAUSAL_LM",
    )


def compute_schedule(runtime, train_loader):
    num_update_steps_per_epoch = max(
        1,
        math.ceil(len(train_loader) / runtime["gradient_accumulation_steps"]),
    )
    total_possible_steps = num_update_steps_per_epoch * runtime["epochs"]
    planned_training_steps = min(runtime["max_steps"], total_possible_steps)
    return {
        "num_update_steps_per_epoch": num_update_steps_per_epoch,
        "total_possible_steps": total_possible_steps,
        "planned_training_steps": planned_training_steps,
    }


def save_runtime_snapshot(runtime):
    checkpoint_dir = Path(runtime["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = checkpoint_dir / "resolved_train_runtime.json"
    with snapshot_path.open("w", encoding="utf-8") as fh:
        fh.write(dump_runtime_config(runtime))
    return snapshot_path


def plot_metrics(metrics, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    titles = ["Loss", "Perplexity", "Learning Rate", "Step Time", "Throughput"]
    colors = ["b", "orange", "red", "purple", "brown"]
    keys = ["loss", "perplexity", "lr", "step_time", "throughput"]

    for ax, title, key, color in zip(axes, titles, keys, colors):
        ax.plot(range(1, len(metrics[key]) + 1), metrics[key], marker="o", color=color)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)

    if len(axes) > len(keys):
        for ax in axes[len(keys):]:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def simulate_pipeline(runtime):
    validate_paths(runtime)
    set_reproducibility(runtime["seed"])
    tokenizer = load_tokenizer(runtime)
    dataset = load_training_dataset(runtime, tokenizer)
    train_loader = build_data_loader(runtime, dataset, tokenizer)
    lora_config = create_lora_config(runtime)
    schedule = compute_schedule(runtime, train_loader)
    first_batch = next(iter(train_loader))

    summary = {
        "dataset_size": len(dataset),
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


def load_model(runtime, lora_config):
    model = AutoModelForCausalLM.from_pretrained(
        runtime["model_name_or_path"],
        load_in_8bit=runtime["load_in_8bit"],
        device_map="auto",
        trust_remote_code=True,
    )
    if runtime["load_in_8bit"]:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    return model


def train(runtime):
    validate_paths(runtime)
    set_reproducibility(runtime["seed"])
    snapshot_path = save_runtime_snapshot(runtime)

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=runtime["gradient_accumulation_steps"],
    )

    tokenizer = load_tokenizer(runtime)
    dataset = load_training_dataset(runtime, tokenizer)
    train_loader = build_data_loader(runtime, dataset, tokenizer)
    lora_config = create_lora_config(runtime)
    schedule = compute_schedule(runtime, train_loader)

    model = load_model(runtime, lora_config)
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

    metrics = {
        "loss": [],
        "perplexity": [],
        "lr": [],
        "step_time": [],
        "throughput": [],
    }

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
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False, disable=not accelerator.is_local_main_process)

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
