import os
import time
import math
import torch
from torch.utils.data import DataLoader, Sampler
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm import tqdm
import matplotlib.pyplot as plt
from peft import prepare_model_for_kbit_training

# ==================== 配置区 ====================
CONFIG = {
    "BASE_MODEL": "../../Cuming-models/Qwen/Qwen2.5-7B-Instruct",
    "CHECKPOINT_DIR": "../../HOMOboa-model/checkpoints/ka512-7b",
    "MAX_LENGTH": 512,
    "LOAD_IN_8BIT": True,

    "DATA_FILE_PATH": "../../json/ka_512.jsonl",
    "BATCH_SIZE": 3,
    "SHUFFLE": True,  # 总体上是否打乱 batch 排序

    # 自动 padding 开关
    "AUTO_PAD_BATCH": True,  # 测试显存关闭即可

    "LORA_R": 32,
    "LORA_ALPHA": 128,
    "LORA_TARGET_MODULES": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "LORA_DROPOUT": 0.2,
    "LORA_BIAS": "none",

    "EPOCHS": 2000,
    "GRADIENT_ACCUMULATION_STEPS": 4,
    "LR": 1e-5,
    "WARMUP_STEPS": 50
}
# ================================================

accelerator = Accelerator(
    mixed_precision="fp16",   # 自动 fp16，加速、减少显存
    gradient_accumulation_steps=CONFIG["GRADIENT_ACCUMULATION_STEPS"]
)

# ==================== 数据加载 ====================
dataset = load_dataset("json", data_files={"train": CONFIG["DATA_FILE_PATH"]})["train"]
tokenizer = AutoTokenizer.from_pretrained(CONFIG["BASE_MODEL"], use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=CONFIG["MAX_LENGTH"], return_attention_mask=True)
dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# ==================== DataCollator ====================
class DataCollatorForCausalLMWith8xPadding:
    def __init__(self, tokenizer, max_length=None, pad_to_max_length=False, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        batch_max_len = self.max_length if self.pad_to_max_length else max(len(f["input_ids"]) for f in features)
        padded_len = int(math.ceil(batch_max_len / 8) * 8)
        batch = self.tokenizer.pad(features, padding="max_length", max_length=padded_len, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = self.label_pad_token_id
        batch["labels"] = labels
        return batch

pad_to_max = not CONFIG["AUTO_PAD_BATCH"]
data_collator = DataCollatorForCausalLMWith8xPadding(
    tokenizer,
    max_length=CONFIG["MAX_LENGTH"],
    pad_to_max_length=pad_to_max,
    label_pad_token_id=-100
)

# ==================== BatchSampler ====================
class SortedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lengths = [len(item["input_ids"]) for item in dataset]

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices.sort(key=lambda i: self.lengths[i], reverse=True)
        if self.shuffle:
            batch_indices = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
            import random
            random.shuffle(batch_indices)
            indices = [i for batch in batch_indices for i in batch]
        return iter(indices)

    def __len__(self):
        return len(self.dataset)

sampler = SortedBatchSampler(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=CONFIG["SHUFFLE"])
# ==================== DataLoader ====================
train_loader = DataLoader(
    dataset,
    batch_size=CONFIG["BATCH_SIZE"],
    sampler=sampler,
    collate_fn=data_collator,
    num_workers=8,         # CPU 并行处理 batch
    pin_memory=True        # 直接拷贝到 GPU，加速
)

# ==================== 模型加载优化 ====================
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["BASE_MODEL"],
    load_in_8bit=CONFIG["LOAD_IN_8BIT"],
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)  # ⭐关键

model.gradient_checkpointing_enable()

peft_cfg = LoraConfig(
    r=CONFIG["LORA_R"],
    lora_alpha=CONFIG["LORA_ALPHA"],
    target_modules=CONFIG["LORA_TARGET_MODULES"],
    lora_dropout=CONFIG["LORA_DROPOUT"],
    bias=CONFIG["LORA_BIAS"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)


optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"])
num_update_steps_per_epoch = max(1, len(train_loader) // CONFIG["GRADIENT_ACCUMULATION_STEPS"])
num_training_steps = num_update_steps_per_epoch * CONFIG["EPOCHS"]
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=CONFIG["WARMUP_STEPS"], num_training_steps=num_training_steps)

# Accelerator 准备模型和优化器
model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_loader, lr_scheduler
)

# ==================== 训练监控数据 ====================
metrics = {
    "loss": [],
    "perplexity": [],
    "lr": [],
    "step_time": [],
    "throughput": []
}

# ==================== 绘图函数 ====================
def plot_metrics(metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    titles = ["Loss", "Perplexity", "Learning Rate", "Step Time", "Throughput"]
    colors = ["b", "orange", "red", "purple", "brown"]
    keys = ["loss", "perplexity", "lr", "step_time", "throughput"]

    for ax, title, key, color in zip(axes, titles, keys, colors):
        ax.plot(range(1, len(metrics[key])+1), metrics[key], marker="o", color=color)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)

    # 删除多余子图
    if len(axes) > len(keys):
        for ax in axes[len(keys):]:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# ==================== max_steps 训练控制 ====================
CONFIG["MAX_STEPS"] = 50000        # 按 tokens 控制，总训练步数
CONFIG["SAVE_STEPS"] = 2000        # 每隔多少 steps 保存一次

global_step = 0
epoch = 0
while global_step < CONFIG["MAX_STEPS"]:
    model.train()
    epoch_loss = 0.0
    epoch_lr = []
    epoch_step_time = []
    epoch_throughput = []
    start_time = time.time()
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

    for step, batch in enumerate(progress):
        if global_step >= CONFIG["MAX_STEPS"]:
            break

        step_start = time.time()
        outputs = model(**batch)
        loss = outputs.loss / CONFIG["GRADIENT_ACCUMULATION_STEPS"]
        accelerator.backward(loss)

        if (step + 1) % CONFIG["GRADIENT_ACCUMULATION_STEPS"] == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            epoch_loss += loss.item() * CONFIG["GRADIENT_ACCUMULATION_STEPS"]

            # 记录 lr、step_time、吞吐量
            epoch_lr.append(lr_scheduler.get_last_lr()[0])
            step_time = time.time() - step_start
            epoch_step_time.append(step_time)
            epoch_throughput.append(CONFIG["BATCH_SIZE"] / step_time)

            # ==================== 每隔 SAVE_STEPS 保存模型 ====================
            if global_step % CONFIG["SAVE_STEPS"] == 0:
                save_dir = os.path.join(CONFIG["CHECKPOINT_DIR"], f"step_{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(save_dir, safe_serialization=True)
                tokenizer.save_pretrained(save_dir)
                print(f"💾 模型已保存至 {save_dir}")

                metrics_path = os.path.join(CONFIG["CHECKPOINT_DIR"], "metrics_images", f"metrics_up_to_step_{global_step}.png")
                plot_metrics(metrics, metrics_path)
                print(f"📊 指标图已保存至 {metrics_path}")

    # 计算 epoch 平均值
    metrics["loss"].append(epoch_loss / len(train_loader))
    metrics["perplexity"].append(math.exp(epoch_loss / len(train_loader)))
    metrics["lr"].append(sum(epoch_lr)/len(epoch_lr))
    metrics["step_time"].append(sum(epoch_step_time)/len(epoch_step_time))
    metrics["throughput"].append(sum(epoch_throughput)/len(epoch_throughput))

    epoch += 1
    epoch_elapsed = time.time() - start_time
    print(f"Epoch {epoch} | avg_loss: {metrics['loss'][-1]:.4f} | time: {epoch_elapsed:.1f}s | speed: {epoch_elapsed/len(train_loader):.2f}s/step")

print("🎉 全部训练完成！")

