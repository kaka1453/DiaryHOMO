import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

# =============== 配置 ===============
BASE_MODEL = "./Qwen1.5-4B"
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 1
LR = 2e-4
SAVE_STEPS = 50
MAX_LENGTH = 128

JSONL_FILES = [
    r"C:\Users\WEINI\Desktop\diaryHOMO\json\diary_2021.jsonl",
    r"C:\Users\WEINI\Desktop\diaryHOMO\json\diary_2022.jsonl",
    r"C:\Users\WEINI\Desktop\diaryHOMO\json\diary_2023.jsonl",
    r"C:\Users\WEINI\Desktop\diaryHOMO\json\diary_2024.jsonl"
]
# ====================================

accelerator = Accelerator()

# ========== 数据加载 ==========
datasets_list = [load_dataset("json", data_files=f)["train"] for f in JSONL_FILES]
train_dataset = concatenate_datasets(datasets_list)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)

# ========== 模型加载 ==========
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
peft_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer,
    num_warmup_steps=50,
    num_training_steps=(len(train_loader)//GRADIENT_ACCUMULATION_STEPS)*EPOCHS
)

# Accelerator 封装
model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

# ========== 训练 ==========
global_step = 0
for epoch in range(EPOCHS):
    model.train()
    print(f"===== Epoch {epoch+1}/{EPOCHS} =====")
    for step, batch in enumerate(train_loader):
        outputs = model(**batch)
        loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
        accelerator.backward(loss)

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 10 == 0:
                print(f"Step {global_step}, loss: {loss.item():.4f}")

            if global_step % SAVE_STEPS == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                save_dir = os.path.join(CHECKPOINT_DIR, f"step_{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                unwrapped_model.save_pretrained(save_dir, safe_serialization=True)
                tokenizer.save_pretrained(save_dir)
                print(f"💾 已保存 checkpoint 到 {save_dir}")

print("🎉 训练完成！")
