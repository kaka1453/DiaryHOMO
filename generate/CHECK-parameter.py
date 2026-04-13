import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datetime import datetime
import json

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ================= 配置 =================
BASE_MODEL = "../../Cuming-models/Qwen1.5-4B"
CHECKPOINT_DIR = "../../HOMOboa-model/checkpoints/epoch_10"
DEVICE = "cuda"
INPUT_FILE = "../../diary_put/input.txt"
BATCH_SIZE = 6
MAX_NEW_TOKENS = 648
TEMPERATURE = 1.4
TOP_P = 0.9
TOP_K = 60
REPETITION_PENALTY = 6.0
NUM_BEAMS = 3
DO_SAMPLE = True
# ================= 输出文件时间命名 =================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"../../../checkpoint/miniled/202511419198/pths/memtestllmlog_{timestamp}.md"

# ================= 加载 tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================= 加载模型 =================
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
model.to(DEVICE)
model.eval()

# ================= 生成函数 =================
def batch_generate(prompts):
    """
    批量生成文本
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            num_beams=NUM_BEAMS,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id
        )
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return responses

# ================= 主逻辑 =================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read().strip()
prompts = content.split()  # 空格分隔引言

all_results = []

# 写入参数信息到第一行
param_info = {
    "BASE_MODEL": BASE_MODEL,
    "CHECKPOINT_DIR": CHECKPOINT_DIR,
    "BATCH_SIZE": BATCH_SIZE,
    "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
    "TEMPERATURE": TEMPERATURE,
    "TOP_P": TOP_P,
    "TOP_K": TOP_K,
    "REPETITION_PENALTY": REPETITION_PENALTY,
    "NUM_BEAMS": NUM_BEAMS,
    "DO_SAMPLE": DO_SAMPLE
}
all_results.append(f"<!-- 参数信息: {json.dumps(param_info, ensure_ascii=False)} -->\n\n")

for i in range(0, len(prompts), BATCH_SIZE):
    batch_prompts = prompts[i:i+BATCH_SIZE]
    batch_responses = batch_generate(batch_prompts)

    for p, r in zip(batch_prompts, batch_responses):
        idx = len(all_results)  # 序号从1开始算，包括参数信息占一行
        result = (
            f"## 第{idx}篇\n\n"
            f"引言: {p}\n\n"
            f"{r}\n\n"
            f"---\n"
        )
        print(result)
        all_results.append(result)

# ================= 写入文件时统一替换 /n 为 <br> =================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(r.replace("/n", "<br>") for r in all_results)

print(f"\n✅ 已完成生成，共 {len(all_results)-1} 篇，结果已写入 {OUTPUT_FILE}")