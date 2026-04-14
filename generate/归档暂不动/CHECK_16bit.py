import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ================= 配置 =================
BASE_MODEL = "../../Cuming-models/Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT_DIR = "../../HOMOboa-model/checkpoints/ka512-7b/step_10000"
DEVICE = "cuda"
INPUT_FILE = "../../diary_put/input.txt"
OUTPUT_FILE = "../../diary_put/output.md"
BATCH_SIZE = 10
MAX_NEW_TOKENS = 648                                      # 生成文本长度
TEMPERATURE = 0.9                                         # 采样温度，范围: 0.0~2.0，默认1.0。值越低越确定性，值越高越随机
TOP_P = 0.9                                               # nucleus sampling，范围0~1，默认1.0
TOP_K = 60                                                # top-k sampling，范围0~模型词表大小，默认0
REPETITION_PENALTY = 6.0                                  # 重复惩罚，>=1.0，默认1.0。值越大越避免重复
NUM_BEAMS = 3 #6                                          # 束搜索数，>=1，默认1（不使用束搜索），亲测系核心流畅、上下文关联参数，但谨防爆显存
DO_SAMPLE = True                                          # 是否使用采样，默认False。True开启随机性
# ============================================

# ================= GPU 加速优化 =================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

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
            max_new_tokens=MAX_NEW_TOKENS,         # 最大生成长度
            temperature=TEMPERATURE,               # 控制随机性，越低越确定性
            top_p=TOP_P,                           # nucleus sampling
            top_k=TOP_K,                           # top-k sampling
            repetition_penalty=REPETITION_PENALTY,# 重复惩罚
            num_beams=NUM_BEAMS,                   # 束搜索数量
            do_sample=DO_SAMPLE,                   # 是否使用采样
            pad_token_id=tokenizer.pad_token_id
        )
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return responses

# ================= 主逻辑 =================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read().strip()
prompts = content.split()  # 空格分隔引言

all_results = []
for i in range(0, len(prompts), BATCH_SIZE):
    batch_prompts = prompts[i:i+BATCH_SIZE]
    batch_responses = batch_generate(batch_prompts)

    for p, r in zip(batch_prompts, batch_responses):
        idx = len(all_results) + 1
        result = (
            f"## 第{idx}篇\n\n"
            f"引言: {p}\n\n"
            f"{r}\n\n"
            f"---\n"
        )
        print(result)
        all_results.append(result)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(all_results)

print(f"\n✅ 已完成生成，共 {len(all_results)} 篇，结果已写入 {OUTPUT_FILE}")
