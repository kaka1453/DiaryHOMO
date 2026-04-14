import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ========== 配置 ==========
BASE_MODEL = "../../Cuming-models/Qwen1.5-4B"  # 基础主模型路径
CHECKPOINT_DIR = "../../HOMOboa-model/checkpoints/step_400"  # 微调后权重路径
DEVICE = "cuda"  # GPU 测试
INPUT_FILE = "../../diary_put/input.txt"  # 存放引言的文件
OUTPUT_FILE = "../../diary_put/output.md"  # 保存结果的 markdown
BATCH_SIZE = 8                                           # 每批数量（可调）

# ========== 加载 tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ========== 加载模型 ==========
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
model.to(DEVICE)
model.eval()

# ========== 生成函数 ==========
def batch_generate(prompts, max_new_tokens=200, temperature=0.1, do_sample=True):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id
        )
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return responses

# ========== 主逻辑 ==========
# 1. 读取输入文件
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    content = f.read().strip()
prompts = content.split()  # 空格分隔

# 2. 批量生成
all_results = []
for i in range(0, len(prompts), BATCH_SIZE):
    batch_prompts = prompts[i:i+BATCH_SIZE]
    batch_responses = batch_generate(batch_prompts)

    for p, r in zip(batch_prompts, batch_responses):
        idx = len(all_results) + 1
        result = f"## 第{idx}篇\n\n引言: {p}\n\n{r}\n\n---\n"
        print(result)  # 控制台输出
        all_results.append(result)

# 3. 写入 Markdown
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(all_results)

print(f"\n✅ 已完成生成，共 {len(all_results)} 篇，结果已写入 {OUTPUT_FILE}")
