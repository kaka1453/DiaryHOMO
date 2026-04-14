import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ================= 配置 =================
BASE_MODEL = "../../Cuming-models/Qwen/Qwen2.5-14B-Instruct"
CHECKPOINT_DIR = "../../HOMOboa-model/checkpoints/ckpt/qwen14b_boa_256/step_12000"

INPUT_FILE = "../../diary_put/inputb.txt"
OUTPUT_FILE = "../../diary_put/output.md"

DEVICE = "cuda"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.9
TOP_P = 0.9
TOP_K = 60
REPETITION_PENALTY = 6.0
NUM_BEAMS = 3 #6                                          # 束搜索数，>=1，默认1（不使用束搜索），亲测系核心流畅、上下文关联参数，但谨防爆显存
# =======================================

torch.backends.cuda.matmul.allow_tf32 = True

# ================= tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================= 8bit 模型加载 =================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,       # 真正的 8bit
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
model.eval()

# ================= 推理函数 =================
def generate_batch(prompts):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            num_beams=NUM_BEAMS,  # 束搜索数量
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# ================= 主流程 =================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

results = []

for i in range(0, len(prompts), BATCH_SIZE):
    batch = prompts[i:i + BATCH_SIZE]
    texts = generate_batch(batch)

    for p, t in zip(batch, texts):
        idx = len(results) + 1
        block = (
            f"## 第{idx}篇\n\n"
            f"引言: {p}\n\n"
            f"{t}\n\n"
            f"---\n"
        )
        print(block)
        results.append(block)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(results)

print(f"\n✅ 8bit 推理完成，共 {len(results)} 篇，已写入 {OUTPUT_FILE}")
