from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import os

# ================= 配置 =================
BASE_MODEL = "../../Cuming-models/Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT_DIR = "../../HOMOboa-model/checkpoints/ckpt/qwen7b_boa_256/epoch_103"

INPUT_FILE = "../../diary_put/input.txt"
OUTPUT_FILE = "../../diary_put/output.md"

BATCH_SIZE = 8
MAX_NEW_TOKENS = 648
TEMPERATURE = 0.9
TOP_P = 0.9
TOP_K = 60
REPETITION_PENALTY = 6.0
NUM_BEAMS = 3 #6                                          # 束搜索数，>=1，默认1（不使用束搜索），亲测系核心流畅、上下文关联参数，但谨防爆显存
# =======================================

# ================= tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# ================= vLLM 初始化 =================
llm = LLM(
    model=BASE_MODEL,
    dtype="float16",        # 对 3090 / A 系列最稳
    trust_remote_code=True,
    enable_lora=True,       # ✅ 打开 LoRA 功能
    gpu_memory_utilization=0.85,
    max_num_seqs=64,
    enforce_eager=True,
)

# ================= LoRA 请求 =================
# name: 任意字符串
# lora_int_id: 进程内唯一的 int（不要重复）
# lora_path: LoRA checkpoint 目录
lora_request = LoRARequest(
    lora_name="qwen7b_boa",
    lora_int_id=1,
    lora_path=CHECKPOINT_DIR,
)

# ================= 采样参数 =================
sampling_params = SamplingParams(
    max_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    repetition_penalty=REPETITION_PENALTY,
)

# ================= 读取输入 =================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

results = []

# ================= 批量推理 =================
for i in range(0, len(prompts), BATCH_SIZE):
    batch = prompts[i:i + BATCH_SIZE]

    outputs = llm.generate(
        batch,
        sampling_params,
        lora_request=lora_request,   # ✅ 关键点：在这里挂 LoRA
    )

    for j, out in enumerate(outputs):
        text = out.outputs[0].text
        idx = len(results) + 1
        block = (
            f"## 第{idx}篇\n\n"
            f"引言: {batch[j]}\n\n"
            f"{text}\n\n"
            f"---\n"
        )
        print(block)
        results.append(block)

# ================= 写入文件 =================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(results)

print(f"\n✅ vLLM 推理完成，共 {len(results)} 篇，已写入 {OUTPUT_FILE}")
