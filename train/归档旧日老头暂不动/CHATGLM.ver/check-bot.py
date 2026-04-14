import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ========== 配置 ==========
BASE_MODEL = "THUDM/chatglm-6b-int4"           # 基础模型
CHECKPOINT_DIR = "./checkpoints/best_model"    # LoRA 微调权重
DEVICE = "cuda"

MAX_CONTEXT_TOKENS = 512
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.8
DO_SAMPLE = True

# ========== 加载 tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ========== 加载模型 ==========
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,      # INT4 模型
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
model.eval()

# ========== 对话函数 ==========
def chat(context_text):
    inputs = tokenizer(context_text, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS).to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id
        )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text[len(context_text):]

# ========== 多轮对话 ==========
conversation_history = ""
print("🔹 多轮对话模式启动，输入 'exit' 或 'quit' 退出")

while True:
    user_input = input("你: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    conversation_history += f"你: {user_input}\n模型: "
    response = chat(conversation_history)
    print(f"模型: {response}\n")

    conversation_history += f"{response}\n"

    # 保持上下文长度不过长
    context_tokens = tokenizer(conversation_history)["input_ids"]
    if len(context_tokens) > MAX_CONTEXT_TOKENS:
        conversation_history = tokenizer.decode(context_tokens[-MAX_CONTEXT_TOKENS:], skip_special_tokens=True)
