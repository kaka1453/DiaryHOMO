import os
import datetime
import gradio as gr
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ================== 基本配置 ==================
BASE_MODEL = "../../Cuming-models/Qwen/Qwen2.5-7B-Instruct"
LORA_DIR = "../../HOMOboa-model/checkpoints/epoch_103"

LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ================== 加载 tokenizer ==================
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================== 加载 vLLM ==================
llm = LLM(
    model=BASE_MODEL,
    load_in_8bit=True,      # ✅ 8bit
    lora_config={
        "dir": LORA_DIR,
        "modules": None,
    },
)

# ================== 核心生成函数 ==================
def build_prompt(system_prompt, history, user_input):
    """
    把 system + 历史对话 + 当前输入 拼成一个 prompt
    """
    prompt = ""
    if system_prompt.strip():
        prompt += f"[系统]\n{system_prompt.strip()}\n\n"

    for u, a in history:
        prompt += f"[用户]\n{u}\n\n[助手]\n{a}\n\n"

    prompt += f"[用户]\n{user_input}\n\n[助手]\n"
    return prompt


def generate_reply(
    user_input,
    system_prompt,
    history,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    max_tokens,
    min_tokens,
    save_md,
):

    prompt = build_prompt(system_prompt, history, user_input)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=int(top_k),
        repetition_penalty=repetition_penalty,
        max_tokens=int(max_tokens),
        min_tokens=int(min_tokens),
    )

    outputs = llm.generate(
        [prompt],
        sampling_params=sampling_params,
        tokenizer=tokenizer,
    )

    reply = outputs[0].output_text.strip()

    history.append((user_input, reply))

    # ========== 保存 md ==========
    if save_md:
        ts = datetime.datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join(LOG_DIR, f"{ts}.md")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"## {datetime.datetime.now()}\n\n"
                f"**System:** {system_prompt}\n\n"
                f"**User:** {user_input}\n\n"
                f"**Assistant:**\n{reply}\n\n---\n"
            )

    return history, history


def clear_history():
    return [], []

# ================== Gradio UI ==================
with gr.Blocks(title="vLLM 8bit WebUI") as demo:

    gr.Markdown("## 🧠 vLLM + 8bit + LoRA 对话 WebUI")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)

            user_input = gr.Textbox(
                label="用户输入",
                placeholder="在这里输入内容，支持连续对话 / 续写",
                lines=4,
            )

            send_btn = gr.Button("发送 / 继续生成")
            clear_btn = gr.Button("清空对话")

        with gr.Column(scale=2):
            system_prompt = gr.Textbox(
                label="系统提示词 / 角色设定",
                placeholder="例如：你是一位写私人日记的AI，语言自然、克制、不说教。",
                lines=4,
            )

            temperature = gr.Slider(0.1, 1.5, value=0.9, step=0.05, label="temperature")
            top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="top_p")
            top_k = gr.Slider(0, 100, value=60, step=1, label="top_k")
            repetition_penalty = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="repetition_penalty")

            max_tokens = gr.Slider(64, 2048, value=512, step=32, label="max_new_tokens")
            min_tokens = gr.Slider(0, 512, value=80, step=10, label="min_new_tokens")

            save_md = gr.Checkbox(value=True, label="保存推理记录为 md")

    state = gr.State([])

    send_btn.click(
        generate_reply,
        inputs=[
            user_input,
            system_prompt,
            state,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            max_tokens,
            min_tokens,
            save_md,
        ],
        outputs=[chatbot, state],
    )

    clear_btn.click(clear_history, outputs=[chatbot, state])

demo.launch(server_name="0.0.0.0", server_port=7860)
