import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gradio as gr
import os
from pathlib import Path

# ================= 配置 =================
BASE_MODEL = "../../Cuming-models/Qwen/Qwen2.5-14B-Instruct"
CHECKPOINT_DIR = "../../HOMOboa-model/checkpoints/ckpt/qwen14b_boa_256/step_12000"

DEVICE = "cuda"
SAVE_MD = True  # 是否保存推理记录
OUTPUT_DIR = "./diary_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 默认参数
DEFAULT_PARAMS = {
    "max_new_tokens": 648,
    "temperature": 0.9,
    "top_p": 0.9,
    "top_k": 60,
    "repetition_penalty": 6.0,
    "num_beams": 3
}

# ================= 全局变量 =================
tokenizer = None
model = None
system_prompt = ""
role_name = ""
conversation_history = []


# ================= 模型加载/卸载 =================
def load_model(max_new_tokens, temperature, top_p, top_k, repetition_penalty, num_beams):
    global tokenizer, model, DEFAULT_PARAMS
    if model is not None:
        return "模型已加载"

    DEFAULT_PARAMS.update({
        "max_new_tokens": float(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "repetition_penalty": float(repetition_penalty),
        "num_beams": int(num_beams)
    })

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
    model.eval()

    return "模型加载完成"


def unload_model():
    global model, tokenizer, conversation_history
    if model is not None:
        del model
        del tokenizer
        torch.cuda.empty_cache()
        model = None
        tokenizer = None
        conversation_history = []
        return "模型已卸载"
    return "模型未加载"


# ================= 推理函数 =================
def generate(prompt, save_md_checkbox, system="", role=""):
    global conversation_history, model, tokenizer, system_prompt, role_name

    if model is None:
        return "请先加载模型"

    system_prompt = system
    role_name = role
    conversation_history.append(f"{role_name}: {prompt}")

    # 拼接对话
    full_prompt = ""
    if system_prompt:
        full_prompt += f"[系统]: {system_prompt}\n"
    full_prompt += "\n".join(conversation_history) + "\nAI:"

    inputs = tokenizer(
        [full_prompt],
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=DEFAULT_PARAMS["max_new_tokens"],
            temperature=DEFAULT_PARAMS["temperature"],
            top_p=DEFAULT_PARAMS["top_p"],
            top_k=DEFAULT_PARAMS["top_k"],
            repetition_penalty=DEFAULT_PARAMS["repetition_penalty"],
            num_beams=DEFAULT_PARAMS["num_beams"],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 取 AI 回复部分
    result_split = result.split("AI:")[-1].strip()
    conversation_history.append(f"AI: {result_split}")

    if save_md_checkbox:
        md_path = Path(OUTPUT_DIR) / "conversation.md"
        with open(md_path, "a", encoding="utf-8") as f:
            f.write(f"## {len(conversation_history) // 2}\n")
            f.write(f"{role_name}: {prompt}\n")
            f.write(f"AI: {result_split}\n\n---\n")

    return result_split


# ================= Gradio UI =================
with gr.Blocks() as demo:
    gr.Markdown("### 8bit Qwen WebUI 推理界面")

    with gr.Row():
        with gr.Column():
            max_new_tokens = gr.Number(value=DEFAULT_PARAMS["max_new_tokens"], label="max_new_tokens")
            temperature = gr.Number(value=DEFAULT_PARAMS["temperature"], label="temperature")
            top_p = gr.Number(value=DEFAULT_PARAMS["top_p"], label="top_p")
            top_k = gr.Number(value=DEFAULT_PARAMS["top_k"], label="top_k")
            repetition_penalty = gr.Number(value=DEFAULT_PARAMS["repetition_penalty"], label="repetition_penalty")
            num_beams = gr.Number(value=DEFAULT_PARAMS["num_beams"], label="num_beams")

            load_btn = gr.Button("加载模型")
            unload_btn = gr.Button("卸载模型")

        with gr.Column():
            system_input = gr.Textbox(lines=2, placeholder="系统提示词", label="系统提示词")
            role_input = gr.Textbox(lines=1, placeholder="角色名", label="角色名")
            prompt_input = gr.Textbox(lines=3, placeholder="请输入内容", label="输入")
            save_md_checkbox = gr.Checkbox(value=True, label="保存到MD")
            output_box = gr.Textbox(lines=10, label="AI输出")

    load_btn.click(
        load_model,
        inputs=[max_new_tokens, temperature, top_p, top_k, repetition_penalty, num_beams],
        outputs=output_box
    )
    unload_btn.click(unload_model, outputs=output_box)

    prompt_btn = gr.Button("生成/续写")
    prompt_btn.click(
        generate,
        inputs=[prompt_input, save_md_checkbox, system_input, role_input],
        outputs=output_box
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
