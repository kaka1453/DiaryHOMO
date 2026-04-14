import argparse
import json
from pathlib import Path

import gradio as gr
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "webui.yaml"
CONVERSATION_LOG_FILENAME = "conversation.md"

tokenizer = None
model = None
conversation_history = []
app_config = {}
generation_params = {}


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value}")


def build_parser():
    parser = argparse.ArgumentParser(description="8bit HuggingFace WebUI")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config file path.",
    )
    parser.add_argument(
        "--model-name-or-path",
        dest="model_name_or_path",
        type=str,
        default=None,
        help="Override model path from config.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        type=str,
        default=None,
        help="Override LoRA checkpoint path from config.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default=None,
        help="Override inference device from config.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Override markdown output directory from config.",
    )
    parser.add_argument(
        "--save-md",
        dest="save_md",
        type=str2bool,
        default=None,
        help="Override whether markdown logging is enabled by default.",
    )
    parser.add_argument(
        "--server-name",
        dest="server_name",
        type=str,
        default=None,
        help="Override Gradio server host from config.",
    )
    parser.add_argument(
        "--server-port",
        dest="server_port",
        type=int,
        default=None,
        help="Override Gradio server port from config.",
    )
    parser.add_argument(
        "--share",
        dest="share",
        type=str2bool,
        default=None,
        help="Override whether Gradio share is enabled.",
    )
    parser.add_argument(
        "--max-new-tokens",
        dest="max_new_tokens",
        type=int,
        default=None,
        help="Override default max_new_tokens in UI.",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=None,
        help="Override default temperature in UI.",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=None,
        help="Override default top_p in UI.",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="Override default top_k in UI.",
    )
    parser.add_argument(
        "--repetition-penalty",
        dest="repetition_penalty",
        type=float,
        default=None,
        help="Override default repetition_penalty in UI.",
    )
    parser.add_argument(
        "--num-beams",
        dest="num_beams",
        type=int,
        default=None,
        help="Override default num_beams in UI.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Only validate config and print resolved values without launching the WebUI.",
    )
    return parser


def load_yaml_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("config/webui.yaml 顶层必须是映射对象。")
    return data


def resolve_path(value: str):
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_runtime_config(args):
    config_path = Path(args.config).expanduser().resolve()
    raw_config = load_yaml_config(config_path)
    generation_cfg = raw_config.get("generation") or {}

    runtime = {
        "model_name_or_path": raw_config.get("model_name_or_path"),
        "checkpoint_dir": raw_config.get("checkpoint_dir"),
        "device": raw_config.get("device"),
        "output_dir": raw_config.get("output_dir"),
        "save_md": raw_config.get("save_md"),
        "server_name": raw_config.get("server_name"),
        "server_port": raw_config.get("server_port"),
        "share": raw_config.get("share"),
        "max_new_tokens": generation_cfg.get("max_new_tokens"),
        "temperature": generation_cfg.get("temperature"),
        "top_p": generation_cfg.get("top_p"),
        "top_k": generation_cfg.get("top_k"),
        "repetition_penalty": generation_cfg.get("repetition_penalty"),
        "num_beams": generation_cfg.get("num_beams"),
    }

    cli_overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in {"config", "dry_run"} and value is not None
    }
    runtime.update(cli_overrides)

    required_keys = [
        "model_name_or_path",
        "checkpoint_dir",
        "device",
        "output_dir",
        "save_md",
        "server_name",
        "server_port",
        "share",
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "num_beams",
    ]
    missing = [key for key in required_keys if runtime.get(key) is None]
    if missing:
        raise ValueError(f"webui 配置缺少字段: {', '.join(missing)}")

    runtime["config_path"] = config_path
    runtime["model_name_or_path"] = str(resolve_path(runtime["model_name_or_path"]))
    runtime["checkpoint_dir"] = str(resolve_path(runtime["checkpoint_dir"]))
    runtime["output_dir"] = str(resolve_path(runtime["output_dir"]))
    runtime["save_md"] = bool(runtime["save_md"])
    runtime["share"] = bool(runtime["share"])
    runtime["server_port"] = int(runtime["server_port"])
    runtime["max_new_tokens"] = int(runtime["max_new_tokens"])
    runtime["temperature"] = float(runtime["temperature"])
    runtime["top_p"] = float(runtime["top_p"])
    runtime["top_k"] = int(runtime["top_k"])
    runtime["repetition_penalty"] = float(runtime["repetition_penalty"])
    runtime["num_beams"] = int(runtime["num_beams"])
    return runtime


def dump_runtime_config(runtime):
    printable = dict(runtime)
    printable["config_path"] = str(printable["config_path"])
    return json.dumps(printable, ensure_ascii=False, indent=2)


def load_model(
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    num_beams,
):
    global tokenizer, model, generation_params
    if model is not None:
        return "模型已加载"

    generation_params.update(
        {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "num_beams": int(num_beams),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(
        app_config["model_name_or_path"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        app_config["model_name_or_path"],
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, app_config["checkpoint_dir"])
    model.eval()
    return "模型加载完成"


def unload_model():
    global tokenizer, model, conversation_history
    if model is None:
        return "模型未加载"

    del model
    del tokenizer
    torch.cuda.empty_cache()
    model = None
    tokenizer = None
    conversation_history = []
    return "模型已卸载"


def generate(prompt, save_md_checkbox, system="", role=""):
    global conversation_history

    if model is None:
        return "请先加载模型"

    conversation_history.append(f"{role}: {prompt}")

    full_prompt = ""
    if system:
        full_prompt += f"[系统]: {system.strip()}\n"
    full_prompt += "\n".join(conversation_history) + "\nAI:"

    inputs = tokenizer(
        [full_prompt],
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(app_config["device"])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=generation_params["max_new_tokens"],
            temperature=generation_params["temperature"],
            top_p=generation_params["top_p"],
            top_k=generation_params["top_k"],
            repetition_penalty=generation_params["repetition_penalty"],
            num_beams=generation_params["num_beams"],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = result.split("AI:")[-1].strip()
    conversation_history.append(f"AI: {reply}")

    if save_md_checkbox:
        output_dir = Path(app_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / CONVERSATION_LOG_FILENAME
        with md_path.open("a", encoding="utf-8") as fh:
            fh.write(f"## {len(conversation_history) // 2}\n")
            fh.write(f"{role}: {prompt}\n")
            fh.write(f"AI: {reply}\n\n---\n")

    return reply


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("### 8bit Qwen WebUI 推理界面")

        with gr.Row():
            with gr.Column():
                max_new_tokens = gr.Number(
                    value=generation_params["max_new_tokens"],
                    label="max_new_tokens",
                )
                temperature = gr.Number(
                    value=generation_params["temperature"],
                    label="temperature",
                )
                top_p = gr.Number(value=generation_params["top_p"], label="top_p")
                top_k = gr.Number(value=generation_params["top_k"], label="top_k")
                repetition_penalty = gr.Number(
                    value=generation_params["repetition_penalty"],
                    label="repetition_penalty",
                )
                num_beams = gr.Number(
                    value=generation_params["num_beams"],
                    label="num_beams",
                )

                load_btn = gr.Button("加载模型")
                unload_btn = gr.Button("卸载模型")

            with gr.Column():
                system_input = gr.Textbox(lines=2, placeholder="系统提示词", label="系统提示词")
                role_input = gr.Textbox(lines=1, placeholder="角色名", label="角色名")
                prompt_input = gr.Textbox(lines=3, placeholder="请输入内容", label="输入")
                save_md_checkbox = gr.Checkbox(
                    value=app_config["save_md"],
                    label="保存到MD",
                )
                output_box = gr.Textbox(lines=10, label="AI输出")

        load_btn.click(
            load_model,
            inputs=[
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                num_beams,
            ],
            outputs=output_box,
        )
        unload_btn.click(unload_model, outputs=output_box)

        prompt_btn = gr.Button("生成/续写")
        prompt_btn.click(
            generate,
            inputs=[prompt_input, save_md_checkbox, system_input, role_input],
            outputs=output_box,
        )

    return demo


def main():
    global app_config, generation_params

    parser = build_parser()
    args = parser.parse_args()
    app_config = build_runtime_config(args)
    generation_params = {
        "max_new_tokens": app_config["max_new_tokens"],
        "temperature": app_config["temperature"],
        "top_p": app_config["top_p"],
        "top_k": app_config["top_k"],
        "repetition_penalty": app_config["repetition_penalty"],
        "num_beams": app_config["num_beams"],
    }

    if args.dry_run:
        print("WebUI 配置检查通过。")
        print(dump_runtime_config(app_config))
        return

    Path(app_config["output_dir"]).mkdir(parents=True, exist_ok=True)
    demo = build_demo()
    demo.launch(
        server_name=app_config["server_name"],
        server_port=app_config["server_port"],
        share=app_config["share"],
    )


if __name__ == "__main__":
    main()
