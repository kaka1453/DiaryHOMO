from __future__ import annotations

import argparse

import gradio as gr

from diary_core.config.common import dump_runtime_config
from diary_core.config.infer_config import build_webui_parser, build_webui_runtime_config
from diary_core.webui.handlers import generate_handler, load_model_handler, unload_model_handler
from diary_core.webui.state import WebUIState


def build_demo(state: WebUIState):
    with gr.Blocks() as demo:
        gr.Markdown("### 8bit Qwen WebUI 推理界面")

        with gr.Row():
            with gr.Column():
                max_new_tokens = gr.Number(
                    value=state.generation_params["max_new_tokens"],
                    label="max_new_tokens",
                )
                temperature = gr.Number(
                    value=state.generation_params["temperature"],
                    label="temperature",
                )
                top_p = gr.Number(value=state.generation_params["top_p"], label="top_p")
                top_k = gr.Number(value=state.generation_params["top_k"], label="top_k")
                repetition_penalty = gr.Number(
                    value=state.generation_params["repetition_penalty"],
                    label="repetition_penalty",
                )
                num_beams = gr.Number(
                    value=state.generation_params["num_beams"],
                    label="num_beams",
                )

                load_btn = gr.Button("加载模型")
                unload_btn = gr.Button("卸载模型")

            with gr.Column():
                system_input = gr.Textbox(lines=2, placeholder="系统提示词", label="系统提示词")
                role_input = gr.Textbox(lines=1, placeholder="角色名", label="角色名")
                prompt_input = gr.Textbox(lines=3, placeholder="请输入内容", label="输入")
                save_md_checkbox = gr.Checkbox(
                    value=state.app_config["save_md"],
                    label="保存到MD",
                )
                output_box = gr.Textbox(lines=10, label="AI输出")

        load_btn.click(
            lambda *args: load_model_handler(state, *args),
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
        unload_btn.click(lambda: unload_model_handler(state), outputs=output_box)

        prompt_btn = gr.Button("生成/续写")
        prompt_btn.click(
            lambda prompt, save_md, system, role: generate_handler(state, prompt, save_md, system, role),
            inputs=[prompt_input, save_md_checkbox, system_input, role_input],
            outputs=output_box,
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="WebUI app smoke test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    argv = ["--dry-run"]
    if args.config:
        argv.extend(["--config", args.config])
    runtime = build_webui_runtime_config(build_webui_parser().parse_args(argv))
    state = WebUIState(runtime)
    print(dump_runtime_config(runtime))
    if not args.dry_run:
        build_demo(state)


if __name__ == "__main__":
    main()

