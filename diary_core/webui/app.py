from __future__ import annotations

from pathlib import Path

from diary_core.config.assets import build_or_exit, configure_caches
from diary_core.config.common import dump_runtime_config
from diary_core.config.infer_config import (
    build_webui_parser,
    build_webui_runtime_config,
    inference_summary,
)
from diary_core.infer.output_bundle import prepare_output_bundle, write_parameters


def build_demo(state):
    import gradio as gr

    from diary_core.webui.handlers import (
        clear_conversation,
        generate_handler,
        load_model_handler,
        unload_model_handler,
    )
    from diary_core.webui.state import Conversation

    def respond(prompt, save_md, system, role, conversation):
        reply = generate_handler(
            state, prompt, save_md, system, role, session=conversation
        )
        return gr.skip() if reply is None else reply

    with gr.Blocks() as demo:
        conversation = gr.State(Conversation())
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
                system_input = gr.Textbox(
                    lines=2, placeholder="系统提示词", label="系统提示词"
                )
                role_input = gr.Textbox(lines=1, placeholder="角色名", label="角色名")
                prompt_input = gr.Textbox(
                    lines=3, placeholder="请输入内容", label="输入"
                )
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
        unload_btn.click(
            lambda current: unload_model_handler(state, current),
            inputs=conversation,
            outputs=output_box,
        )

        prompt_btn = gr.Button("生成/续写")
        prompt_btn.click(
            respond,
            inputs=[
                prompt_input,
                save_md_checkbox,
                system_input,
                role_input,
                conversation,
            ],
            outputs=output_box,
        )

        clear_btn = gr.Button("清空当前会话")
        clear_btn.click(
            lambda current: clear_conversation(state, current),
            inputs=conversation,
            outputs=output_box,
            queue=False,
        )

    return demo


def main() -> None:
    parser = build_webui_parser()
    parser.add_argument("--init-only", action="store_true", help="Build the UI without launching a server.")
    args = parser.parse_args()
    initialize = vars(args).pop("init_only")
    runtime = build_or_exit(parser, build_webui_runtime_config, args)
    if args.dry_run:
        print(dump_runtime_config(inference_summary(runtime)))
        return
    configure_caches(Path(runtime["asset_root"]))
    from diary_core.webui.state import WebUIState

    prepare_output_bundle(runtime)
    write_parameters(runtime)
    demo = build_demo(WebUIState(runtime))
    if initialize:
        print(dump_runtime_config(inference_summary(runtime)))
        return
    demo.launch(server_name=runtime["server_name"], server_port=runtime["server_port"], share=runtime["share"])


if __name__ == "__main__":
    main()
