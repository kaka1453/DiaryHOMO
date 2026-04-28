from __future__ import annotations

import argparse
import json
from pathlib import Path

from diary_core.config.common import dump_runtime_config
from diary_core.config.infer_config import build_webui_parser, build_webui_runtime_config
from diary_core.infer.diary_runtime import DiaryRuntime
from diary_core.model.loader import load_model_and_tokenizer
from diary_core.webui.state import WebUIState


CONVERSATION_LOG_FILENAME = "conversation.md"


def load_model_handler(
    state: WebUIState,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    num_beams,
) -> str:
    if state.is_loaded:
        return "模型已加载"

    state.generation_params.update(
        {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "num_beams": int(num_beams),
        }
    )
    runtime = {**state.app_config, **state.generation_params}
    state.tokenizer, state.model = load_model_and_tokenizer(runtime)
    return "模型加载完成"


def unload_model_handler(state: WebUIState) -> str:
    if not state.is_loaded:
        return "模型未加载"
    state.clear_model()
    return "模型已卸载"


def append_conversation_log(state: WebUIState, role: str, prompt: str, reply: str) -> None:
    output_dir = Path(state.app_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / CONVERSATION_LOG_FILENAME
    with md_path.open("a", encoding="utf-8") as fh:
        fh.write(f"## {len(state.conversation_history) // 2}\n")
        fh.write(f"{role}: {prompt}\n")
        fh.write(f"AI: {reply}\n\n---\n")


def generate_handler(state: WebUIState, prompt: str, save_md_checkbox, system: str = "", role: str = "") -> str:
    if not state.is_loaded:
        return "请先加载模型"

    runtime = {**state.app_config, **state.generation_params}
    diary_runtime = DiaryRuntime(runtime, state.tokenizer, state.model)
    result = diary_runtime.generate(
        prompt,
        conversation_history=state.conversation_history,
        system=system,
        role=role,
    )
    reply = result.final_text

    state.conversation_history.append(f"{role}: {prompt}")
    state.conversation_history.append(f"AI: {reply}")

    if save_md_checkbox:
        append_conversation_log(state, role, prompt, reply)

    return reply


def _build_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    excluded = {"real_test", "message", "system", "role"}
    return argparse.Namespace(**{key: value for key, value in vars(args).items() if key not in excluded})


def main() -> None:
    parser = build_webui_parser()
    parser.add_argument("--real-test", dest="real_test", action="store_true", help="Load model and run one message.")
    parser.add_argument("--message", default="今天测试一下拆分后的 WebUI 后端。")
    parser.add_argument("--system", default="你是包包。")
    parser.add_argument("--role", default="用户")
    args = parser.parse_args()

    if args.real_test:
        runtime = build_webui_runtime_config(_build_runtime_args(args))
        state = WebUIState(runtime)
        print(dump_runtime_config(runtime))
        print(
            load_model_handler(
                state,
                runtime["max_new_tokens"],
                runtime["temperature"],
                runtime["top_p"],
                runtime["top_k"],
                runtime["repetition_penalty"],
                runtime["num_beams"],
            )
        )
        reply = generate_handler(state, args.message, runtime["save_md"], system=args.system, role=args.role)
        print(json.dumps({"reply": reply, "history_items": len(state.conversation_history)}, ensure_ascii=False, indent=2))
        print(unload_model_handler(state))
        return

    state = WebUIState(
        {
            "max_new_tokens": 8,
            "temperature": 0.9,
            "top_p": 0.9,
            "top_k": 60,
            "repetition_penalty": 1.2,
            "num_beams": 1,
            "output_dir": "webui/diary_outputs",
        }
    )
    print(json.dumps({"handler": "ready", "history_items": len(state.conversation_history)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
