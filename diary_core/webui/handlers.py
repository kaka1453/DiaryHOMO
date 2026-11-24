from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from diary_core.config.common import dump_runtime_config
from diary_core.config.infer_config import (
    build_webui_parser,
    build_webui_runtime_config,
)
from diary_core.infer.audit import (
    AUDIT_JSONL_FILENAME,
    append_audit_record,
    build_audit_record,
    read_jsonl,
    refresh_audit_artifacts_from_jsonl,
)
from diary_core.infer.diary_runtime import DiaryRuntime
from diary_core.infer.output_bundle import (
    OUTPUT_MD_FILENAME,
    prepare_output_bundle,
    write_parameters,
)
from diary_core.infer.prompt_io import format_guard_audit
from diary_core.model.loader import load_model_and_tokenizer
from diary_core.webui.state import Conversation, WebUIState


CONVERSATION_LOG_FILENAME = OUTPUT_MD_FILENAME


def load_model_handler(
    state: WebUIState,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    num_beams,
) -> str:
    with state.lock:
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
        state.app_config.update(prepare_output_bundle(runtime))
        write_parameters(runtime)
        state.tokenizer, state.model = load_model_and_tokenizer(runtime)
        return "模型加载完成"


def unload_model_handler(state: WebUIState, session: Conversation | None = None) -> str:
    with state.lock:
        if not state.is_loaded:
            return "模型未加载"
        state.clear_model()
        if session is not None and session is not state.default_session:
            session.clear()
        return "模型已卸载"


def clear_conversation(state: WebUIState, session: Conversation) -> str:
    with state.lock:
        session.clear()
    return ""


def append_conversation_log(
    state: WebUIState,
    role: str,
    prompt: str,
    reply: str,
    guard: dict | None = None,
    debug_dir: str | None = None,
    *,
    runtime=None,
    session=None,
) -> None:
    runtime = state.app_config if runtime is None else runtime
    session = state.default_session if session is None else session
    output_dir = Path(runtime["output_run_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / CONVERSATION_LOG_FILENAME
    audit = format_guard_audit(
        guard, debug_dir=debug_dir, audit_config=runtime.get("audit")
    )
    with md_path.open("a", encoding="utf-8") as fh:
        fh.write(f"## {len(session.history) // 2}\n")
        fh.write(f"{role}: {prompt}\n")
        fh.write(f"AI: {reply}{audit}\n\n---\n")


def generate_handler(
    state: WebUIState,
    prompt: str,
    save_md_checkbox,
    system: str = "",
    role: str = "",
    *,
    session=None,
):
    session = state.default_session if session is None else session
    with state.lock:
        if not state.is_loaded:
            return "请先加载模型"
        if not state.app_config.get("output_run_dir"):
            prepare_output_bundle(state.app_config)
            write_parameters(state.app_config)
        runtime = {**state.app_config, **state.generation_params}
        if session is not state.default_session:
            new_session = session.output_run_dir is None
            if new_session:
                session.output_run_dir = str(
                    Path(runtime["output_run_dir"]) / "sessions" / uuid4().hex
                )
            runtime["output_run_dir"] = session.output_run_dir
            prepare_output_bundle(runtime)
            if new_session:
                write_parameters(runtime)
        revision = (state.model_revision, session.revision)
        history = session.history.copy()
        diary_runtime = DiaryRuntime(runtime, state.tokenizer, state.model)

    result = diary_runtime.generate(
        prompt, conversation_history=history, system=system, role=role
    )
    with state.lock:
        if revision != (state.model_revision, session.revision):
            return None
        reply = result.final_text
        session.history.extend([f"{role}: {prompt}", f"AI: {reply}"])
        session.revision += 1
        audit_path = Path(runtime["output_run_dir"]) / AUDIT_JSONL_FILENAME
        audit_index = len(read_jsonl(audit_path)) + 1 if audit_path.exists() else 1
        audit_record = build_audit_record(
            index=audit_index, result=result, audit_config=runtime.get("audit")
        )
        append_audit_record(
            runtime["output_run_dir"], audit_record, runtime.get("audit")
        )
        refresh_audit_artifacts_from_jsonl(
            runtime["output_run_dir"], runtime.get("audit")
        )
        if save_md_checkbox:
            append_conversation_log(
                state,
                role,
                prompt,
                reply,
                guard=result.guard,
                debug_dir=result.debug_dir,
                runtime=runtime,
                session=session,
            )
        return reply


def _build_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    excluded = {"real_test", "message", "system", "role"}
    return argparse.Namespace(
        **{key: value for key, value in vars(args).items() if key not in excluded}
    )


def main() -> None:
    parser = build_webui_parser()
    parser.add_argument(
        "--real-test",
        dest="real_test",
        action="store_true",
        help="Load model and run one message.",
    )
    parser.add_argument("--message", default="今天测试一下拆分后的 WebUI 后端。")
    parser.add_argument("--system", default="你是包包。")
    parser.add_argument("--role", default="用户")
    args = parser.parse_args()

    if args.real_test:
        runtime = build_webui_runtime_config(_build_runtime_args(args))
        prepare_output_bundle(runtime)
        write_parameters(runtime)
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
        reply = generate_handler(
            state, args.message, runtime["save_md"], system=args.system, role=args.role
        )
        print(
            json.dumps(
                {"reply": reply, "history_items": len(state.conversation_history)},
                ensure_ascii=False,
                indent=2,
            )
        )
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
            "output_root": "generate/output",
            "output_name": "boa256日记",
            "output_run_dir": "generate/output/boa256日记/dev",
        }
    )
    print(
        json.dumps(
            {"handler": "ready", "history_items": len(state.conversation_history)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
