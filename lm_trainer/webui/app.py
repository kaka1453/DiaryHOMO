"""Local Gradio workbench: chat debugging and anonymous blind review.

Privacy defaults that are enforced structurally:

* the server binds ``127.0.0.1`` and ``share=False``; a public bind is rejected
  before the server starts;
* conversations are never written to a shared directory. Saving requires two
  independent explicit actions: enabling the save toggle **and** providing a
  private ``0700`` directory. Task C performs the actual write with ``0600``
  files;
* debug logging is a separate opt-in and only records metadata, never prompt or
  answer text.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import gradio as gr

from lm_trainer.chat import (
    ChatError,
    ContextCard,
    ModelInfo,
    RoleConfig,
)
from lm_trainer.eval.blind import (
    DIMENSION_LABELS,
    DIMENSIONS,
    load_blind_pack,
    load_reviews,
    reveal,
    submit_review,
)
from lm_trainer.eval.cards import build_conditions, load_card_fixture
from lm_trainer.eval.provider import REAL, RuntimeProvider
from lm_trainer.eval.report import benchmark_rows, load_benchmark_aggregate
from lm_trainer.eval.schema import EvalError
from lm_trainer.webui.markdown_safe import safe_markdown
from lm_trainer.webui.state import UIState

GENERATION_LIMITS = {
    "temperature": (0.0, 2.0, 0.7),
    "top_p": (0.0, 1.0, 0.9),
    "top_k": (0, 500, 50),
    "repetition_penalty": (0.5, 2.0, 1.0),
    "max_new_tokens": (1, 2048, 128),
}
SEED_MAX = 2**63 - 1
MAX_INPUT_CHARS = 4000


def launch_args(
    *,
    server_name: str | None = None,
    share: bool | None = None,
    server_port: int | None = None,
) -> dict:
    """Validate launch options and return safe ``launch()`` kwargs.

    Any bind other than loopback, or ``share=True``, is a hard error: this UI
    shows model output and review data and must not be exposed on a network.
    """

    host = (server_name or os.environ.get("GRADIO_SERVER_NAME") or "127.0.0.1").strip()
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise EvalError("PUBLIC_BIND_REFUSED", f"server_name {host!r} is not loopback")
    share_env = os.environ.get("GRADIO_SHARE", "").strip().lower()
    do_share = share if share is not None else share_env == "true"
    if do_share:
        raise EvalError("PUBLIC_SHARE_REFUSED", "share=True is not allowed")
    # ``prevent_thread_lock=False`` is required: Gradio serves its event queue
    # from the launch thread. Returning immediately would leave the queue unserved
    # and every generator handler would hang.
    kwargs = {
        "server_name": "127.0.0.1",
        "share": False,
        "prevent_thread_lock": False,
        "quiet": True,
        "show_error": True,
    }
    if server_port is not None:
        kwargs["server_port"] = int(server_port)
    return kwargs


def default_runtime_provider(args) -> RuntimeProvider:
    """Build the runtime provider from CLI-like arguments (single seam)."""

    from lm_trainer.eval.provider import (
        MockRuntimeProvider,
        RegistryRuntimeProvider,
        manual_cards_provider,
    )

    if getattr(args, "mock", False):
        payload = {}
        if getattr(args, "mock_responder_file", None):
            raw = json.loads(Path(args.mock_responder_file).read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload = raw
        model_ids = list(args.model or ["base", "sft_final"])
        models = tuple(
            ModelInfo(m, "base" if m == "base" else "adapter", True, True, None, None)
            for m in model_ids
        )
        role = RoleConfig(
            role_id=getattr(args, "role", "CS_AGENT") or "CS_AGENT",
            namespace="fixture-line-iris",
            version="1",
            system_prompt=(
                "You are a fictional machine-vision support agent. "
                "Separate observations from hypotheses and ask for missing facts."
            ),
        )

        def respond(model_id, role, history, text):
            return payload.get(model_id) or payload.get(
                "*", "请先确认触发计数与实际存图数量是否一致。"
            )

        return MockRuntimeProvider(models, (role,), responder=respond)
    if not getattr(args, "models", None) or not getattr(args, "roles", None):
        raise EvalError("SCHEMA_INVALID", "real runtime requires --models and --roles")
    return RegistryRuntimeProvider(
        args.models,
        args.roles,
        allow_fixture=bool(getattr(args, "allow_fixture", False)),
        base_override=getattr(args, "base", None),
        device_override=getattr(args, "device", None),
        memory_provider=manual_cards_provider(),
    )


def _model_choices(provider: RuntimeProvider) -> tuple[list[tuple[str, str]], str]:
    """Return (choices, first available) and mark unavailable models."""

    choices: list[tuple[str, str]] = []
    fallback = ""
    for info in provider.list_models():
        if info.available:
            choices.append((info.model_id, info.model_id))
            if not fallback:
                fallback = info.model_id
        else:
            label = f"{info.model_id} · unavailable（{info.reason or 'unavailable'}）"
            choices.append((label, label))
    return choices, fallback


def _settings(state: UIState, values: dict):
    from lm_trainer.chat import GenerationSettings

    def number(key: str, *, integer: bool = False):
        raw = values.get(key)
        if raw is None or isinstance(raw, bool):
            raise EvalError("INVALID_PARAMETER", f"{key} 缺失")
        try:
            value = int(raw) if integer else float(raw)
        except (TypeError, ValueError) as exc:
            raise EvalError("INVALID_PARAMETER", f"{key} 不是数字") from exc
        low, high, _ = GENERATION_LIMITS[key]
        if not low <= value <= high:
            raise EvalError(
                "INVALID_PARAMETER", f"{key}={value} 超出范围 [{low}, {high}]"
            )
        return value

    do_sample = bool(values.get("do_sample"))
    seed_raw = values.get("seed")
    seed: int | None = None
    if seed_raw not in (None, ""):
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError) as exc:
            raise EvalError("INVALID_PARAMETER", "seed 必须是整数") from exc
        if not 0 <= seed <= SEED_MAX:
            raise EvalError("INVALID_PARAMETER", f"seed 超出范围 [0, {SEED_MAX}]")
    return GenerationSettings(
        max_new_tokens=number("max_new_tokens", integer=True),
        do_sample=do_sample,
        temperature=number("temperature"),
        top_p=number("top_p"),
        top_k=number("top_k", integer=True),
        repetition_penalty=number("repetition_penalty"),
        seed=seed,
    )


def _model_id_from_choice(choice: str) -> str:
    return (choice or "").split(" · ")[0]


def _role_from_choice(choice: str) -> str:
    return (choice or "CS_AGENT").split(" · ")[0]


def _card_map(cards: tuple[ContextCard, ...]) -> dict[str, ContextCard]:
    return {c.card_id: c for c in cards}


def _audit_lines(audit) -> list[str]:
    lines = []
    for entry in audit or ():
        if entry.reason == "injected":
            lines.append(f"✅ {entry.card_id} v{entry.version} 注入成功")
        else:
            lines.append(f"⛔ {entry.card_id} v{entry.version} 拒载：{entry.reason}")
    return lines


def _fmt(value, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value}{suffix}"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def _status_line(state: UIState, provider: RuntimeProvider) -> str:
    mode = "真实推理" if provider.mode == REAL else "Mock（无真实推理）"
    model = state.model_id or "（未选择）"
    busy = " · 生成中" if state.busy else ""
    saved = " · 会话保存：开" if state.save_enabled else " · 会话保存：关"
    debug = " · 调试日志：开" if state.debug_enabled else " · 调试日志：关"
    session = state.session_id or "（新会话）"
    return (
        f"`fixture_only: true` · `mode: {provider.mode}`（{mode}）\n"
        f"- 当前模型：`{model}`{busy}\n"
        f"- 会话：`{session}`\n"
        f"- {saved}{debug}"
    )


def _empty_history() -> list[dict]:
    return []


def build_demo(
    provider: RuntimeProvider,
    *,
    cards_fixture: Path | str | None = None,
    benchmark_path: Path | str | None = None,
    blind_dir: Path | str | None = None,
) -> gr.Blocks:
    """Build the two-tab demo. All handlers close over the injected provider."""

    raw_cards: tuple[ContextCard, ...] = ()
    if cards_fixture:
        fixture, raw_cards = load_card_fixture(cards_fixture)
        build_conditions(fixture)  # validates all M0–M4 groups exist
    card_by_id = _card_map(raw_cards)

    model_choices, default_model = _model_choices(provider)
    role_choices = [role.role_id for role in _provider_roles(provider)]

    with gr.Blocks(title="lm_trainer · 本地客服调试台") as demo:
        state = gr.State(UIState().as_dict())
        with gr.Tabs():
            with gr.Tab("客服多轮对话调试"):
                with gr.Row():
                    with gr.Column(scale=2):
                        model_dd = gr.Dropdown(
                            choices=model_choices,
                            value=default_model,
                            label="模型（来自 Task C list_models）",
                            interactive=True,
                        )
                        role_dd = gr.Dropdown(
                            choices=role_choices,
                            value=role_choices[0] if role_choices else "CS_AGENT",
                            label="角色配置",
                            interactive=True,
                        )
                    with gr.Column(scale=3):
                        status_md = gr.Markdown(
                            _status_line(UIState(), provider), label="当前状态"
                        )
                chat = gr.Chatbot(
                    value=[],
                    label="多轮对话",
                    height=420,
                    show_label=True,
                    sanitize_html=True,
                    render_markdown=True,
                    group_consecutive_messages=False,
                    buttons=["copy"],
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="输入（虚构问题）",
                        placeholder="例：外触发模式下传送带提速后开始漏拍。",
                        lines=2,
                        max_lines=6,
                    )
                    with gr.Column(min_width=180):
                        send_btn = gr.Button("发送", variant="primary")
                        cancel_btn = gr.Button("取消", variant="stop")
                with gr.Row():
                    retry_btn = gr.Button("重试（分支：新会话重发）")
                    new_btn = gr.Button("新会话")
                    clear_btn = gr.Button("清空当前会话")
                error_box = gr.Markdown("", label="错误")
                with gr.Accordion("生成参数", open=False):
                    with gr.Row():
                        temperature = gr.Slider(
                            0.0, 2.0, 0.7, step=0.05, label="temperature"
                        )
                        top_p = gr.Slider(0.0, 1.0, 0.9, step=0.05, label="top_p")
                        top_k = gr.Slider(0, 500, 50, step=1, label="top_k")
                    with gr.Row():
                        repetition_penalty = gr.Slider(
                            0.5, 2.0, 1.0, step=0.05, label="repetition_penalty"
                        )
                        max_new_tokens = gr.Slider(
                            1, 2048, 128, step=1, label="max_new_tokens"
                        )
                        do_sample = gr.Checkbox(False, label="do_sample")
                        seed = gr.Number(
                            value=1234, label="seed（do_sample 时生效）", precision=0
                        )
                with gr.Accordion("设备工单卡片（候选，默认关闭）", open=False):
                    card_selector = gr.CheckboxGroup(
                        choices=sorted(card_by_id),
                        value=[],
                        label="候选测试卡片（仅注入 C 允许的 ID/版本）",
                    )
                    audit_md = gr.Markdown("（尚未发送请求）", label="实际注入审计")
                with gr.Row():
                    with gr.Column():
                        save_toggle = gr.Checkbox(
                            False, label="授权保存本会话到私有目录"
                        )
                        save_dir = gr.Textbox(
                            label="私有目录（必须已存在，权限 0700）",
                            placeholder="/path/to/private/0700/dir",
                        )
                        save_btn = gr.Button("保存本会话", interactive=False)
                    with gr.Column():
                        debug_toggle = gr.Checkbox(False, label="开启详细调试信息")
                        debug_md = gr.Markdown("（调试日志关闭）", label="调试信息")
                metrics_md = gr.Markdown("（尚无生成）", label="实测指标")

            with gr.Tab("匿名盲评与报告查看"), gr.Accordion(
                "加载匿名盲评任务包", open=True
            ), gr.Row():
                blind_path = gr.Textbox(
                    label="blind_pack.json 路径（不含模型名的盲评包）"
                )
                load_blind_btn = gr.Button("加载盲评包", variant="primary")
                with gr.Row():
                    pack_token = gr.Textbox(
                        label="pack_token（加载后自动填写）",
                        interactive=True,
                        scale=3,
                    )
                    pack_index = gr.Textbox(
                        label="当前项序号（1 起）", value="1", interactive=True, scale=1
                    )
                blind_status = gr.Markdown("（未加载）", label="盲评包状态")
                with gr.Row():
                    with gr.Column():
                        blind_prompt = gr.Markdown("（题目）", label="题目与上下文")
                    with gr.Column():
                        blind_left = gr.Markdown("（左）", label="输出 A（左）")
                        blind_right = gr.Markdown("（右）", label="输出 B（右）")
                with gr.Row():
                    blind_nav = gr.Radio(
                        choices=[], label="评审项", interactive=True, scale=2
                    )
                    blind_choice = gr.Radio(
                        choices=[
                            ("左优", "left"),
                            ("右优", "right"),
                            ("平局", "tie"),
                            ("均不合格", "both_fail"),
                            ("跳过", "skip"),
                        ],
                        label="成对偏好",
                        interactive=True,
                    )
                with gr.Accordion("分维度评分（1–5）", open=True):
                    score_inputs = []
                    with gr.Row():
                        for name in DIMENSIONS[:3]:
                            score_inputs.append(
                                gr.Slider(1, 5, 3, step=1, label=DIMENSION_LABELS[name])
                            )
                    with gr.Row():
                        for name in DIMENSIONS[3:]:
                            score_inputs.append(
                                gr.Slider(1, 5, 3, step=1, label=DIMENSION_LABELS[name])
                            )
                blind_reason = gr.Textbox(label="评审理由", lines=2)
                reviewer_name = gr.Textbox(label="评审人（可匿名）", value="anonymous")
                with gr.Row():
                    submit_btn = gr.Button("提交评审", variant="primary")
                    next_btn = gr.Button("下一项")
                    prev_btn = gr.Button("上一项")
                blind_review_status = gr.Markdown("", label="提交状态")
                with gr.Accordion("揭盲与汇总（需要私有映射文件）", open=False):
                    with gr.Row():
                        map_path = gr.Textbox(label="blind-map.json 路径（私有）")
                        review_dir = gr.Textbox(label="评审记录目录")
                        reveal_btn = gr.Button("揭盲汇总", variant="primary")
                    reveal_md = gr.Markdown("（尚未揭盲）", label="揭盲结果")
                with gr.Accordion("Task B 基准聚合指标", open=False):
                    benchmark_file = gr.Textbox(
                        label="benchmark_results.json 路径（可选）",
                        value=str(benchmark_path) if benchmark_path else "",
                    )
                    benchmark_btn = gr.Button("读取聚合指标")
                    benchmark_md = gr.Markdown("（未读取，缺失项将显示 NOT RUN）")

        # Blind-review pack registry. Only an opaque token and the current item
        # index travel through the UI; pack content and the private map path stay
        # server-side, so a browser cannot read another session's mapping.
        _packs: dict[str, dict] = {}

        def _new_pack(data: dict) -> str:
            import secrets

            token = secrets.token_urlsafe(16)
            _packs[token] = data
            return token

        def _get_pack(pack) -> dict:
            token = (pack or {}).get("token")
            if not token or token not in _packs:
                return {"loaded": False, "items": [], "index": 0}
            data = dict(_packs[token])
            data["index"] = (pack or {}).get("index", data.get("index", 0))
            return data

        def _load_state(value) -> UIState:
            return UIState.from_dict(value)

        def _save_state(s: UIState) -> dict:
            return s.as_dict()

        # ---- Tab 1 handlers -------------------------------------------------

        def on_model_change(choice, st):
            s = _load_state(st)
            model_id = _model_id_from_choice(choice)
            if not choice or "unavailable" in choice:
                s.last_error = "该模型不可用（unavailable），请选择其他模型。"
                s.note(f"model refused: {model_id}")
                return _save_state(s), _status_line(s, provider)
            s.model_dirty = True
            s.last_error = None
            s.note(f"model selected: {model_id} (new session on next send)")
            return _save_state(s), _status_line(s, provider)

        model_dd.change(
            on_model_change, [model_dd, state], [state, status_md], queue=False
        )

        def on_card_change(values, st):
            s = _load_state(st)
            s.selected_cards = tuple(values or ())
            s.note(f"cards selected: {len(s.selected_cards)}")
            return _save_state(s), gr.update()

        card_selector.change(
            on_card_change, [card_selector, state], [state, status_md], queue=False
        )

        def on_save_toggle(enabled, directory, st):
            s = _load_state(st)
            s.save_enabled = bool(enabled)
            s.save_dir = directory or ""
            if s.save_enabled:
                s.note("save authorised (writes only on explicit save)")
            return _save_state(s)

        save_toggle.change(
            on_save_toggle, [save_toggle, save_dir, state], [state], queue=False
        )
        save_dir.change(
            on_save_toggle, [save_toggle, save_dir, state], [state], queue=False
        )

        def on_debug_toggle(enabled, st):
            s = _load_state(st)
            s.debug_enabled = bool(enabled)
            if s.debug_enabled:
                s.note("debug logging enabled")
            return (
                _save_state(s),
                _status_line(s, provider),
                "调试日志：开（仅元数据）"
                if s.debug_enabled
                else "（调试日志关闭）",
            )

        debug_toggle.change(
            on_debug_toggle,
            [debug_toggle, state],
            [state, status_md, debug_md],
            queue=False,
        )

        def _refresh_metrics(s: UIState, result) -> str:
            s.last_result = {
                "request_id": result.request_id,
                "turn_id": result.turn_id,
                "finish_reason": result.finish_reason,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "ttft_ms": result.ttft_ms,
                "tokens_per_second": result.tokens_per_second,
                "latency_ms": result.latency_ms,
                "peak_vram_mb": result.peak_vram_mb,
                "error_code": result.error_code,
            }
            return (
                f"- finish_reason：`{result.finish_reason}`\n"
                f"- input_tokens：{_fmt(result.input_tokens, 0)}\n"
                f"- output_tokens：{_fmt(result.output_tokens, 0)}\n"
                f"- ttft_ms：{_fmt(result.ttft_ms)}\n"
                f"- tokens_per_second：{_fmt(result.tokens_per_second, 2)}\n"
                f"- latency_ms：{_fmt(result.latency_ms)}\n"
                f"- peak_vram_mb：{_fmt(result.peak_vram_mb, 1, ' MiB')}\n"
                f"- card_audit：{len(result.card_audit or ())} 条"
            )

        def _history_from_session(session_info) -> list[dict]:
            out = []
            for message in session_info.messages:
                out.append(
                    {
                        "role": message.role,
                        "content": safe_markdown(message.content),
                    }
                )
            return out

        def _send(text, st, params, model_choice=None, role_choice=None):
            """Generator: emit streaming partial replies, then a terminal frame.

            The authoritative history is read back from Task C after each turn,
            never from a UI component, so two browser sessions cannot observe each
            other's conversation.
            """

            s = _load_state(st)
            text = (text or "").strip()
            history: list[dict] = []
            audit_text = "（本次请求未注入卡片）"
            metrics = "（尚无生成）"

            def frame(*, chat_value, msg_value, error_text, audit, metric_text):
                return (
                    chat_value,
                    msg_value,
                    _save_state(s),
                    error_text,
                    _status_line(s, provider),
                    audit,
                    metric_text,
                )

            def early(error_text):
                yield frame(
                    chat_value=history,
                    msg_value="",
                    error_text=error_text,
                    audit=audit_text,
                    metric_text=metrics,
                )

            if not text:
                yield from early("")
                return
            if len(text) > MAX_INPUT_CHARS:
                s.last_error = f"输入过长（>{MAX_INPUT_CHARS} 字符）"
                yield from early(s.last_error)
                return
            try:
                settings = _settings(s, params)
            except EvalError as exc:
                s.last_error = f"参数校验失败：{exc.detail or exc.code}"
                yield from early(s.last_error)
                return
            model_choice = model_choice or ""
            model_id = _model_id_from_choice(model_choice)
            if not model_id or "unavailable" in model_choice:
                s.last_error = "当前模型不可用，请先选择可用模型。"
                yield from early(s.last_error)
                return
            runtime = provider.build(model_id)
            s.last_question = text
            if s.model_dirty or s.session_id is None or s.model_id != model_id:
                s.session_id = None
                history = []
                s.model_dirty = False
            try:
                if s.session_id is None:
                    role_id = _role_from_choice(role_choice or s.role_id)
                    session = runtime.create_session(model_id, role_id, settings)
                    s.role_id = role_id
                    s.session_id = session.session_id
                    s.model_id = model_id
                    history = []
                selected_cards = tuple(
                    card_by_id[c] for c in s.selected_cards if c in card_by_id
                )
                s.busy = True
                s.note(f"send: model={model_id} cards={len(selected_cards)}")
                placeholder = [
                    {"role": "user", "content": safe_markdown(text)},
                    {"role": "assistant", "content": ""},
                ]
                yield frame(
                    chat_value=history + placeholder,
                    msg_value="",
                    error_text="",
                    audit=audit_text,
                    metric_text="（生成中…）",
                )
                result = None
                with runtime.stream(s.session_id, text, list(selected_cards)) as events:
                    partial = ""
                    for event in events:
                        if event.kind == "delta":
                            partial += event.delta
                            yield frame(
                                chat_value=history
                                + [
                                    {"role": "user", "content": safe_markdown(text)},
                                    {
                                        "role": "assistant",
                                        "content": safe_markdown(partial),
                                    },
                                ],
                                msg_value="",
                                error_text="",
                                audit=audit_text,
                                metric_text="（生成中…）",
                            )
                        elif event.result is not None:
                            result = event.result
                s.busy = False
                if result is None:
                    raise ChatError("GENERATION_FAILED")
                session = runtime.get_session(s.session_id)
                history = _history_from_session(session)
                audit_text = (
                    "\n".join(_audit_lines(result.card_audit)) or "（无卡片审计）"
                )
                metrics = _refresh_metrics(s, result)
                if result.finish_reason in ("error", "cancel"):
                    s.last_error = (
                        f"生成未完成：`{result.finish_reason}`"
                        f"（`{result.error_code or 'cancelled'}`），未写入历史。"
                    )
                    s.note(f"terminal: {result.finish_reason} {result.error_code}")
                else:
                    s.last_error = None
                    s.note(f"done: {result.finish_reason} committed={result.committed}")
                yield frame(
                    chat_value=history,
                    msg_value="",
                    error_text=s.last_error or "",
                    audit=audit_text,
                    metric_text=metrics,
                )
            except ChatError as exc:
                s.busy = False
                s.last_error = f"Task C 拒绝请求：`{exc.code}`"
                s.note(f"chat error: {exc.code}")
                yield frame(
                    chat_value=history,
                    msg_value=text,
                    error_text=s.last_error,
                    audit=audit_text,
                    metric_text=metrics,
                )
            except EvalError as exc:
                s.busy = False
                s.last_error = f"{exc.code}：{exc.detail or ''}"
                yield frame(
                    chat_value=history,
                    msg_value=text,
                    error_text=s.last_error,
                    audit=audit_text,
                    metric_text=metrics,
                )
            finally:
                s.busy = False

        def _params(*values) -> dict:
            keys = (
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "max_new_tokens",
                "do_sample",
                "seed",
            )
            return dict(zip(keys, values))

        def send(text, st, model_choice, role_choice, *values):
            yield from _send(
                text, st, _params(*values), model_choice, role_choice
            )

        def cancel(st):
            s = _load_state(st)
            accepted = False
            if s.session_id and s.request_id:
                try:
                    runtime = provider.build(s.model_id)
                    accepted = runtime.cancel(s.session_id, s.request_id).accepted
                except ChatError as exc:
                    s.note(f"cancel error: {exc.code}")
            s.busy = False
            return (
                _save_state(s),
                f"取消{'已接受' if accepted else '未生效（请求可能已结束）'}",
                gr.update(),
            )

        def new_session(st):
            s = _load_state(st)
            s.session_id = None
            s.request_id = None
            s.last_error = None
            s.last_result = None
            s.model_dirty = False
            return (
                [],
                "",
                _save_state(s),
                "",
                _status_line(s, provider),
                "（尚无生成）",
            )

        def clear_session(st):
            s = _load_state(st)
            if s.session_id:
                try:
                    runtime = provider.build(s.model_id)
                    runtime.clear_session(s.session_id)
                except ChatError as exc:
                    s.last_error = f"清空失败：`{exc.code}`"
                    return (
                        [],
                        "",
                        _save_state(s),
                        s.last_error,
                        _status_line(s, provider),
                        "（尚无生成）",
                    )
            return (
                [],
                "",
                _save_state(s),
                "",
                _status_line(s, provider),
                "（尚无生成）",
            )

        def retry(st, model_choice, role_choice, *values):
            s = _load_state(st)
            s.session_id = None
            s.model_dirty = True
            s.last_error = None
            # Retry is an explicit branch: a fresh session re-sends the last
            # question stored in this browser's state.
            yield from _send(
                s.last_question or "", _save_state(s), _params(*values),
                model_choice, role_choice,
            )

        def save_session(st):
            s = _load_state(st)
            if not s.save_enabled:
                return (
                    _save_state(s),
                    "未授权：请先勾选“授权保存本会话到私有目录”。",
                )
            if not s.session_id:
                return _save_state(s), "当前没有可保存的会话。"
            directory = (s.save_dir or "").strip()
            if not directory:
                return _save_state(s), "缺少私有目录。"
            try:
                path = Path(directory).expanduser()
                mode = stat.S_IMODE(path.stat().st_mode)
                if not path.is_dir() or mode != 0o700:
                    return _save_state(s), "目录必须是已存在且权限为 0700 的私有目录。"
                runtime = provider.build(s.model_id)
                output = runtime.save_session(s.session_id, path)
                s.note("session saved by explicit request")
                return _save_state(s), f"已保存：`{output.name}`（0600）"
            except ChatError as exc:
                return _save_state(s), f"保存失败：`{exc.code}`"
            except OSError:
                return _save_state(s), "保存失败：目录不可访问。"

        # ---- wire Tab 1 -----------------------------------------------------

        send_inputs = [
            msg,
            state,
            model_dd,
            role_dd,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            max_new_tokens,
            do_sample,
            seed,
        ]
        send_outputs = [
            chat,
            msg,
            state,
            error_box,
            status_md,
            audit_md,
            metrics_md,
        ]
        send_event = send_btn.click(
            send,
            send_inputs,
            send_outputs,
            concurrency_limit=1,
            api_name="send",
        )
        msg.submit(
            send,
            send_inputs,
            send_outputs,
            concurrency_limit=1,
        )
        cancel_btn.click(
            cancel,
            [state],
            [state, error_box, status_md],
            cancels=[send_event],
            queue=False,
            api_name="cancel",
        )
        new_btn.click(
            new_session, [state], [chat, msg, state, error_box, status_md, metrics_md]
        )
        clear_btn.click(
            clear_session,
            [state],
            [chat, msg, state, error_box, status_md, metrics_md],
        )
        retry_btn.click(
            retry,
            [
                state,
                model_dd,
                role_dd,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_new_tokens,
                do_sample,
                seed,
            ],
            send_outputs,
            concurrency_limit=1,
        )
        save_btn.click(save_session, [state], [state, error_box], queue=False)
        save_toggle.change(
            lambda enabled: gr.Button(interactive=bool(enabled)),
            [save_toggle],
            [save_btn],
            queue=False,
        )

        # ---- Tab 2 handlers -------------------------------------------------

        def _render_blind(item: dict, index: int, total: int) -> str:
            prompt = item.get("prompt") or "（无题目文本）"
            return (
                f"### 第 {index + 1} / {total} 项\n\n"
                f"- case：`{item['case_id']}`\n"
                f"- event_group：`{item['event_group']}`\n"
                f"- 条件：`{item['memory_condition']}`\n\n"
                f"**题目与上下文**\n\n{safe_markdown(prompt)}"
            )

        def load_blind(path):
            path = (path or "").strip()
            if not path:
                return (
                    "请输入 blind_pack.json 路径。",
                    "（题目）",
                    "（左）",
                    "（右）",
                    gr.update(choices=[], value=None),
                )
            try:
                loaded = load_blind_pack(path)
            except EvalError as exc:
                return (
                    f"加载失败：`{exc.code}` {exc.detail or ''}".strip(),
                    "（题目）",
                    "（左）",
                    "（右）",
                    gr.update(choices=[], value=None),
                )
            items = [item.to_dict() for item in loaded.items]
            choices = [
                f"{i + 1}. {item['case_id']} · {item['memory_condition']}"
                for i, item in enumerate(items)
            ]
            data = {
                "items": items,
                "index": 0,
                "loaded": True,
                "pack_id": loaded.pack_id,
                "pack_path": path,
                "review_dir": "",
            }
            first = items[0]
            token = _new_pack(data)
            status_text = (
                f"已加载 `{loaded.pack_id}`：{len(items)} 项（匿名，无模型名）。"
                f"\n\n`pack_token: {token}`"
            )
            return (
                status_text,
                _render_blind(first, 0, len(items)),
                safe_markdown(first["left"]),
                safe_markdown(first["right"]),
                gr.update(choices=choices, value=choices[0]),
            )

        # The pack token is stored server-side by ``_new_pack``; the response
        # deliberately omits the state object so the API stays JSON-clean.
        load_blind_btn.click(
            load_blind,
            [blind_path],
            [
                blind_status,
                blind_prompt,
                blind_left,
                blind_right,
                blind_nav,
            ],
            queue=False,
        )

        def show_item(index_text, token_text):
            try:
                index = int(str(index_text).strip() or 1) - 1
            except ValueError:
                index = 0
            data = _get_pack({"token": (token_text or "").strip(), "index": 0})
            if not data.get("loaded"):
                return "（题目）", "（左）", "（右）", gr.update(), gr.update()
            items = data["items"]
            index = max(0, min(index, len(items) - 1))
            item = items[index]
            return (
                _render_blind(item, index, len(items)),
                safe_markdown(item["left"]),
                safe_markdown(item["right"]),
                gr.update(),
                gr.update(value=str(index + 1)),
            )

        def nav_change(choice, token_text):
            try:
                index = int(str(choice).split(".", 1)[0])
            except (TypeError, ValueError):
                index = 1
            return show_item(str(index), token_text)

        blind_nav.change(
            nav_change,
            [blind_nav, pack_token],
            [blind_prompt, blind_left, blind_right, blind_nav, pack_index],
            queue=False,
        )

        def step(delta, index_text, token_text):
            try:
                current = int(str(index_text).strip() or 1)
            except ValueError:
                current = 1
            return show_item(str(current + delta), token_text)

        next_btn.click(
            lambda i, t: step(1, i, t),
            [pack_index, pack_token],
            [blind_prompt, blind_left, blind_right, blind_nav, pack_index],
            queue=False,
        )
        prev_btn.click(
            lambda i, t: step(-1, i, t),
            [pack_index, pack_token],
            [blind_prompt, blind_left, blind_right, blind_nav, pack_index],
            queue=False,
        )

        def submit(choice, reason, reviewer, review_dir_value, token_text, index_text, *scores):
            try:
                current = int(str(index_text).strip() or 0)
            except ValueError:
                current = 0
            data = _get_pack({"token": (token_text or "").strip(), "index": current})
            if not data.get("loaded"):
                return "请先加载盲评包。"
            items = data["items"]
            index = current
            item = items[index]
            names = list(DIMENSIONS)
            score_map = {}
            for name, value in zip(names, scores):
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    continue
                if 1 <= number <= 5:
                    score_map[name] = number
            try:
                pack_obj = load_blind_pack(data.get("pack_path") or "")
                submit_review(
                    pack_obj,
                    review_dir_value or ".",
                    item_id=item["item_id"],
                    choice=choice or "skip",
                    scores=score_map or None,
                    reason=reason or "",
                    reviewer=reviewer or "anonymous",
                )
            except EvalError as exc:
                return f"提交失败：`{exc.code}` {exc.detail or ''}".strip()
            next_index = min(index + 1, len(items) - 1)
            return (
                f"已提交第 {index + 1} 项（`{choice or 'skip'}`）。"
                f"下一项序号：{next_index + 1}（请同步到“当前项序号”后继续）。"
            )

        submit_btn.click(
            submit,
            [
                blind_choice,
                blind_reason,
                reviewer_name,
                review_dir,
                pack_token,
                pack_index,
                *score_inputs,
            ],
            [blind_review_status],
            queue=False,
        )

        def do_reveal(map_file, review_dir_value, token_text):
            data = _get_pack({"token": (token_text or "").strip(), "index": 0})
            if not data.get("loaded"):
                return "请先加载盲评包。"
            map_file = (map_file or "").strip()
            if not map_file:
                return "缺少 blind-map.json 路径。"
            try:
                loaded = load_blind_pack(data.get("pack_path") or "")
                mapping = json.loads(Path(map_file).read_text(encoding="utf-8"))
                reviews = load_reviews(loaded, review_dir_value or ".")
                summary = reveal(loaded, mapping, reviews)
            except (EvalError, OSError, ValueError) as exc:
                detail = getattr(exc, "detail", "") or str(exc)
                return f"揭盲失败：`{getattr(exc, 'code', 'INVALID')}` {detail}"
            rows = [
                f"- `{key}`：{value}" for key, value in summary.counts.items()
            ]
            wins = [
                f"- `{model}`：{count}" for model, count in sorted(summary.wins.items())
            ]
            dims = []
            for name in DIMENSIONS:
                stats = summary.dimension_stats[name]
                dims.append(
                    f"| `{name}` | {stats['count']} | {stats['mean'] if stats['mean'] is not None else 'N/A'} | {stats['missing']} |"
                )
            return (
                f"### 揭盲汇总\n\n"
                f"- 独立 case：{summary.independent_cases}；独立 event：{summary.independent_events}\n"
                f"- 已评审：{summary.reviewed_count} / {summary.item_count}\n\n"
                f"**胜场**\n" + "\n".join(wins) + "\n\n"
                "**选择分布**\n" + "\n".join(rows) + "\n\n"
                "| 维度 | 样本数 | 均值 | 缺失 |\n|---|---:|---:|---:|\n"
                + "\n".join(dims)
            )

        reveal_btn.click(
            do_reveal,
            [map_path, review_dir, pack_token],
            [reveal_md],
            queue=False,
        )

        def do_benchmark(path):
            path = (path or "").strip()
            if not path:
                return "请输入 benchmark_results.json 路径。"
            try:
                aggregate = load_benchmark_aggregate(path)
            except EvalError as exc:
                return f"读取失败：`{exc.code}`"
            lines = [
                f"**schema**：`{aggregate.schema}`　**status**：`{aggregate.status}`",
                "",
                "| 范围 | 状态 | 指标 | 值 |",
                "|---|---|---|---:|",
            ]
            for row in benchmark_rows(aggregate):
                value = row["value"]
                if value is None:
                    value = "NOT RUN"
                lines.append(
                    f"| `{row['scope']}` | {row['status']} | {row['metric']} | {value} |"
                )
            lines += ["", "> 缺失项显示 NOT RUN，不补造数字。"]
            return "\n".join(lines)

        benchmark_btn.click(do_benchmark, [benchmark_file], [benchmark_md], queue=False)

    demo.css = _CSS
    return demo


_CSS = """
footer {display:none !important}
.gradio-container {max-width: 1280px !important; margin: auto !important}
"""


def _provider_roles(provider: RuntimeProvider) -> tuple[RoleConfig, ...]:
    roles = getattr(provider, "roles", None)
    if roles:
        return tuple(roles)
    return (
        RoleConfig(
            role_id="CS_AGENT",
            namespace="fixture-line-iris",
            version="1",
            system_prompt="Fictional machine-vision support agent.",
        ),
    )


def create_app(
    provider: RuntimeProvider,
    **kwargs,
) -> gr.Blocks:
    return build_demo(provider, **kwargs)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Task D local Gradio workbench")
    parser.add_argument("--models", type=Path, help="chat-models.v1 JSON")
    parser.add_argument("--roles", type=Path, help="chat-roles.v1 JSON")
    parser.add_argument("--model", action="append", help="model_id to expose")
    parser.add_argument("--role", default="CS_AGENT")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--allow-fixture", action="store_true")
    parser.add_argument("--mock", action="store_true", help="offline Mock runtime")
    parser.add_argument("--mock-responder-file", type=Path)
    parser.add_argument("--cards", type=Path, help="fictional card fixture")
    parser.add_argument("--benchmark", type=Path, help="Task B aggregate JSON")
    parser.add_argument("--blind-dir", type=Path, help="directory with blind_pack.json")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        help="loopback only; any other host is refused",
    )
    args = parser.parse_args(argv)
    provider = default_runtime_provider(args)
    kwargs = launch_args(server_name=args.server_name, server_port=args.port)
    demo = build_demo(
        provider,
        cards_fixture=args.cards,
        benchmark_path=args.benchmark,
        blind_dir=args.blind_dir,
    )
    demo.launch(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
