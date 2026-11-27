"""D01/D03: drive the real Gradio app through ``gradio_client``.

These tests launch the actual app on a loopback port and exercise every UI
path: model list, unavailable marking, streaming, cancel, retry branch, new
session, parameter validation, card audit, blind review and benchmark reading.
"""

import json
import socket
import time
from pathlib import Path

import pytest

from lm_trainer.chat import ManualCardProvider, ModelInfo, RoleConfig
from lm_trainer.eval.provider import MOCK, MockRuntimeProvider
from lm_trainer.webui.app import build_demo, launch_args

ROOT = Path(__file__).resolve().parents[2]
EX = ROOT / "examples" / "vision_support"
CARDS = EX / "task_d_eval" / "cards_m0_m4.json"
BENCH = EX / "benchmark_results.json"

ROLE = RoleConfig(
    "CS_AGENT",
    "fixture-line-iris",
    "1",
    "Fictional machine-vision support. Separate observations from hypotheses.",
)
MODELS = (
    ModelInfo("base", "base", True, True),
    ModelInfo("sft_final", "adapter", True, True),
    ModelInfo("cpt_stage1", "adapter", False, None, None, "MODEL_UNAVAILABLE"),
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def ui():
    """Launch one app for the module; hand each test a fresh browser session."""

    gradio_client = pytest.importorskip("gradio_client")
    provider = MockRuntimeProvider(
        MODELS,
        (ROLE,),
        responder=lambda m, r, h, t: "虚构回复：请核对触发计数与存图数量。",
        memory_provider=ManualCardProvider(),
    )
    port = _free_port()
    demo = build_demo(
        provider,
        cards_fixture=CARDS,
        benchmark_path=BENCH,
    )
    kwargs = launch_args(server_port=port)
    assert kwargs["server_name"] == "127.0.0.1"
    assert kwargs["share"] is False
    # ``launch()`` blocks (prevent_thread_lock=False) and must serve Gradio's
    # event queue, so run it in a daemon thread while the test thread drives it.
    import threading

    ready = threading.Event()
    original_print = kwargs.get("quiet")

    def serve():
        demo.launch(**kwargs)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                ready.set()
                break
        except OSError:
            time.sleep(0.2)
    assert ready.is_set(), "Gradio server did not start"
    assert original_print is True
    url = f"http://127.0.0.1:{port}/"

    def new_client():
        # A fresh browser session per test prevents cross-test state leaks.
        return gradio_client.Client(url, verbose=False)

    try:
        yield new_client
    finally:
        demo.close()


def _send(client, text, model="base", role="CS_AGENT", **overrides):
    values = {
        "temp": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "rep": 1.0,
        "mnt": 64,
        "sample": False,
        "seed": 1234,
    }
    values.update(overrides)
    return client.predict(
        text,
        model,
        role,
        values["temp"],
        values["top_p"],
        values["top_k"],
        values["rep"],
        values["mnt"],
        values["sample"],
        values["seed"],
        api_name="/send",
    )


def _texts(frame) -> list[str]:
    out = []
    for message in frame[0]:
        content = message["content"]
        if isinstance(content, list):
            out.append("".join(part.get("text", "") for part in content))
        else:
            out.append(str(content))
    return out


def test_d01_send_streams_and_commits_real_history(ui):
    ui = ui()
    frame = _send(ui, "外触发模式下传送带提速后开始漏拍。")
    texts = _texts(frame)
    assert texts[0].startswith("外触发")
    assert "虚构回复" in texts[-1]
    # Real metrics: Mock has no token counters, so N/A is the honest answer.
    assert "input_tokens：N/A" in frame[5]
    assert "finish_reason：`eos`" in frame[5]


def test_d01_multi_turn_history_is_authoritative(ui):
    ui = ui()
    _send(ui, "第一问：触发计数与存图数量不一致。")
    frame = _send(ui, "第二问：曝光 5ms。")
    texts = _texts(frame)
    assert len(texts) >= 4, "history must grow from Task C, not be rebuilt locally"
    assert texts[0].startswith("第一问")
    assert texts[2].startswith("第二问")


def test_d01_model_switch_starts_a_new_session(ui):
    ui = ui()
    _send(ui, "切换前的会话。", model="base")
    ui.predict("sft_final", api_name="/on_model_change")
    frame = _send(ui, "切换后的会话。", model="sft_final")
    texts = _texts(frame)
    assert len(texts) == 2, "switching the model must start a new session"
    assert "切换前" not in "".join(texts)


def _error_text(frame) -> str:
    """Return the error markdown from a send frame.

    ``gr.State`` is omitted from the API response, so the tuple length varies.
    The error is always the only string that mentions 参数校验/不可用/拒绝.
    """

    for item in frame:
        if isinstance(item, str) and (
            "参数校验" in item or "不可用" in item or "拒绝请求" in item
        ):
            return item
    return ""


def test_d01_unavailable_model_is_refused(ui):
    ui = ui()
    label = "cpt_stage1 · unavailable（MODEL_UNAVAILABLE）"
    # Selecting an unavailable model marks the session dirty and records the
    # refusal; the next send must be refused before any model is loaded.
    ui.predict(label, api_name="/on_model_change")
    sent = _send(ui, "不可用模型提问", model=label)
    assert "不可用" in _error_text(sent)
    assert _texts(sent) == []


def test_d01_parameter_validation_rejects_out_of_range(ui):
    ui = ui()
    # Sliders are clamped by the browser, so the handler-level guard is checked
    # with a seed outside [0, 2**63-1], which no slider constrains.
    frame = _send(ui, "参数越界", seed=2**64)
    assert "参数校验失败" in _error_text(frame)
    assert _texts(frame) == []


def test_d01_empty_input_is_rejected(ui):
    ui = ui()
    frame = _send(ui, "   ")
    assert _texts(frame) == []


def test_d01_card_audit_reports_injection_and_rejection(ui):
    ui = ui()
    ui.predict(
        [
            "fictional-iris7-spec",
            "fictional-cross-namespace",
            "fictional-denied-read",
        ],
        api_name="/on_card_change",
    )
    frame = _send(ui, "IRIS-7 工位改过曝光吗？")
    audit = frame[4]
    assert "fictional-cross-namespace" in audit
    assert "namespace" in audit
    assert "read_denied" in audit
    assert "superseded" in audit
    # No card plaintext leaks into the audit view.
    assert "exposure 8ms" not in audit


def test_d01_new_session_clears_history(ui):
    ui = ui()
    _send(ui, "将被清空的会话。")
    frame = ui.predict(api_name="/new_session")
    assert _texts(frame) == []


def test_d01_cancel_reports_when_no_request_is_active(ui):
    ui = ui()
    frame = ui.predict(api_name="/cancel")
    message = next(x for x in frame if isinstance(x, str) and "取消" in x)
    assert "取消未生效" in message


def test_d01_retry_opens_a_fresh_branch(ui):
    ui = ui()
    _send(ui, "重试前的问题。")
    frame = ui.predict(
        "base", "CS_AGENT", 0.7, 0.9, 50, 1.0, 64, False, 1234, api_name="/retry"
    )
    texts = _texts(frame)
    assert len(texts) == 2, "retry must resend in a new session, not append"


def test_d02_blind_review_submit_and_reveal(ui, tmp_path):
    ui = ui()
    # Build a pack from a tiny mock run so the UI path is exercised end to end.
    from lm_trainer.eval.blind import make_blind_pack, save_blind_pack

    records = []
    index = 0
    for case_id in ("vision-eval-001", "vision-eval-002"):
        for model_id in ("base", "sft_final"):
            index += 1
            records.append(
                {
                    "case_id": case_id,
                    "model_id": model_id,
                    "kind": "fixed_history",
                    "memory_condition": "M0",
                    "event_group": f"demo:{case_id}",
                    "split": "dev",
                    "target_person": "CS_AGENT",
                    "turn_index": 0,
                    "prompt": f"{case_id} 的题目",
                    # Neutral text: the blind view rejects model identity hints.
                    "output": f"候选回复 {index}",
                }
            )
    pack, private = make_blind_pack(records, seed=42, mode=MOCK)
    pack_path, map_path = save_blind_pack(pack, private, tmp_path / "blind")
    loaded = ui.predict(str(pack_path), api_name="/load_blind")
    assert "匿名" in loaded[0]
    assert "pack_token:" in loaded[0]
    assert "model_id" not in json.dumps(loaded, ensure_ascii=False)
    token = loaded[0].split("pack_token: ")[1].split("`")[0].strip()
    status = ui.predict(
        "left",
        "虚构理由",
        "reviewer",
        str(tmp_path / "reviews"),
        token,
        "1",
        4,
        3,
        3,
        3,
        3,
        3,
        api_name="/submit",
    )
    assert "已提交" in status
    revealed = ui.predict(
        str(map_path), str(tmp_path / "reviews"), token, api_name="/do_reveal"
    )
    assert "揭盲汇总" in revealed
    assert "base" in revealed or "sft_final" in revealed


def test_d02_benchmark_rows_show_not_run_for_missing(ui):
    ui = ui()
    text = ui.predict(str(BENCH), api_name="/do_benchmark")
    assert "lm_trainer.benchmark.v1" in text
    assert "PASS" in text
    # The real fixture has no NOT RUN level, so check the sentinel rendering rule
    # with a synthetic aggregate that has one.
    synthetic = {
        "schema": "lm_trainer.benchmark.v1",
        "fixture_only": True,
        "status": "NOT RUN",
        "levels": [{"level": "L9", "status": "NOT RUN", "scope": "nothing"}],
        "acceptance": [],
        "aggregate": [],
    }
    path = Path(BENCH).parent / "task_d_eval" / "_tmp_bench_ui.json"
    path.write_text(json.dumps(synthetic), encoding="utf-8")
    try:
        text = ui.predict(str(path), api_name="/do_benchmark")
        assert "NOT RUN" in text
    finally:
        path.unlink()
