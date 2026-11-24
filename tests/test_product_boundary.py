from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from threading import Event

import torch

from diary_core.infer.diary_runtime import DiaryRuntime


class Batch(dict):
    def to(self, device):
        assert device == "cpu"
        return self


class Tokenizer:
    pad_token_id = 0

    def __init__(self):
        self.renders = []
        self.replies = iter(
            [
                "# unrelated heading",
                "蓝盒上的标签翘起来了。我按住一角，把新标签慢慢贴平。<END>后续部分",
            ]
        )

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.renders.append(messages)
        return "\n".join(row["content"] for row in messages)

    def __call__(self, prompts, **kwargs):
        return Batch(input_ids=torch.tensor([[2, 3, 4]]))

    def batch_decode(self, sequences, skip_special_tokens):
        assert sequences.tolist() == [[8, 9]] and skip_special_tokens
        return [next(self.replies)]


class Model:
    def __init__(self):
        self.options = []

    def generate(self, input_ids, **kwargs):
        self.options.append(kwargs)
        return torch.cat([input_ids, torch.tensor([[8, 9]])], dim=1)


def config():
    return dict(
        device="cpu",
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.9,
        top_k=12,
        repetition_penalty=1.1,
        num_beams=1,
        stop_sequences=["<END>"],
        prompt_debug={"enabled": False},
        diary_profile={"enabled": False},
        use_chat_template=True,
        guard={"enabled": True, "max_retry": 1, "retry_on_warnings": True},
    )


def probe():
    tokenizer, model = Tokenizer(), Model()
    result = DiaryRuntime(config(), tokenizer, model).generate(
        "我给蓝盒贴了新标签。",
        system="测试语气",
        role="测试员",
        conversation_history=["测试员: 蓝盒", "AI: 标签"],
    )
    assert (
        result.guard["retry_count"] == 1
        and len(model.options) == len(tokenizer.renders) == 2
    )
    assert "<END>" not in result.final_text
    return dict(
        messages=result.messages,
        renders=tokenizer.renders,
        requests=model.options,
        selected=result.guard["selected_attempt"],
        decisions=[a["guard"]["decision"] for a in result.guard["attempts"]],
        final=result.final_text,
        postprocess=result.postprocess,
    )


def test_prompt_retry_and_decode_match_baseline():
    data = json.dumps(probe(), sort_keys=True, ensure_ascii=False).encode()
    assert (
        sha256(data).hexdigest()
        == "5f9e4d83c39967bb3f7db32bbdf79e954f89fa1b7b685939f074d0520e45497e"
    )


def test_interleaved_browser_sessions_and_late_result(monkeypatch, tmp_path):
    from diary_core.webui import handlers
    from diary_core.webui.state import Conversation, WebUIState
    from diary_core.infer.audit import AUDIT_JSONL_FILENAME

    state = WebUIState(
        dict(
            config(),
            output_run_dir=str(tmp_path),
            model_name_or_path="local-test-base",
            checkpoint_dir="local-test-adapter",
            quantization_mode="none",
            save_md=True,
        )
    )
    tokenizer, model = Tokenizer(), Model()
    choices = []

    def load(runtime):
        choices.append(
            (
                runtime["model_name_or_path"],
                runtime["checkpoint_dir"],
                runtime["quantization_mode"],
            )
        )
        return tokenizer, model

    monkeypatch.setattr(handlers, "load_model_and_tokenizer", load)
    assert handlers.generate_handler(state, "before load", False) == "请先加载模型"
    assert (
        handlers.load_model_handler(state, 32, 0.7, 0.9, 12, 1.1, 1) == "模型加载完成"
    )
    assert choices == [("local-test-base", "local-test-adapter", "none")]
    a, b = Conversation(), Conversation()
    entered, resume = Event(), Event()
    histories = {}
    original = DiaryRuntime.generate

    def generate(runtime, prompt, **kwargs):
        assert runtime.model is model and runtime.tokenizer is tokenizer
        histories[prompt] = kwargs["conversation_history"]
        if prompt == "A-late":
            entered.set()
            assert resume.wait(15)
        # Execute message assembly, generation, guard and retry on synthetic state.
        runtime.tokenizer = Tokenizer()
        return original(runtime, prompt, **kwargs)

    monkeypatch.setattr(DiaryRuntime, "generate", generate)

    def send(session, prompt):
        return handlers.generate_handler(
            state, prompt, True, "synthetic system", "operator", session=session
        )

    send(a, "A-first")
    send(b, "B-first")
    old_a = Path(a.output_run_dir)
    b_dir = Path(b.output_run_dir)
    assert old_a != b_dir
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(send, a, "A-late")
        assert entered.wait(15)
        try:
            assert handlers.clear_conversation(state, a) == ""
            send(b, "B-next")
        finally:
            resume.set()
        assert future.result(timeout=15) is None
    assert not a.history and a.output_run_dir is None
    assert len(b.history) == 4 and histories["B-next"] == b.history[:2]
    assert histories["A-late"][0] == "operator: A-first"
    assert not any("A-" in msg for msg in b.history)
    assert len((old_a / AUDIT_JSONL_FILENAME).read_text().splitlines()) == 1
    assert "A-late" not in (old_a / "output.md").read_text()
    assert "A-first" not in (b_dir / "output.md").read_text()
    send(a, "A-new")
    assert histories["A-new"] == [] and Path(a.output_run_dir) != old_a
    assert handlers.unload_model_handler(state) == "模型已卸载"


def test_gradio_wires_independent_state_and_clear():
    import gradio as gr
    from diary_core.webui.app import build_demo
    from diary_core.webui.state import WebUIState

    demo = build_demo(WebUIState(dict(config(), save_md=False)))
    deps = demo.config["dependencies"]
    (state_id,) = [
        item["id"] for item in demo.config["components"] if item["type"] == "state"
    ]
    assert any(state_id in dep["inputs"] and len(dep["inputs"]) == 5 for dep in deps)
    assert any(dep["inputs"] == [state_id] and dep["queue"] is False for dep in deps)
    from gradio.state_holder import SessionState

    one, two = SessionState(demo), SessionState(demo)
    one[state_id].history.append("fixture")
    assert not two[state_id].history


def test_legacy_entrypoints_with_synthetic_product_config(tmp_path):
    import subprocess
    import sys

    cfg = config()
    cfg["generation"] = {
        key: cfg[key]
        for key in (
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "num_beams",
            "stop_sequences",
        )
    }
    cfg.update(
        model_name_or_path="synthetic-base",
        checkpoint_dir="synthetic-adapter",
        input_file="synthetic-input",
        output_root=str(tmp_path),
        batch_size=1,
        print_prompts=False,
        save_md=False,
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
    fixture = tmp_path / "product.yaml"
    fixture.write_text(json.dumps(cfg))
    root = Path(__file__).resolve().parents[1]
    for entry in ("generate/predict.py", "webui/webui_8bithf.py"):
        output = subprocess.run(
            [sys.executable, "-B", entry, "--config", str(fixture), "--dry-run"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        runtime = json.loads(output.stdout.split("\n", 1)[1])
        assert runtime["model_name_or_path"] == str(root / "synthetic-base")
        assert (
            runtime["checkpoint_dir"] == str(root / "synthetic-adapter")
            and not runtime["diary_profile"]["enabled"]
        )
    cfg["diary_profile"] = {
        "enabled": True,
        "profile_path": str(tmp_path / "missing-profile.yaml"),
    }
    fixture.write_text(json.dumps(cfg))
    missing = subprocess.run(
        [
            sys.executable,
            "-B",
            "generate/predict.py",
            "--dry-run",
            "--config",
            str(fixture),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert missing.returncode and "missing-profile.yaml" in missing.stderr
