from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from typing import Any

import torch

from diary_core.infer.generation import generation_kwargs, trim_stop_sequences
from diary_core.infer.prompt_builder import build_messages, render_prompt
from diary_core.infer.prompt_contract import DiaryContract, analyze_prompt, build_contract
from diary_core.infer.prompt_debug import PromptDebugRun, normalize_prompt_debug_config


HUMOR_PROFILE_STYLE_HINTS = {"搞笑", "幽默", "吐槽", "夸张", "戏剧化", "反差", "段子", "难蚌", "破防", "场景化"}
GENERATION_PROFILE_KEYS = {"temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens"}


@dataclass
class DiaryResult:
    raw_prompt: str
    final_text: str
    raw_model_output: str
    intent: dict
    contract: DiaryContract
    attachments: dict
    messages: list[dict]
    rendered_prompt: str
    generation_request: dict
    postprocess: dict
    guard: dict
    debug_dir: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["contract"] = self.contract.to_dict()
        return data


class DiaryRuntime:
    def __init__(self, runtime_config: dict, tokenizer, model):
        self.runtime_config = dict(runtime_config)
        self.tokenizer = tokenizer
        self.model = model

    def generate(self, raw_prompt: str, **kwargs) -> DiaryResult:
        runtime, context = self._build_call_runtime(kwargs)
        debug = PromptDebugRun(runtime.get("prompt_debug"))
        debug.write_text("00_raw_input.txt", raw_prompt)

        intent = analyze_prompt(raw_prompt)
        debug.write_json("01_intent.json", intent)

        contract = build_contract(raw_prompt, intent=intent, config=runtime.get("prompt_contract"))
        debug.write_json("02_contract.json", contract.to_dict())
        runtime = self.apply_generation_profile(runtime, contract)

        attachments = self.collect_attachments(contract, context)
        debug.write_json("03_attachments.json", attachments)

        messages = build_messages(contract, attachments, runtime.get("prompt_builder"))
        debug.write_json(
            "04_messages.json",
            messages,
            enabled=runtime["prompt_debug"]["include_messages"],
        )

        rendered_prompt = render_prompt(
            self.tokenizer,
            messages,
            use_chat_template=runtime.get("use_chat_template", True),
        )
        debug.write_text(
            "05_rendered_prompt.txt",
            rendered_prompt,
            enabled=runtime["prompt_debug"]["include_rendered_prompt"],
        )

        generation_request = self.build_generation_request(runtime)
        debug.write_json("06_generation_request.json", generation_request)

        raw_model_output = self.generate_rendered_prompt(rendered_prompt, runtime)
        debug.write_text(
            "07_raw_model_output.txt",
            raw_model_output,
            enabled=runtime["prompt_debug"]["include_model_output"],
        )

        final_text, postprocess = self.postprocess(raw_model_output, runtime)
        debug.write_json("08_postprocess.json", postprocess)

        guard = self.guard(contract, final_text, runtime)
        debug.write_json("09_guard.json", guard)
        debug.write_text("10_final_output.txt", final_text)
        debug.log("prompt_debug run completed")

        return DiaryResult(
            raw_prompt=raw_prompt,
            final_text=final_text,
            raw_model_output=raw_model_output,
            intent=intent,
            contract=contract,
            attachments=attachments,
            messages=messages,
            rendered_prompt=rendered_prompt,
            generation_request=generation_request,
            postprocess=postprocess,
            guard=guard,
            debug_dir=debug.run_dir_text,
        )

    def collect_attachments(self, contract: DiaryContract, context: dict) -> dict:
        items = []
        if context.get("system"):
            items.append({"type": "webui_system_note", "content": context["system"]})
        if context.get("conversation_history"):
            items.append(
                {
                    "type": "conversation_history",
                    "content": context["conversation_history"][-6:],
                    "note": "占位附件：后续可接入摘要、事实卡或 RAG。",
                }
            )
        if context.get("role"):
            items.append({"type": "webui_role", "content": context["role"]})
        return {
            "status": "placeholder",
            "items": items,
            "future_slots": ["style_card", "fact_card", "memory_summary", "rag_context"],
        }

    def build_generation_request(self, runtime: dict) -> dict:
        request = {
            "max_new_tokens": runtime["max_new_tokens"],
            "temperature": runtime["temperature"],
            "top_p": runtime["top_p"],
            "top_k": runtime["top_k"],
            "repetition_penalty": runtime["repetition_penalty"],
            "num_beams": runtime["num_beams"],
            "do_sample": True,
            "stop_sequences": runtime.get("stop_sequences", []),
            "use_chat_template": runtime.get("use_chat_template", True),
            "generation_profile": runtime.get("_generation_profile", {"enabled": False, "name": None}),
        }
        return request

    def apply_generation_profile(self, runtime: dict, contract: DiaryContract) -> dict:
        runtime = dict(runtime)
        profile_config = runtime.get("generation_profiles") or {}
        if not _as_bool(profile_config.get("enabled", False)):
            runtime["_generation_profile"] = {"enabled": False, "name": None, "reason": "generation_profiles disabled"}
            return runtime

        style_hints = set(getattr(contract, "style_hints", []))
        risk_tags = set(getattr(contract, "risk_tags", []))
        if "short_input_expansion_risk" in risk_tags:
            short_profile = {
                "temperature": 0.62,
                "top_p": 0.82,
                "top_k": 40,
                "repetition_penalty": 1.1,
                "max_new_tokens": 120,
            }
            short_profile.update(profile_config.get("short") or {})
            overrides = _apply_profile_overrides(runtime, short_profile)
            runtime["_generation_profile"] = {
                "enabled": True,
                "name": "short",
                "reason": "short_input_expansion_risk matched",
                "matched_style_hints": sorted(style_hints & HUMOR_PROFILE_STYLE_HINTS),
                "matched_risk_tags": sorted(risk_tags & {"short_input_expansion_risk"}),
                "overrides": overrides,
            }
            return runtime

        if "humor_or_absurd_prompt" not in risk_tags and not (style_hints & HUMOR_PROFILE_STYLE_HINTS):
            runtime["_generation_profile"] = {"enabled": True, "name": "default", "reason": "no style profile matched"}
            return runtime

        humor_profile = {
            "temperature": 0.78,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "max_new_tokens": 180,
        }
        humor_profile.update(profile_config.get("humor") or {})
        overrides = _apply_profile_overrides(runtime, humor_profile)
        runtime["_generation_profile"] = {
            "enabled": True,
            "name": "humor",
            "reason": "humor_or_absurd_prompt/style_hints matched",
            "matched_style_hints": sorted(style_hints & HUMOR_PROFILE_STYLE_HINTS),
            "matched_risk_tags": sorted(risk_tags & {"humor_or_absurd_prompt"}),
            "overrides": overrides,
        }
        return runtime

    def generate_rendered_prompt(self, rendered_prompt: str, runtime: dict) -> str:
        inputs = self.tokenizer(
            [rendered_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(runtime["device"])

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs(runtime, self.tokenizer))

        input_len = inputs["input_ids"].shape[-1]
        new_tokens = outputs[:, input_len:]
        texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return texts[0] if texts else ""

    def postprocess(self, text: str, runtime: dict) -> tuple[str, dict]:
        trimmed = trim_stop_sequences(text, runtime.get("stop_sequences", []))
        tail_trimmed = _trim_incomplete_tail(trimmed.strip())
        final = tail_trimmed.strip()
        return final, {
            "raw_length": len(text),
            "after_stop_length": len(trimmed.strip()),
            "final_length": len(final),
            "stop_sequences": runtime.get("stop_sequences", []),
            "tail_trimmed": final != trimmed.strip(),
            "changed": final != text.strip(),
        }

    def guard(self, contract: DiaryContract, final_text: str, runtime: dict) -> dict:
        return {
            "enabled": False,
            "status": "skipped",
            "decision": "pass",
            "reason": "DiaryGuard 尚未实现；当前仅保留 prompt_debug 占位结构。",
        }

    def _build_call_runtime(self, kwargs: dict) -> tuple[dict, dict]:
        runtime = dict(self.runtime_config)
        context_keys = {"conversation_history", "system", "role"}
        context = {key: kwargs.pop(key) for key in list(kwargs) if key in context_keys}

        prompt_debug_override = kwargs.pop("prompt_debug", None)
        prompt_debug_output_dir = kwargs.pop("prompt_debug_output_dir", None)
        runtime.update(kwargs)
        prompt_debug_config = normalize_prompt_debug_config(runtime.get("prompt_debug"))
        if prompt_debug_override is not None:
            prompt_debug_config["enabled"] = bool(prompt_debug_override)
        if prompt_debug_output_dir is not None:
            prompt_debug_config["output_dir"] = prompt_debug_output_dir
        runtime["prompt_debug"] = normalize_prompt_debug_config(prompt_debug_config)
        runtime.setdefault("stop_sequences", [])
        runtime.setdefault("use_chat_template", True)
        return runtime, context


def generate_diary(raw_prompt: str, runtime_config: dict | None = None, tokenizer=None, model=None) -> DiaryResult:
    if tokenizer is None or model is None:
        raise ValueError("generate_diary 第一版需要显式传入 tokenizer 和 model。")
    runtime = DiaryRuntime(runtime_config or {}, tokenizer, model)
    return runtime.generate(raw_prompt)


def _apply_profile_overrides(runtime: dict, profile: dict) -> dict:
    overrides = {key: profile[key] for key in GENERATION_PROFILE_KEYS if key in profile}
    if "max_new_tokens" in overrides:
        overrides["max_new_tokens"] = min(
            int(runtime.get("max_new_tokens", overrides["max_new_tokens"])),
            int(overrides["max_new_tokens"]),
        )
    runtime.update(overrides)

    stop_sequences = list(profile.get("stop_sequences") or [])
    if stop_sequences:
        merged = list(runtime.get("stop_sequences") or [])
        for sequence in stop_sequences:
            if sequence not in merged:
                merged.append(sequence)
        runtime["stop_sequences"] = merged
        overrides["stop_sequences_added"] = stop_sequences
    return overrides


def _trim_incomplete_tail(text: str) -> str:
    if not text:
        return text
    stripped = text.rstrip()
    if stripped.endswith(("。", "！", "？", "～", "~", "”", "」", "』")):
        return stripped
    last_positions = [stripped.rfind(mark) for mark in ("。", "！", "？", "～", "~")]
    last_pos = max(last_positions)
    if last_pos >= max(20, int(len(stripped) * 0.55)):
        return stripped[: last_pos + 1]
    return stripped


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = []
        for message in messages:
            rendered.append(f"<{message['role']}>\n{message['content']}")
        if add_generation_prompt:
            rendered.append("<assistant>\n")
        return "\n\n".join(rendered)

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
        max_len = max(len(text) for text in texts)
        return _FakeBatch({"input_ids": torch.ones((len(texts), max_len), dtype=torch.long)})

    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["今天和朋友吃饭，确实挺开心的。那种开心不是很夸张，就是回来的路上还会觉得今天没白过。\n--- 后面应该被截断"]


class _FakeModel:
    def generate(self, input_ids, **kwargs):
        new_tokens = torch.ones((input_ids.shape[0], 8), dtype=torch.long)
        return torch.cat([input_ids, new_tokens], dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="DiaryRuntime fake-model smoke test")
    parser.add_argument("--prompt", default="今天和朋友吃饭，很开心")
    parser.add_argument("--prompt-debug", action="store_true")
    parser.add_argument("--prompt-debug-output-dir", default="debug/prompt")
    args = parser.parse_args()

    runtime_config = {
        "device": "cpu",
        "max_new_tokens": 32,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.08,
        "num_beams": 1,
        "stop_sequences": ["\n---"],
        "use_chat_template": True,
        "prompt_debug": {
            "enabled": args.prompt_debug,
            "output_dir": args.prompt_debug_output_dir,
            "include_rendered_prompt": True,
            "include_messages": True,
            "include_model_output": True,
        },
    }
    result = DiaryRuntime(runtime_config, _FakeTokenizer(), _FakeModel()).generate(args.prompt)
    print(json.dumps({"final_text": result.final_text, "debug_dir": result.debug_dir}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
