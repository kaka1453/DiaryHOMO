from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from typing import Any

import torch

from diary_core.infer.generation import generation_kwargs, trim_stop_sequences
from diary_core.infer.guard import GuardResult, guard_diary, normalize_guard_config
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

        guard_result = self.guard(contract, final_text, runtime)
        attempts = [
            self.build_attempt_record(
                attempt=1,
                final_text=final_text,
                raw_model_output=raw_model_output,
                rendered_prompt=rendered_prompt,
                generation_request=generation_request,
                postprocess=postprocess,
                guard_result=guard_result,
            )
        ]
        debug.write_json("09_guard_attempt_1.json", guard_result.to_dict())

        guard_config = normalize_guard_config(runtime.get("guard"))
        previous_text = final_text
        previous_guard = guard_result
        for retry_index in range(1, guard_config["max_retry"] + 1):
            if not self.should_retry(previous_guard, guard_config):
                break

            retry_messages = self.build_retry_messages(contract, previous_text, previous_guard)
            debug.write_json(
                f"09_retry_{retry_index}_messages.json",
                retry_messages,
                enabled=runtime["prompt_debug"]["include_messages"],
            )
            retry_rendered_prompt = render_prompt(
                self.tokenizer,
                retry_messages,
                use_chat_template=runtime.get("use_chat_template", True),
            )
            debug.write_text(
                f"09_retry_{retry_index}_rendered_prompt.txt",
                retry_rendered_prompt,
                enabled=runtime["prompt_debug"]["include_rendered_prompt"],
            )
            retry_generation_request = self.build_generation_request(runtime)
            debug.write_json(f"09_retry_{retry_index}_generation_request.json", retry_generation_request)

            retry_raw_model_output = self.generate_rendered_prompt(retry_rendered_prompt, runtime)
            debug.write_text(
                f"09_retry_{retry_index}_raw_model_output.txt",
                retry_raw_model_output,
                enabled=runtime["prompt_debug"]["include_model_output"],
            )
            retry_final_text, retry_postprocess = self.postprocess(retry_raw_model_output, runtime)
            debug.write_json(f"09_retry_{retry_index}_postprocess.json", retry_postprocess)

            retry_guard = self.guard(contract, retry_final_text, runtime)
            debug.write_json(f"09_guard_attempt_{retry_index + 1}.json", retry_guard.to_dict())
            attempts.append(
                self.build_attempt_record(
                    attempt=retry_index + 1,
                    final_text=retry_final_text,
                    raw_model_output=retry_raw_model_output,
                    rendered_prompt=retry_rendered_prompt,
                    generation_request=retry_generation_request,
                    postprocess=retry_postprocess,
                    guard_result=retry_guard,
                )
            )
            previous_text = retry_final_text
            previous_guard = retry_guard
            if retry_guard.decision == "pass":
                break
            if retry_guard.decision == "pass_with_warnings" and not guard_config.get("retry_on_warnings"):
                break

        selected_attempt = self.select_attempt(attempts, guard_config)
        final_text = selected_attempt["final_text"]
        raw_model_output = selected_attempt["raw_model_output"]
        rendered_prompt = selected_attempt["rendered_prompt"]
        generation_request = selected_attempt["generation_request"]
        postprocess = selected_attempt["postprocess"]
        guard = self.build_guard_summary(attempts, selected_attempt, guard_config)
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

    def guard(self, contract: DiaryContract, final_text: str, runtime: dict) -> GuardResult:
        return guard_diary(final_text, contract, runtime.get("guard"))

    def should_retry(self, guard_result: GuardResult, guard_config: dict) -> bool:
        if not guard_result.enabled or guard_config["max_retry"] <= 0:
            return False
        if guard_result.decision == "pass":
            return False
        if guard_result.decision == "pass_with_warnings":
            return bool(guard_config.get("retry_on_warnings", False))
        if guard_result.decision == "revise":
            return guard_config["retry_on_revise"]
        return True

    def build_retry_messages(self, contract: DiaryContract, previous_draft: str, guard_result: GuardResult) -> list[dict]:
        guard_data = guard_result.to_dict()
        user_content = "\n".join(
            [
                "[ORIGINAL_CONTRACT]",
                f"MAIN_TOPIC: {contract.main_topic}",
                f"TOPIC_TERMS: {_format_inline_list(contract.topic_terms)}",
                f"FORBIDDEN_DRIFT_TOPICS: {_format_inline_list(contract.forbidden_drift_topics)}",
                "",
                "[PREVIOUS_DRAFT]",
                previous_draft.strip(),
                "",
                "[GUARD_FAILURE]",
                "失败原因：",
                _format_bullets(guard_data.get("reasons") or []),
                f"命中禁区词：{_format_forbidden_hits(guard_data.get('forbidden_hits') or [])}",
                f"缺失主题词：{_format_inline_list(guard_data.get('missing_topic_terms') or [])}",
                f"格式问题：{_format_inline_list(guard_data.get('format_hits') or [])}",
                f"语言噪声：{_format_inline_list(guard_data.get('language_noise_hits') or [])}",
                f"质量警告：{_format_inline_list(guard_data.get('warnings') or guard_data.get('quality_warnings') or [])}",
                "",
                "[REWRITE_TASK]",
                "请重写为一篇新的日记正文。",
                "必须围绕 MAIN_TOPIC 和 TOPIC_TERMS。",
                "删除所有禁区内容、格式漂移和语言噪声。",
                "不要解释，不要列点，不要输出标题，不要 Markdown。",
                "",
                "[REWRITE_STYLE_REQUIREMENT]",
                "重写时不要只复述 MAIN_TOPIC；请保留主旨，同时在主旨范围内补充 1-2 个具体动作、内心吐槽或小对话。",
                "允许的合理展开：现场动作、内心 OS、一句对话、轻微夸张比喻、与主题直接相关的小后果。",
                "不允许的展开：跳到股票/算法/考试/旧聊天/旧人物长故事/未授权地点，或从当前 prompt 扩成完全无关的历史回忆。",
                "如果原 prompt 有搞笑、难蚌、离谱、吐槽感，重写后必须保留笑点锐度；不要升华成大道理，不要写成总结报告。",
                "对缺失主题词要自然补回，不能只写泛泛的自由、努力、生活感受。",
            ]
        )
        return [
            {
                "role": "system",
                "content": "你是一个私人日记重写助手。只输出修订后的单篇日记正文，不做安全审核，不解释。",
            },
            {"role": "user", "content": user_content},
        ]

    def build_attempt_record(
        self,
        *,
        attempt: int,
        final_text: str,
        raw_model_output: str,
        rendered_prompt: str,
        generation_request: dict,
        postprocess: dict,
        guard_result: GuardResult,
    ) -> dict:
        return {
            "attempt": attempt,
            "final_text": final_text,
            "final_text_preview": final_text[:240],
            "raw_model_output": raw_model_output,
            "rendered_prompt": rendered_prompt,
            "generation_request": generation_request,
            "postprocess": postprocess,
            "guard": guard_result.to_dict(),
        }

    def select_attempt(self, attempts: list[dict], guard_config: dict) -> dict:
        for attempt in attempts:
            if attempt["guard"]["decision"] == "pass":
                return attempt
        for attempt in attempts:
            if attempt["guard"]["decision"] == "pass_with_warnings":
                return attempt
        if guard_config.get("fail_strategy") == "best_attempt":
            return max(attempts, key=lambda item: item["guard"].get("final_score", 0))
        return attempts[-1]

    def build_guard_summary(self, attempts: list[dict], selected_attempt: dict, guard_config: dict) -> dict:
        selected_guard = selected_attempt["guard"]
        summary = {
            "enabled": selected_guard.get("enabled", False),
            "decision": selected_guard.get("decision", "pass"),
            "selected_attempt": selected_attempt["attempt"],
            "retry_count": max(0, len(attempts) - 1),
            "config": guard_config,
            "attempts": [
                {
                    "attempt": attempt["attempt"],
                    "guard": attempt["guard"],
                    "postprocess": attempt["postprocess"],
                    "final_text_preview": attempt["final_text_preview"],
                }
                for attempt in attempts
            ],
        }
        for key in [
            "final_score",
            "topic_score",
            "drift_score",
            "format_score",
            "language_score",
            "quality_score",
            "warnings",
            "quality_warnings",
            "reasons",
        ]:
            summary[key] = selected_guard.get(key)
        return summary

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
        runtime.setdefault("guard", {})
        return runtime, context


def generate_diary(raw_prompt: str, runtime_config: dict | None = None, tokenizer=None, model=None) -> DiaryResult:
    if tokenizer is None or model is None:
        raise ValueError("generate_diary 第一版需要显式传入 tokenizer 和 model。")
    runtime = DiaryRuntime(runtime_config or {}, tokenizer, model)
    return runtime.generate(raw_prompt)


def _format_inline_list(items: list[str]) -> str:
    return "、".join(str(item) for item in items if str(item)) or "无"


def _format_bullets(items: list[str]) -> str:
    if not items:
        return "- 无"
    return "\n".join(f"- {item}" for item in items)


def _format_forbidden_hits(hits: list[dict]) -> str:
    if not hits:
        return "无"
    chunks = []
    for hit in hits[:12]:
        chunks.append(f"{hit.get('term')}({hit.get('severity')}x{hit.get('count')})")
    return "、".join(chunks)


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
