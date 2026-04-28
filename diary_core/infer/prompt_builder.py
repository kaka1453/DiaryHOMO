from __future__ import annotations

from diary_core.infer.prompt_contract import DiaryContract


DEFAULT_SYSTEM_PROMPT = (
    "你是一个私人日记写作助手。你的任务是根据用户给出的写作契约生成一篇私人日记。\n"
    "必须围绕本次主题写作，不得加入写作契约之外的具体事实。\n"
    "如果用户输入很短，只能围绕输入本身写感受和模糊日常，不要编造具体人名、地点、学校、股票、算法、考试、旧聊天记录。\n"
    "只输出日记正文，不要解释，不要标题，不要 Markdown 表格、代码块或链接。"
)


def build_messages(contract: DiaryContract, attachments: dict | None = None, config: dict | None = None) -> list[dict]:
    config = config or {}
    attachments = attachments or {}
    system_prompt = config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    user_content = "\n".join(
        [
            "[RAW_PROMPT]",
            contract.raw_prompt.strip(),
            "",
            "[CURRENT_TOPIC_LOCK]",
            contract.topic_lock,
            "",
            "[ALLOWED_FACTS]",
            _format_list(contract.allowed_facts),
            "",
            "[FORBIDDEN_TOPICS]",
            _format_list(contract.forbidden_topics),
            "",
            "[FORBIDDEN_FACT_TYPES]",
            _format_list(contract.forbidden_fact_types),
            "",
            "[ATTACHMENTS]",
            _format_attachments(attachments),
            "",
            "[STYLE]",
            "第一人称，口语化，像真实私人日记，可以有轻微自嘲和碎碎念。",
            "",
            "[OUTPUT]",
            "只输出日记正文。",
        ]
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def render_prompt(tokenizer, messages: list[dict], use_chat_template: bool = True) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return _fallback_render_messages(messages)


def _format_list(items: list[str]) -> str:
    if not items:
        return "- 无"
    return "\n".join(f"- {item}" for item in items)


def _format_attachments(attachments: dict) -> str:
    if not attachments or not attachments.get("items"):
        return "- 无附件。本轮不要引入额外事实。"
    lines = []
    for item in attachments["items"]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def _fallback_render_messages(messages: list[dict]) -> str:
    chunks = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        chunks.append(f"[{role}]\n{content}")
    chunks.append("[ASSISTANT]\n")
    return "\n\n".join(chunks)
