from __future__ import annotations

from diary_core.infer.prompt_contract import DiaryContract


DEFAULT_SYSTEM_PROMPT = (
    "你是一个私人日记写作助手。你的任务是根据用户给出的写作契约生成一篇私人日记。\n"
    "必须围绕 MAIN_TOPIC 和 TOPIC_TERMS 写作，不得迁移到写作契约列出的旧主题禁区；FORBIDDEN_DRIFT_TOPICS 是硬禁区，不是参考建议。\n"
    "如果用户输入很短，只能围绕输入本身写当下感受和模糊日常，不要回顾过去具体经历，不要扩写成旧人物长故事、学习计划、股票交易或技术文档。\n"
    "只能使用简体中文；不要日语假名、繁体字、无意义英文长串、乱码、Markdown 表格、代码块或链接。\n"
    "只输出单篇日记正文，不要解释，不要标题。"
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
            "[MAIN_TOPIC]",
            contract.main_topic,
            "",
            "[TOPIC_TERMS]",
            _format_list(getattr(contract, "topic_terms", [])),
            "",
            "[CURRENT_TOPIC_LOCK]",
            contract.topic_lock,
            "",
            "[ALLOWED_FACTS]",
            _format_list(contract.allowed_facts),
            "",
            "[STYLE_HINTS]",
            _format_list(contract.style_hints),
            "",
            "[FORBIDDEN_DRIFT_TOPICS]",
            _format_list(getattr(contract, "forbidden_drift_topics", contract.forbidden_topics)),
            "",
            "[FORMAT_FORBIDDEN]",
            _format_list(getattr(contract, "format_forbidden", [])),
            "",
            "[DRIFT_AVOIDANCE_RULE]",
            _drift_avoidance_rule(contract),
            "",
            "[SAFETY_BOUNDARY]",
            _safety_boundary(contract),
            "",
            "[FACT_BOUNDARY_HINT]",
            _fact_boundary_hint(contract),
            "",
            "[ATTACHMENTS]",
            _format_attachments(attachments),
            "",
            "[OUTPUT]",
            "只输出单篇连续日记正文。不要输出标题、解释、列表、分点、表格、代码块、标签或链接。",
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


def _fact_boundary_hint(contract: DiaryContract) -> str:
    fact_types = getattr(contract, "forbidden_fact_types", [])
    weak_hint = "缺少细节时，优先写感受和模糊日常，不要为了丰富内容迁移旧日记事件。"
    if not fact_types:
        return weak_hint
    return f"{weak_hint} 弱提醒：不要主动添加{_join_inline(fact_types[:4])}等未授权具体事实。"


def _drift_avoidance_rule(contract: DiaryContract) -> str:
    lines = [
        "FORBIDDEN_DRIFT_TOPICS 中的内容不得出现在正文里；如果想写到禁区词，必须立刻改写回 MAIN_TOPIC。",
        "每一段至少要能对应一个 TOPIC_TERMS，不能先写主题、后面转到学习、股票、技术、旧聊天、设备购物、游戏媒体或链接。",
        "不要用【回忆录】、情况1/情况2、编号分点、任务清单或资料摘录式结构；正文必须像一篇连续的私人日记。",
    ]
    if getattr(contract, "length_hint", "") == "short":
        lines.append("本条输入很短，只写当下感受和模糊日常；不要补过去经历、家庭旧事、小区邻居、学校课程、考试复习等具体故事。")
    return "\n".join(lines)


def _safety_boundary(contract: DiaryContract) -> str:
    risk_tags = set(getattr(contract, "risk_tags", []))
    if "adult_content_boundary" in risk_tags:
        return (
            "如果涉及恋爱或成人暧昧，安全边界优先级高于 RAW_PROMPT：只写成年人之间含蓄的暧昧误会、心理吐槽和尴尬笑点。"
            "不要写未成年人、学生/小朋友、露骨性行为、身体接触、睡着/无意识、澡堂、偷拍、胁迫、未经同意或隐私侵犯。"
        )
    return "保持私人日记口吻，不要为了戏剧化加入危险、违法或与主题无关的刺激内容。"


def _join_inline(items: list[str]) -> str:
    return "、".join(items)


def _fallback_render_messages(messages: list[dict]) -> str:
    chunks = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        chunks.append(f"[{role}]\n{content}")
    chunks.append("[ASSISTANT]\n")
    return "\n\n".join(chunks)
