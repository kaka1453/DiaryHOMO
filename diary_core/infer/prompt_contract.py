from __future__ import annotations

from dataclasses import asdict, dataclass
import re


SHORT_PROMPT_MAX_CHARS = 12

DEFAULT_FORBIDDEN_TOPICS = [
    "股票交易",
    "期权",
    "算法题",
    "技术文档",
    "考试计划",
    "旧聊天记录",
    "Markdown 表格",
    "代码块",
    "外部链接",
]

DEFAULT_FORBIDDEN_FACT_TYPES = [
    "具体人名",
    "具体地点",
    "学校或公司名称",
    "精确日期",
    "消费金额",
    "股票代码或交易价格",
    "未出现的旧事件",
]

DEFAULT_STYLE_HINTS = [
    "第一人称",
    "口语化",
    "真实私人日记",
    "轻微自嘲",
    "碎碎念",
]


@dataclass
class DiaryContract:
    raw_prompt: str
    main_topic: str
    prompt_type: str
    topic_lock: str
    allowed_facts: list[str]
    forbidden_topics: list[str]
    forbidden_fact_types: list[str]
    style_hints: list[str]
    uncertainty_policy: str

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_prompt(raw_prompt: str) -> dict:
    prompt = raw_prompt.strip()
    prompt_len = len(prompt)
    if prompt_len <= SHORT_PROMPT_MAX_CHARS:
        prompt_type = "short_prompt"
    elif re.search(r"写一篇|记录|模拟|用.+风格|请|生成", prompt):
        prompt_type = "instruction"
    elif prompt.endswith(("，", "。", "：", ":")) or prompt_len < 40:
        prompt_type = "opening"
    else:
        prompt_type = "topic_or_opening"

    return {
        "raw_prompt": raw_prompt,
        "normalized_prompt": prompt,
        "prompt_length": prompt_len,
        "prompt_type": prompt_type,
        "is_short_prompt": prompt_type == "short_prompt",
        "main_topic": _infer_main_topic(prompt),
    }


def build_contract(raw_prompt: str, intent: dict | None = None, config: dict | None = None) -> DiaryContract:
    intent = intent or analyze_prompt(raw_prompt)
    config = config or {}
    prompt = intent["normalized_prompt"]
    main_topic = intent["main_topic"]
    allowed_facts = _extract_allowed_facts(prompt)
    forbidden_topics = _filter_forbidden_items(
        config.get("forbidden_topics") or DEFAULT_FORBIDDEN_TOPICS,
        prompt,
    )
    forbidden_fact_types = list(config.get("forbidden_fact_types") or DEFAULT_FORBIDDEN_FACT_TYPES)

    if intent["is_short_prompt"]:
        topic_lock = (
            f"本文只能围绕“{main_topic}”产生感受、语气和模糊日常联想；"
            "不要编造具体人名、地点、学校、股票、算法、考试或旧聊天记录。"
        )
        uncertainty_policy = "输入很短时，宁可写模糊感受和当下心理，不补具体事实。"
    else:
        topic_lock = f"本文必须围绕“{main_topic}”展开，每一段都要能回到这个主题。"
        uncertainty_policy = "缺少细节时，可以补充主观感受，但不要补充未经允许的具体事实。"

    return DiaryContract(
        raw_prompt=raw_prompt,
        main_topic=main_topic,
        prompt_type=intent["prompt_type"],
        topic_lock=topic_lock,
        allowed_facts=allowed_facts,
        forbidden_topics=forbidden_topics,
        forbidden_fact_types=forbidden_fact_types,
        style_hints=list(config.get("style_hints") or DEFAULT_STYLE_HINTS),
        uncertainty_policy=uncertainty_policy,
    )


def _infer_main_topic(prompt: str) -> str:
    if not prompt:
        return "今天的私人日记"
    cleaned = re.sub(r"^请?写一篇", "", prompt)
    cleaned = re.sub(r"日记[，,。.]?", "日记", cleaned)
    return cleaned[:80]


def _extract_allowed_facts(prompt: str) -> list[str]:
    parts = re.split(r"[\s，。！？、,.!?；;：:\n]+", prompt)
    facts = []
    for part in parts:
        item = part.strip()
        if not item or item in {"写一篇", "日记", "内容", "要求", "今天"}:
            continue
        if 1 <= len(item) <= 24 and item not in facts:
            facts.append(item)
    return facts[:12]


def _filter_forbidden_items(items: list[str], prompt: str) -> list[str]:
    filtered = []
    for item in items:
        compact = item.replace(" ", "")
        if compact and compact in prompt.replace(" ", ""):
            continue
        filtered.append(item)
    return filtered
