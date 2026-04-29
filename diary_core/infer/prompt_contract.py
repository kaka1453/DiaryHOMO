from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import re


SHORT_PROMPT_MAX_CHARS = 12

STYLE_KEYWORDS = [
    "搞笑",
    "幽默",
    "轻松",
    "吐槽",
    "八卦",
    "自嘲",
    "甜味",
    "无奈",
    "可爱",
    "羞涩",
    "真实",
    "反转",
    "意外结尾",
    "调侃",
    "打工人式",
    "夸张",
    "戏剧化",
    "反讽",
    "emo",
    "哲理",
    "严肃",
    "摆烂",
    "倒霉",
    "段子",
    "一本正经",
    "成熟",
    "躺平",
    "轻松口吻",
    "随意",
    "聊天一样",
    "自黑",
]

DRIFT_TOPIC_GROUPS = {
    "finance": ["股票", "期权", "TradingLog", "上证", "投资", "杠杆", "交易", "ETF", "EC期货", "恒生科技"],
    "algorithm": ["算法", "布尔矩阵", "组合数", "代码", "HTTP", "getByURL", "技术文档", "SQL", "SELECT"],
    "study_exam": [
        "考试",
        "CET6",
        "高数",
        "线性代数",
        "作业",
        "背单词",
        "复习",
        "备考",
        "考研",
        "课程",
        "申请",
        "NUS",
        "雅思",
        "高考",
        "期末",
    ],
    "lab_project": ["实验室", "论文", "项目", "报告", "开题", "工程", "物理化学模型"],
    "old_chat": ["旧聊天记录", "微信聊天", "截图", "消息记录", "Logseq", "Notes", "对话总结"],
    "device_shopping": ["AirPods", "苹果店", "Apple Store", "手机号泄露", "iCloud", "AppleWatch", "闲鱼", "屏幕"],
    "game_media": ["Genshin", "原神", "B站", "bilibili", "YouTube", "GitHub", "番剧", "外部链接"],
    "old_person_story": ["旧人物长故事", "旧恋爱策略", "旧同学纠葛", "旧群聊复盘", "未授权人物关系扩写"],
    "language_noise": ["日语假名", "繁体字", "英文长串", "乱码字符", "混杂外语叙述"],
    "format_drift": ["Markdown 表格", "代码块", "外部链接", "多篇分隔符", "## 标题", "### 标题", "HTML 标签"],
}

DEFAULT_FORMAT_FORBIDDEN = [
    "Markdown 标题",
    "Markdown 表格",
    "代码块",
    "外部链接",
    "多篇分隔符",
    "HTML 注释",
    "任务清单",
    "引用块",
    "【回忆录】标签",
    "情况1/情况2分点",
    "编号分点",
]

DEFAULT_FORBIDDEN_FACT_TYPES = [
    "未经提示的具体人名",
    "未经提示的具体地点",
    "未经提示的学校或公司名称",
    "未经提示的精确日期",
    "未经提示的消费金额",
    "未经提示的股票代码或交易价格",
    "未出现的旧事件",
]

DEFAULT_STYLE_HINTS = [
    "第一人称",
    "口语化",
    "真实私人日记",
    "轻微自嘲",
    "碎碎念",
]

SHORT_DAILY_PROMPTS = {
    "啥都没干",
    "又迟到",
    "今天真累",
    "家里人",
    "我喜欢你",
    "我爱你",
}

KEYWORD_ALIASES = {
    "cat": "cat/猫",
    "nus": "NUS",
    "kaka": "kaka",
    "sq": "SQ",
    "山东": "山东",
}

DOMAIN_TOPIC_KEYWORDS = [
    "恋爱",
    "色情",
    "反抗",
    "共和党",
    "暴政",
    "黑纸运动",
    "起义",
    "重生",
    "五百强",
    "企业家儿子",
    "表白",
    "周瑜",
    "cat",
    "猫",
    "NUS",
    "kaka",
    "SQ",
    "山东",
    "考研",
    "复习",
    "备考",
    "考试",
    "倒霉",
    "拖延",
    "熬夜",
    "崩溃",
    "计划",
    "摆烂",
    "觉醒",
    "成熟",
    "躺平",
    "emo",
    "离谱",
    "尴尬社交",
    "琐事",
    "趣事",
    "日常",
    "朋友",
    "吃饭",
]

STOPWORDS = {
    "写一篇",
    "日记",
    "今天",
    "内容",
    "相关",
    "关于",
    "记录",
    "一下",
    "一个",
    "一种",
    "语气",
    "方式",
    "风格",
    "心态",
    "模拟",
    "描述",
    "表达",
    "不要",
    "不能",
    "可以",
    "比如",
    "像是",
    "以及",
    "或者",
    "但是",
    "有点",
    "带点",
    "别太正经",
    "正文",
}

CONTRACT_PATTERNS = [
    (
        "style_instruction",
        re.compile(r"用[“\"](?P<style>.+?)[”\"]风格写(?P<topic>.+?)(?:，|。|$)"),
    ),
    (
        "style_instruction",
        re.compile(r"用(?P<style>.+?)的语气描述(?P<topic>.+?)(?:，|。|$)"),
    ),
    (
        "style_instruction",
        re.compile(r"以[“\"](?P<style>.+?)[”\"]的心态写(?P<topic>.+?)(?:，|。|$)"),
    ),
    (
        "style_instruction",
        re.compile(r"请用(?P<style>.+?)的?方式记录(?P<topic>.+?)(?:，|。|$)"),
    ),
    (
        "simulation_instruction",
        re.compile(r"模拟一个[“\"](?P<topic>.+?)[”\"](?P<topic_tail>[^，。,.]{0,16})(?:，|。|$)"),
    ),
    (
        "simulation_instruction",
        re.compile(r"模拟一个[“\"]?(?P<topic>.+?)[”\"]?(?:，|用|$)"),
    ),
    (
        "instruction",
        re.compile(r"写一篇关于[“\"]?(?P<topic>.+?)[”\"]?的(?P<style>[^，。,.]{0,12})日记"),
    ),
    (
        "instruction",
        re.compile(r"写一篇[“\"](?P<topic>.+?)[”\"](?P<topic_tail>[^，。,.]{0,18})日记"),
    ),
    (
        "instruction",
        re.compile(r"写一篇[“\"](?P<topic>.+?)[”\"]的日记"),
    ),
    (
        "instruction",
        re.compile(r"写一篇(?P<topic>.+?)的日记"),
    ),
    (
        "instruction",
        re.compile(r"写一篇(?P<topic>.+?)日记"),
    ),
    (
        "instruction",
        re.compile(r"记录一下(?P<topic>.+?)(?:，|。|$)"),
    ),
    (
        "instruction",
        re.compile(r"记录(?P<topic>.+?)(?:，|。|$)"),
    ),
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
    topic_terms: list[str]
    forbidden_drift_topics: list[str]
    format_forbidden: list[str]
    length_hint: str
    risk_tags: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_prompt(raw_prompt: str) -> dict:
    prompt = raw_prompt.strip()
    prompt_len = len(prompt)
    match_info = _match_contract_pattern(prompt)
    prompt_type = _classify_prompt(prompt, match_info)
    style_hints = _extract_style_hints(prompt, match_info)
    main_topic = _infer_main_topic(prompt, prompt_type, match_info, style_hints)
    topic_terms = _extract_topic_terms(prompt, main_topic)
    length_hint = _infer_length_hint(prompt, prompt_type)
    risk_tags = _infer_risk_tags(prompt, prompt_type, topic_terms)

    return {
        "raw_prompt": raw_prompt,
        "normalized_prompt": prompt,
        "prompt_length": prompt_len,
        "prompt_type": prompt_type,
        "is_short_prompt": prompt_type in {"keyword", "short_daily"} or prompt_len <= SHORT_PROMPT_MAX_CHARS,
        "main_topic": main_topic,
        "topic_terms": topic_terms,
        "style_hints": style_hints,
        "length_hint": length_hint,
        "risk_tags": risk_tags,
        "pattern": match_info.get("pattern") if match_info else None,
    }


def build_contract(raw_prompt: str, intent: dict | None = None, config: dict | None = None) -> DiaryContract:
    intent = intent or analyze_prompt(raw_prompt)
    config = config or {}
    prompt = intent["normalized_prompt"]
    main_topic = intent["main_topic"]
    topic_terms = list(intent.get("topic_terms") or _extract_topic_terms(prompt, main_topic))
    allowed_facts = _extract_allowed_facts(prompt, topic_terms)
    forbidden_drift_topics = _build_forbidden_drift_topics(
        prompt,
        topic_terms,
        config.get("drift_topic_groups") or DRIFT_TOPIC_GROUPS,
    )
    extra_forbidden = config.get("forbidden_drift_topics") or config.get("forbidden_topics") or []
    forbidden_drift_topics = _unique_preserve_order([*forbidden_drift_topics, *extra_forbidden])
    format_forbidden = list(config.get("format_forbidden") or DEFAULT_FORMAT_FORBIDDEN)
    forbidden_topics = _unique_preserve_order([*forbidden_drift_topics, *format_forbidden])
    forbidden_fact_types = list(config.get("forbidden_fact_types") or DEFAULT_FORBIDDEN_FACT_TYPES)
    style_hints = _unique_preserve_order(
        [
            *(config.get("style_hints") or DEFAULT_STYLE_HINTS),
            *(intent.get("style_hints") or []),
        ]
    )

    if intent["is_short_prompt"]:
        topic_lock = (
            f"本文只能围绕“{main_topic}”产生感受、语气和模糊日常联想；"
            f"核心词是：{_join_terms(topic_terms)}。不要把短词扩写成旧人物长故事、学习计划、股票交易或技术文档。"
        )
        uncertainty_policy = "输入很短时，宁可写模糊感受、当下心理和轻微日常氛围，不补具体事实。"
    else:
        topic_lock = (
            f"本文必须围绕“{main_topic}”展开；每一段都要能回到这些主题词：{_join_terms(topic_terms)}。"
            "不要迁移到未授权旧主题。"
        )
        uncertainty_policy = "缺少细节时，可以补充主观感受和模糊日常，但不要迁移旧日记事件。"

    return DiaryContract(
        raw_prompt=raw_prompt,
        main_topic=main_topic,
        prompt_type=intent["prompt_type"],
        topic_lock=topic_lock,
        allowed_facts=allowed_facts,
        forbidden_topics=forbidden_topics,
        forbidden_fact_types=forbidden_fact_types,
        style_hints=style_hints,
        uncertainty_policy=uncertainty_policy,
        topic_terms=topic_terms,
        forbidden_drift_topics=forbidden_drift_topics,
        format_forbidden=format_forbidden,
        length_hint=str(intent.get("length_hint") or "medium"),
        risk_tags=list(intent.get("risk_tags") or []),
    )


def extract_prompts_from_output_md(path: str) -> list[str]:
    text = _read_text(path)
    prompts = []
    for line in text.splitlines():
        match = re.match(r"\s*引言[:：]\s*(.+?)\s*$", line)
        if match:
            prompts.append(match.group(1).strip())
    return prompts


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _match_contract_pattern(prompt: str) -> dict:
    for prompt_type, pattern in CONTRACT_PATTERNS:
        match = pattern.search(prompt)
        if not match:
            continue
        data = {key: value for key, value in match.groupdict().items() if value}
        if data.get("topic_tail"):
            data["topic"] = f"{data.get('topic', '')}{data.pop('topic_tail')}"
            data["topic"] = re.sub(r"(?:的)?日记$", "", data["topic"])
        data["prompt_type"] = prompt_type
        data["pattern"] = pattern.pattern
        return data
    return {}


def _classify_prompt(prompt: str, match_info: dict) -> str:
    if not prompt:
        return "short_daily"
    if prompt.endswith(("?", "？")) or re.match(r"^(你|我|今天).*(吗|么|嘛)[?？]?$", prompt):
        return "question"
    if match_info:
        return match_info["prompt_type"]
    compact = _compact(prompt)
    if compact in SHORT_DAILY_PROMPTS:
        return "short_daily"
    if _looks_like_opening(prompt):
        return "opening"
    if _looks_like_keyword(prompt):
        return "keyword"
    if re.search(r"写一篇|记录|模拟|请用|生成", prompt):
        return "instruction"
    if len(prompt) <= SHORT_PROMPT_MAX_CHARS:
        return "short_daily"
    return "opening"


def _infer_main_topic(prompt: str, prompt_type: str, match_info: dict, style_hints: list[str]) -> str:
    if not prompt:
        return "今天的私人日记"
    if prompt_type == "keyword":
        alias = KEYWORD_ALIASES.get(_compact(prompt).lower(), prompt)
        return f"{alias} 相关的当下小日记"
    if prompt_type == "short_daily":
        return f"{prompt.strip('。.!！?？')}的当下感受"
    if prompt_type == "question":
        return f"围绕“{prompt}”回应近况的私人日记"

    topic = match_info.get("topic") if match_info else ""
    if not topic:
        topic = prompt
    topic = _clean_topic(topic)
    if not topic and match_info.get("topic") in {"今天", "今天的"}:
        topic = "今天的日常"
    if not topic:
        topic = prompt[:80]

    style_topic_terms = [item for item in style_hints if item and item not in topic]
    if style_topic_terms:
        topic = f"{topic}，风格：{'、'.join(style_topic_terms[:3])}"
    return topic[:120]


def _extract_allowed_facts(prompt: str, topic_terms: list[str] | None = None) -> list[str]:
    parts = re.split(r"[\s，。！？、,.!?；;：:\n]+", prompt)
    facts = list(topic_terms or [])
    for part in parts:
        item = part.strip()
        if not item or item in STOPWORDS:
            continue
        if 1 <= len(item) <= 24 and item not in facts:
            facts.append(item)
    return facts[:12]


def _extract_style_hints(prompt: str, match_info: dict | None = None) -> list[str]:
    hints = []
    style = (match_info or {}).get("style")
    if style:
        hints.extend(_split_terms(style))
    for keyword in STYLE_KEYWORDS:
        if keyword.lower() in prompt.lower():
            hints.append(keyword)
    return _unique_preserve_order(
        [_clean_term(item) for item in hints if _is_valid_style_hint(_clean_term(item))]
    )


def _extract_topic_terms(prompt: str, main_topic: str) -> list[str]:
    compact_prompt = _compact(prompt).lower()
    if compact_prompt in KEYWORD_ALIASES:
        return _unique_preserve_order(_split_terms(KEYWORD_ALIASES[compact_prompt]))[:12]

    candidates = []
    candidates.extend(_split_terms(main_topic))
    candidates.extend(_extract_quoted_terms(prompt))
    source = f"{prompt} {main_topic}"
    for keyword in DOMAIN_TOPIC_KEYWORDS:
        if _contains_term(source, keyword):
            candidates.append(keyword)
    for keyword in STYLE_KEYWORDS:
        if keyword.lower() in prompt.lower():
            candidates.append(keyword)
    for group_terms in DRIFT_TOPIC_GROUPS.values():
        for term in group_terms:
            if _contains_term(prompt, term):
                candidates.append(term)
    normalized = []
    for item in candidates:
        term = _clean_term(item)
        if _is_valid_topic_term(term):
            normalized.append(term)
    if not normalized and prompt:
        normalized.append(prompt[:24])
    return _unique_preserve_order(normalized)[:12]


def _extract_quoted_terms(prompt: str) -> list[str]:
    terms = []
    for match in re.finditer(r"[“\"「『](.+?)[”\"」』]", prompt):
        terms.extend(_split_terms(match.group(1)))
    return terms


def _build_forbidden_drift_topics(prompt: str, topic_terms: list[str], groups: dict[str, list[str]]) -> list[str]:
    forbidden = []
    prompt_terms = f"{prompt} {' '.join(topic_terms)}"
    for group_name, terms in groups.items():
        if group_name != "format_drift" and _group_is_authorized(prompt_terms, terms):
            continue
        forbidden.extend(terms)
    return _unique_preserve_order(forbidden)


def _group_is_authorized(prompt_terms: str, terms: list[str]) -> bool:
    return any(_contains_term(prompt_terms, term) for term in terms if not term.startswith(("旧", "未授权")))


def _infer_length_hint(prompt: str, prompt_type: str) -> str:
    if prompt_type in {"keyword", "short_daily", "question"}:
        return "short"
    if prompt_type == "opening":
        return "medium"
    if len(prompt) > 60:
        return "medium-long"
    return "medium"


def _infer_risk_tags(prompt: str, prompt_type: str, topic_terms: list[str]) -> list[str]:
    tags = []
    if prompt_type in {"keyword", "short_daily"}:
        tags.append("short_input_expansion_risk")
    if re.search(r"[A-Za-z]{2,}", prompt):
        tags.append("multilingual_trigger")
    if any(_contains_term(prompt, term) for term in ["色情", "恋爱"]):
        tags.append("romance_or_adult_style")
    if _contains_term(prompt, "色情"):
        tags.append("adult_content_boundary")
    if any(_contains_term(prompt, term) for term in ["共和党", "黑纸运动", "起义", "暴政"]):
        tags.append("political_fiction")
    if any(_contains_term(prompt, term) for term in ["重生", "模拟"]):
        tags.append("fictional_setup")
    if any(term.lower() in {"cat", "kaka", "sq", "nus"} for term in topic_terms):
        tags.append("named_keyword")
    return _unique_preserve_order(tags)


def _looks_like_keyword(prompt: str) -> bool:
    if re.search(r"[，。！？,.!?；;：:\s]", prompt):
        return False
    compact = _compact(prompt)
    if compact.lower() in KEYWORD_ALIASES:
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,10}", compact):
        return True
    return len(compact) <= 5


def _looks_like_opening(prompt: str) -> bool:
    if len(prompt) > 40:
        return False
    if prompt.endswith(("，", "：", ":")):
        return True
    if prompt.startswith(("今天", "我向", "这不", "黑人")) and not prompt.endswith(("。", "！", "？", ".", "!", "?")):
        return True
    return False


def _clean_topic(topic: str) -> str:
    cleaned = topic.strip(" “”，,。.!！?？")
    cleaned = re.sub(r"^(今天的|今天|一个|关于)", "", cleaned)
    cleaned = re.sub(r"(?:的)?日记$", "", cleaned)
    cleaned = re.sub(r"的$", "", cleaned)
    cleaned = cleaned.replace("”", "").replace("“", "").replace("\"", "")
    cleaned = cleaned.strip(" “”，,。.!！?？")
    return cleaned


def _split_terms(text: str) -> list[str]:
    text = _clean_topic(text)
    parts = re.split(r"[、/+&，,。.!！?？；;：:\s（）()“”\"<>《》]+", text)
    return [part for part in parts if part]


def _clean_term(term: str) -> str:
    term = term.strip(" -_[]【】()（）“”\"'‘’。，,；;：:!?！？")
    term = re.sub(r"^(今天的|今天|关于|内容|语气|风格|方式|心态|表达)", "", term)
    term = re.sub(r"(日记|片段|记录|生活)$", "", term)
    return term.strip(" -_[]【】()（）“”\"'‘’。，,；;：:!?！？")


def _is_valid_topic_term(term: str) -> bool:
    if not term or term in STOPWORDS:
        return False
    if any(marker in term for marker in ["相关的当下", "当下感受", "回应近况"]):
        return False
    if len(term) > 18:
        return False
    if re.fullmatch(r"\d+", term):
        return False
    return True


def _is_valid_style_hint(term: str) -> bool:
    if not term or term in STOPWORDS:
        return False
    if term in STYLE_KEYWORDS:
        return True
    if any(keyword in term for keyword in STYLE_KEYWORDS):
        return True
    if len(term) <= 8 and not re.search(r"记录|描述|事件|日记|生活|片段", term):
        return True
    return False


def _contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    haystack = _compact(text).lower()
    needle = _compact(term).lower()
    return needle in haystack


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _join_terms(terms: list[str]) -> str:
    return "、".join(terms) if terms else "本次输入"


def main() -> None:
    parser = argparse.ArgumentParser(description="DiaryContract rule analyzer")
    parser.add_argument("--prompt", default=None, help="Analyze one prompt.")
    parser.add_argument("--from-output-md", default=None, help="Extract 引言 prompts from output markdown and analyze all.")
    args = parser.parse_args()

    prompts = []
    if args.from_output_md:
        prompts.extend(extract_prompts_from_output_md(args.from_output_md))
    if args.prompt:
        prompts.append(args.prompt)
    if not prompts:
        prompts.append("今天和朋友吃饭，很开心。")

    payload = []
    for prompt in prompts:
        contract = build_contract(prompt)
        payload.append(
            {
                "prompt": prompt,
                "prompt_type": contract.prompt_type,
                "main_topic": contract.main_topic,
                "topic_terms": contract.topic_terms,
                "style_hints": contract.style_hints,
                "forbidden_drift_topics": contract.forbidden_drift_topics,
                "format_forbidden": contract.format_forbidden,
                "length_hint": contract.length_hint,
                "risk_tags": contract.risk_tags,
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
