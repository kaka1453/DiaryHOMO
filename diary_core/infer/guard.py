from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Literal

from diary_core.infer.prompt_contract import DiaryContract, build_contract


GuardDecision = Literal["pass", "revise", "fail"]


DEFAULT_GUARD_CONFIG = {
    "enabled": True,
    "max_retry": 1,
    "min_final_score": 75,
    "min_topic_score": 55,
    "min_drift_score": 85,
    "min_format_score": 85,
    "min_language_score": 75,
    "retry_on_revise": True,
    "fail_strategy": "best_attempt",
}

HIGH_SEVERITY_TERMS = {
    "股票",
    "期权",
    "算法",
    "代码",
    "考试",
    "NUS",
    "实验室",
    "旧聊天记录",
    "链接",
    "Markdown 表格",
    "日语假名",
    "英文长串",
    "外部链接",
    "http",
    "https",
}

MEDIUM_SEVERITY_TERMS = {
    "作业",
    "复习",
    "课程",
    "B站",
    "GitHub",
    "AirPods",
    "Genshin",
    "高数",
    "线性代数",
    "雅思",
    "CET6",
    "NUS",
    "yyj",
    "b计算机",
    "冰激凌",
    "钓鱼用品店",
    "浩海钓具",
    "反向伞",
}

STYLE_OR_GENERIC_TERMS = {
    "搞笑",
    "幽默",
    "吐槽",
    "反差",
    "场景化",
    "真实",
    "可爱",
    "无奈",
    "轻松",
    "严肃",
    "哲理",
    "日常",
    "今天",
    "正文",
}

TERM_ALIASES = {
    "宿舍": ["寝室"],
    "被迫围观": ["围观", "吸引出来", "出来看"],
    "和朋友吃饭": ["朋友们一起吃", "朋友们聚餐", "和朋友们一起吃"],
    "朋友": ["朋友们"],
    "考研": ["备考", "上岸"],
    "复习": ["刷题", "做题", "学习"],
    "调侃": ["吐槽", "难蚌"],
    "反抗": ["对抗"],
    "黑纸运动": ["黑纸"],
    "起义": ["抗议", "行动"],
}

DEFAULT_EXTRA_FORBIDDEN_TERMS = [
    "yyj",
    "b计算机",
    "冰激凌",
    "浩海钓具",
    "钓鱼用品店",
    "反向伞",
    "大明瑾",
    "圣地",
    "分校区",
    "旧人物档案",
    "过去的聊天",
    "聊天记录",
    "顾问",
]

TRADITIONAL_CHARS = set(
    "學習體驗關係個這樣說話還會沒為與對時後點裡讓過來麼"
    "氣實現發現問題應該準備複雜總結開始"
)


@dataclass
class GuardResult:
    enabled: bool
    decision: GuardDecision
    final_score: float
    topic_score: float
    drift_score: float
    format_score: float
    language_score: float
    topic_hits: list[str]
    missing_topic_terms: list[str]
    forbidden_hits: list[dict]
    format_hits: list[str]
    language_noise_hits: list[str]
    tail_drift: bool
    reasons: list[str]
    revision_instruction: str

    def to_dict(self) -> dict:
        return asdict(self)


def guard_diary(text: str, contract: DiaryContract, config: dict | None = None) -> GuardResult:
    guard_config = normalize_guard_config(config)
    if not guard_config["enabled"]:
        return GuardResult(
            enabled=False,
            decision="pass",
            final_score=100.0,
            topic_score=100.0,
            drift_score=100.0,
            format_score=100.0,
            language_score=100.0,
            topic_hits=[],
            missing_topic_terms=[],
            forbidden_hits=[],
            format_hits=[],
            language_noise_hits=[],
            tail_drift=False,
            reasons=["guard disabled"],
            revision_instruction="",
        )

    topic_score, topic_hits, missing_topic_terms, tail_drift = score_topic(text, contract)
    forbidden_terms = build_guard_forbidden_terms(contract, guard_config)
    forbidden_hits = find_forbidden_hits(text, forbidden_terms)
    drift_score = score_drift(forbidden_hits)
    format_hits, format_score = check_format(text)
    language_noise_hits, language_score = check_language_noise(text, contract)

    final_score = round(
        topic_score * 0.4
        + drift_score * 0.35
        + format_score * 0.15
        + language_score * 0.1,
        2,
    )
    reasons = build_reasons(
        topic_score=topic_score,
        drift_score=drift_score,
        format_score=format_score,
        language_score=language_score,
        missing_topic_terms=missing_topic_terms,
        forbidden_hits=forbidden_hits,
        format_hits=format_hits,
        language_noise_hits=language_noise_hits,
        tail_drift=tail_drift,
        config=guard_config,
    )
    decision = decide(
        final_score=final_score,
        topic_score=topic_score,
        drift_score=drift_score,
        format_score=format_score,
        language_score=language_score,
        config=guard_config,
    )
    revision_instruction = build_revision_instruction(
        contract=contract,
        reasons=reasons,
        forbidden_hits=forbidden_hits,
        missing_topic_terms=missing_topic_terms,
        format_hits=format_hits,
        language_noise_hits=language_noise_hits,
    )

    return GuardResult(
        enabled=True,
        decision=decision,
        final_score=final_score,
        topic_score=round(topic_score, 2),
        drift_score=round(drift_score, 2),
        format_score=round(format_score, 2),
        language_score=round(language_score, 2),
        topic_hits=topic_hits,
        missing_topic_terms=missing_topic_terms,
        forbidden_hits=forbidden_hits,
        format_hits=format_hits,
        language_noise_hits=language_noise_hits,
        tail_drift=tail_drift,
        reasons=reasons,
        revision_instruction=revision_instruction,
    )


def normalize_guard_config(config: dict | None = None) -> dict:
    normalized = dict(DEFAULT_GUARD_CONFIG)
    if isinstance(config, dict):
        normalized.update(config)
    normalized["enabled"] = _as_bool(normalized.get("enabled", True))
    normalized["max_retry"] = int(normalized.get("max_retry", 1))
    normalized["retry_on_revise"] = _as_bool(normalized.get("retry_on_revise", True))
    normalized["min_final_score"] = float(normalized.get("min_final_score", 75))
    normalized["min_topic_score"] = float(normalized.get("min_topic_score", 55))
    normalized["min_drift_score"] = float(normalized.get("min_drift_score", 85))
    normalized["min_format_score"] = float(normalized.get("min_format_score", 85))
    normalized["min_language_score"] = float(normalized.get("min_language_score", 75))
    normalized["fail_strategy"] = str(normalized.get("fail_strategy") or "best_attempt")
    normalized["extra_forbidden_terms"] = list(
        normalized.get("extra_forbidden_terms") or DEFAULT_EXTRA_FORBIDDEN_TERMS
    )
    return normalized


def build_guard_forbidden_terms(contract: DiaryContract, config: dict) -> list[str]:
    authorized_text = " ".join([contract.raw_prompt, contract.main_topic, *getattr(contract, "topic_terms", [])])
    terms = [*getattr(contract, "forbidden_drift_topics", []), *config.get("extra_forbidden_terms", [])]
    return [
        term
        for term in unique_preserve_order([str(item).strip() for item in terms])
        if term and not contains_term(authorized_text, term)
    ]


def score_topic(text: str, contract: DiaryContract) -> tuple[float, list[str], list[str], bool]:
    terms = important_topic_terms(contract)
    if not terms:
        return 70.0, [], [], False

    topic_hits = [term for term in terms if contains_term(text, term)]
    missing_topic_terms = [term for term in terms if term not in topic_hits]
    hit_rate = len(topic_hits) / max(1, len(terms))

    paragraphs = split_paragraphs(text)
    if paragraphs:
        paragraph_hits = sum(1 for paragraph in paragraphs if any(contains_term(paragraph, term) for term in terms))
        paragraph_rate = paragraph_hits / len(paragraphs)
    else:
        paragraph_rate = 0.0

    tail = tail_text(text)
    tail_topic_ok = any(contains_term(tail, term) for term in terms)
    tail_component = 1.0 if tail_topic_ok else 0.0

    if contract.prompt_type in {"keyword", "short_daily", "question"} or contract.length_hint == "short":
        score = 70 * min(1.0, hit_rate * 1.5) + 20 * paragraph_rate + 10 * tail_component
        if topic_hits:
            score = max(score, 70.0)
    else:
        score = 50 * hit_rate + 30 * paragraph_rate + 20 * tail_component

    if len(terms) <= 2 and topic_hits:
        score = max(score, 75.0)

    tail_drift = bool(len(text) >= 120 and not tail_topic_ok and topic_hits)
    return _clamp(score), topic_hits, missing_topic_terms, tail_drift


def important_topic_terms(contract: DiaryContract) -> list[str]:
    candidates = []
    for term in list(getattr(contract, "topic_terms", []) or []):
        cleaned = clean_term(term)
        if is_important_topic_term(cleaned):
            candidates.append(cleaned)

    if len(candidates) < 2:
        for term in split_topic_keywords(contract.main_topic):
            cleaned = clean_term(term)
            if is_important_topic_term(cleaned):
                candidates.append(cleaned)

    if not candidates:
        for term in split_topic_keywords(contract.raw_prompt):
            cleaned = clean_term(term)
            if is_important_topic_term(cleaned):
                candidates.append(cleaned)

    return unique_preserve_order(candidates)[:8]


def find_forbidden_hits(text: str, forbidden_drift_topics: list[str]) -> list[dict]:
    hits = []
    for term in unique_preserve_order(forbidden_drift_topics):
        term = str(term).strip()
        if not is_searchable_forbidden_term(term):
            continue
        positions = find_positions(text, term)
        if not positions:
            continue
        hits.append(
            {
                "term": term,
                "count": len(positions),
                "severity": severity_for_term(term),
                "positions": positions[:10],
            }
        )
    return hits


def score_drift(forbidden_hits: list[dict]) -> float:
    score = 100
    for hit in forbidden_hits:
        severity = hit.get("severity")
        count = int(hit.get("count") or 0)
        if severity == "high":
            score -= count * 18
        elif severity == "medium":
            score -= count * 10
        else:
            score -= count * 5
    return _clamp(score)


def check_format(text: str) -> tuple[list[str], float]:
    hits = []
    checks = [
        ("markdown_heading", r"(?m)^#{1,6}\s+|\n#{1,6}\s+"),
        ("markdown_table", r"(?m)\|.+\|.*\n\|?\s*:?-{3,}:?\s*\|"),
        ("code_block", r"```"),
        ("raw_link", r"https?://"),
        ("markdown_link", r"\[[^\]]+\]\([^)]+\)"),
        ("multi_doc_separator", r"\n---+|\n##\s*第"),
        ("numbered_list", r"(?m)^\s*1[.、]\s+.*\n\s*2[.、]\s+"),
        ("case_list", r"情况\s*1|情况\s*2"),
        ("html_comment", r"<!--"),
    ]
    penalties = {
        "markdown_heading": 20,
        "markdown_table": 30,
        "code_block": 30,
        "raw_link": 25,
        "markdown_link": 25,
        "multi_doc_separator": 25,
        "numbered_list": 15,
        "case_list": 15,
        "html_comment": 30,
    }
    score = 100
    for name, pattern in checks:
        if re.search(pattern, text):
            hits.append(name)
            score -= penalties[name]
    return hits, _clamp(score)


def check_language_noise(text: str, contract: DiaryContract) -> tuple[list[str], float]:
    hits = []
    score = 100
    total_len = max(1, len(text))
    ascii_letter_count = len(re.findall(r"[A-Za-z]", text))
    english_ratio = ascii_letter_count / total_len
    prompt_has_ascii = bool(re.search(r"[A-Za-z]", contract.raw_prompt))
    english_threshold = 0.2 if prompt_has_ascii else 0.12
    if english_ratio > english_threshold:
        hits.append(f"english_ratio={english_ratio:.2f}")
        score -= 25
    if re.search(r"[\u3040-\u30ff]", text):
        hits.append("japanese_kana")
        score -= 25
    traditional_count = sum(1 for char in text if char in TRADITIONAL_CHARS)
    if traditional_count / total_len > 0.06:
        hits.append(f"traditional_ratio={traditional_count / total_len:.2f}")
        score -= 15
    if re.search(r"[\ufffd\u0378-\u0379\uFFF0-\uFFFF]", text):
        hits.append("garbled_unicode")
        score -= 25
    if re.search(r"[A-Za-z]{16,}", text):
        hits.append("long_ascii_token")
        score -= 15
    return hits, _clamp(score)


def decide(
    final_score: float,
    topic_score: float,
    drift_score: float,
    format_score: float,
    language_score: float,
    config: dict,
) -> GuardDecision:
    if (
        final_score >= config["min_final_score"]
        and topic_score >= config["min_topic_score"]
        and drift_score >= config["min_drift_score"]
        and format_score >= config["min_format_score"]
        and language_score >= config["min_language_score"]
    ):
        return "pass"
    if final_score >= 50:
        return "revise"
    return "fail"


def build_reasons(
    *,
    topic_score: float,
    drift_score: float,
    format_score: float,
    language_score: float,
    missing_topic_terms: list[str],
    forbidden_hits: list[dict],
    format_hits: list[str],
    language_noise_hits: list[str],
    tail_drift: bool,
    config: dict,
) -> list[str]:
    reasons = []
    if topic_score < config["min_topic_score"]:
        reasons.append(f"主题命中不足，缺失主题词：{', '.join(missing_topic_terms[:8]) or '无'}")
    if tail_drift:
        reasons.append("尾段未命中主题词，疑似后半段跑题")
    if drift_score < config["min_drift_score"] and forbidden_hits:
        terms = [f"{hit['term']}({hit['severity']}x{hit['count']})" for hit in forbidden_hits[:8]]
        reasons.append(f"命中旧主题禁区：{', '.join(terms)}")
    if format_score < config["min_format_score"] and format_hits:
        reasons.append(f"出现格式漂移：{', '.join(format_hits)}")
    if language_score < config["min_language_score"] and language_noise_hits:
        reasons.append(f"出现语言噪声：{', '.join(language_noise_hits)}")
    if not reasons:
        reasons.append("规则检查通过")
    return reasons


def build_revision_instruction(
    *,
    contract: DiaryContract,
    reasons: list[str],
    forbidden_hits: list[dict],
    missing_topic_terms: list[str],
    format_hits: list[str],
    language_noise_hits: list[str],
) -> str:
    forbidden_terms = [hit["term"] for hit in forbidden_hits]
    lines = [
        "请重写为一篇新的日记正文。",
        f"必须围绕 MAIN_TOPIC：{contract.main_topic}",
        f"必须自然出现主题词：{', '.join(important_topic_terms(contract)[:8]) or '本次主题'}",
        "删除所有禁区内容，不要解释，不要列点，不要输出标题。",
    ]
    if reasons:
        lines.append(f"失败原因：{'；'.join(reasons)}")
    if forbidden_terms:
        lines.append(f"需要删除的禁区词：{', '.join(forbidden_terms[:12])}")
    if missing_topic_terms:
        lines.append(f"需要补回的主题词：{', '.join(missing_topic_terms[:8])}")
    if format_hits:
        lines.append(f"需要移除的格式问题：{', '.join(format_hits)}")
    if language_noise_hits:
        lines.append(f"需要移除的语言噪声：{', '.join(language_noise_hits)}")
    return "\n".join(lines)


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    return paragraphs or ([text.strip()] if text.strip() else [])


def tail_text(text: str) -> str:
    if not text:
        return ""
    start = max(0, int(len(text) * 0.7))
    return text[start:]


def split_topic_keywords(text: str) -> list[str]:
    return [part for part in re.split(r"[、/+&，,。.!！?？；;：:\s（）()“”\"<>《》]+", text) if part]


def clean_term(term: str) -> str:
    return str(term).strip(" -_[]【】()（）“”\"'‘’。，,；;：:!?！？")


def is_important_topic_term(term: str) -> bool:
    if not term or term in STYLE_OR_GENERIC_TERMS:
        return False
    if len(term) > 24:
        return False
    if re.fullmatch(r"\d+", term):
        return False
    return True


def is_searchable_forbidden_term(term: str) -> bool:
    if not term or len(term) > 24:
        return False
    if term in {"Markdown 表格", "代码块", "外部链接", "英文长串", "日语假名"}:
        return True
    return not term.startswith(("未授权", "混杂"))


def severity_for_term(term: str) -> str:
    if term in HIGH_SEVERITY_TERMS:
        return "high"
    if term in MEDIUM_SEVERITY_TERMS:
        return "medium"
    return "low"


def find_positions(text: str, term: str) -> list[int]:
    positions = []
    start = 0
    while True:
        position = text.find(term, start)
        if position == -1:
            break
        positions.append(position)
        start = position + max(1, len(term))
    if not positions and re.search(r"[A-Za-z]", term):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        positions = [match.start() for match in pattern.finditer(text)]
    return positions


def contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    compact_text = compact(text).lower()
    compact_term = compact(term).lower()
    if compact_term in compact_text:
        return True
    return any(compact(alias).lower() in compact_text for alias in TERM_ALIASES.get(term, []))


def compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _smoke_cases() -> list[dict]:
    return [
        {
            "name": "normal_pass",
            "prompt": "山东深夜在宿舍开启红烧肉和葱油饼制作模式，结果全寝室在醋香中被迫围观。",
            "text": "山东晚上在宿舍做红烧肉和葱油饼，醋香直接铺满了整个寝室。大家被迫围观，我脑子里只有一句：这锅怕不是酸菜鱼副本。",
            "expect": {"pass"},
        },
        {
            "name": "drift_revise",
            "prompt": "今天和朋友吃饭，很开心。",
            "text": "今天吃饭之后我开始研究股票和算法，顺便复习了高数作业。",
            "expect": {"revise", "fail"},
        },
        {
            "name": "format_revise",
            "prompt": "今天和朋友吃饭，很开心。",
            "text": "# 标题\n\n| 项目 | 内容 |\n| --- | --- |\n| 吃饭 | 开心 |",
            "expect": {"revise", "fail"},
        },
        {
            "name": "language_revise",
            "prompt": "今天和朋友吃饭，很开心。",
            "text": "今天和朋友吃饭很开心。これはテストです。abcdefghijklmnopqrstuvwxyzabcdefg",
            "expect": {"revise", "fail"},
        },
        {
            "name": "short_drift_revise",
            "prompt": "cat",
            "text": "今天去图书馆复习高数和线性代数，顺便写了作业。",
            "expect": {"revise", "fail"},
        },
    ]


def main() -> None:
    payload = []
    for case in _smoke_cases():
        contract = build_contract(case["prompt"])
        result = guard_diary(case["text"], contract)
        data = result.to_dict()
        data["name"] = case["name"]
        data["prompt"] = case["prompt"]
        payload.append(data)
        if result.decision not in case["expect"]:
            raise AssertionError(f"{case['name']} decision={result.decision}, expect={case['expect']}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
