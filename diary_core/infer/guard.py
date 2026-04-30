from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal

from diary_core.infer.prompt_contract import DiaryContract, build_contract


GuardDecision = Literal["pass", "pass_with_warnings", "revise", "fail"]


DEFAULT_GUARD_CONFIG: dict[str, Any] = {
    "enabled": True,
    "max_retry": 1,
    "retry_on_revise": True,
    "retry_on_warnings": False,
    "pass_with_warnings": True,
    "fail_strategy": "best_attempt",
    "thresholds": {
        "min_final_score": 75,
        "min_topic_score": 55,
        "min_drift_score": 85,
        "min_format_score": 85,
        "min_language_score": 75,
        "min_quality_score": 60,
    },
    "extra_forbidden_terms": [
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
        "虚无妹妹",
        "b政治",
        "数学课",
        "高中作业",
        "晚自习",
        "课程预习",
    ],
    "severity": {
        "high": [
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
            "YouTube",
            "youtube",
            "B站",
            "GitHub",
            "虚无妹妹",
            "b政治",
        ],
        "medium": [
            "作业",
            "复习",
            "课程",
            "AirPods",
            "Genshin",
            "雅思",
            "CET6",
            "高数",
            "线性代数",
            "yyj",
            "b计算机",
            "冰激凌",
            "钓鱼用品店",
            "浩海钓具",
            "反向伞",
        ],
    },
    "term_aliases": {
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
        "cat": ["邪恶表妹", "表妹"],
        "邪恶表妹": ["cat", "表妹"],
    },
    "quality": {
        "enabled": True,
        "min_quality_score": 60,
        "min_length_by_type": {
            "keyword": 40,
            "short_daily": 50,
            "question": 50,
            "opening": 90,
            "instruction": 120,
            "style_instruction": 120,
            "simulation_instruction": 120,
        },
        "max_prompt_copy_ratio": 0.45,
        "generic_template_penalty": True,
        "repetition_penalty_enabled": True,
        "require_scene_for_humor": True,
        "generic_template_terms": [
            "总的来说",
            "这种体验让我",
            "这让我意识到",
            "这是一种",
            "也许这就是生活",
            "成为了共同的记忆",
            "我会继续努力",
            "明天会更好",
            "就记录到这里",
            "今天的日记到这里",
            "每个人都有自己的生活",
            "这是值得我学习的",
            "生活中的小确幸",
            "迎接明天",
            "按部就班",
        ],
        "old_memory_risk_terms": [
            "cat弟妹",
            "弟妹",
            "武汉",
            "双休日",
            "小时候",
            "以前",
            "过去",
            "回家",
            "家里人",
            "同学",
            "学校",
            "图书馆",
            "实验室",
            "数学课",
            "作业",
            "复习",
            "b政治",
            "虚无妹妹",
        ],
        "scene_terms": [
            "看到",
            "听到",
            "走",
            "坐",
            "拿",
            "打开",
            "关上",
            "回到",
            "桌",
            "床",
            "门",
            "宿舍",
            "寝室",
            "图书馆",
            "餐厅",
            "书房",
            "笑",
            "问",
            "说",
            "递",
            "看着",
            "闻到",
        ],
        "humor_scene_terms": [
            "吐槽",
            "OS",
            "脑子",
            "难蚌",
            "蚌",
            "寄",
            "草",
            "笑",
            "离谱",
            "破防",
            "当场",
            "邪恶",
            "恼",
        ],
    },
    "audit": {
        "include_guard_table": True,
        "include_warnings": True,
        "mode": "html_comment",
    },
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

TRADITIONAL_CHARS = set(
    "學習體驗關係個這樣說話還會沒為與對時後點裡讓過來麼"
    "氣實現發現問題應該準備複雜總結開始"
)

HUMOR_HINTS = {"搞笑", "幽默", "吐槽", "荒诞", "离谱", "难蚌", "戏剧化", "自嘲", "反差", "段子", "夸张", "破防", "场景化"}


@dataclass
class GuardResult:
    enabled: bool
    decision: GuardDecision
    final_score: float
    topic_score: float
    drift_score: float
    format_score: float
    language_score: float
    quality_score: float
    topic_hits: list[str]
    missing_topic_terms: list[str]
    forbidden_hits: list[dict]
    format_hits: list[str]
    language_noise_hits: list[str]
    tail_drift: bool
    warnings: list[str]
    quality_warnings: list[str]
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
            quality_score=100.0,
            topic_hits=[],
            missing_topic_terms=[],
            forbidden_hits=[],
            format_hits=[],
            language_noise_hits=[],
            tail_drift=False,
            warnings=[],
            quality_warnings=[],
            reasons=["guard disabled"],
            revision_instruction="",
        )

    topic_score, topic_hits, missing_topic_terms, tail_drift = score_topic(text, contract, guard_config)
    forbidden_terms = build_guard_forbidden_terms(contract, guard_config)
    forbidden_hits = find_forbidden_hits(text, forbidden_terms, guard_config)
    drift_score = score_drift(forbidden_hits)
    format_hits, format_score = check_format(text)
    language_noise_hits, language_score = check_language_noise(text, contract)
    quality_score, quality_warnings = check_quality(text, contract, guard_config)
    warnings = unique_preserve_order(quality_warnings)

    final_score = round(
        topic_score * 0.32
        + drift_score * 0.30
        + format_score * 0.12
        + language_score * 0.10
        + quality_score * 0.16,
        2,
    )
    reasons = build_reasons(
        topic_score=topic_score,
        drift_score=drift_score,
        format_score=format_score,
        language_score=language_score,
        quality_score=quality_score,
        missing_topic_terms=missing_topic_terms,
        forbidden_hits=forbidden_hits,
        format_hits=format_hits,
        language_noise_hits=language_noise_hits,
        quality_warnings=quality_warnings,
        tail_drift=tail_drift,
        config=guard_config,
    )
    decision = decide(
        final_score=final_score,
        topic_score=topic_score,
        drift_score=drift_score,
        format_score=format_score,
        language_score=language_score,
        quality_score=quality_score,
        warnings=warnings,
        config=guard_config,
    )
    revision_instruction = build_revision_instruction(
        contract=contract,
        reasons=reasons,
        forbidden_hits=forbidden_hits,
        missing_topic_terms=missing_topic_terms,
        format_hits=format_hits,
        language_noise_hits=language_noise_hits,
        quality_warnings=quality_warnings,
    )

    return GuardResult(
        enabled=True,
        decision=decision,
        final_score=final_score,
        topic_score=round(topic_score, 2),
        drift_score=round(drift_score, 2),
        format_score=round(format_score, 2),
        language_score=round(language_score, 2),
        quality_score=round(quality_score, 2),
        topic_hits=topic_hits,
        missing_topic_terms=missing_topic_terms,
        forbidden_hits=forbidden_hits,
        format_hits=format_hits,
        language_noise_hits=language_noise_hits,
        tail_drift=tail_drift,
        warnings=warnings,
        quality_warnings=quality_warnings,
        reasons=reasons,
        revision_instruction=revision_instruction,
    )


def normalize_guard_config(config: dict | None = None) -> dict:
    normalized = deep_merge_dict(DEFAULT_GUARD_CONFIG, config or {})
    normalized["enabled"] = _as_bool(normalized.get("enabled", True))
    normalized["max_retry"] = int(normalized.get("max_retry", 1))
    normalized["retry_on_revise"] = _as_bool(normalized.get("retry_on_revise", True))
    normalized["retry_on_warnings"] = _as_bool(normalized.get("retry_on_warnings", False))
    normalized["pass_with_warnings"] = _as_bool(normalized.get("pass_with_warnings", True))
    normalized["fail_strategy"] = str(normalized.get("fail_strategy") or "best_attempt")

    thresholds = deep_merge_dict(DEFAULT_GUARD_CONFIG["thresholds"], normalized.get("thresholds") or {})
    for key in list(DEFAULT_GUARD_CONFIG["thresholds"]):
        if key in normalized and normalized.get(key) is not None:
            thresholds[key] = normalized[key]
        normalized[key] = float(thresholds.get(key, DEFAULT_GUARD_CONFIG["thresholds"][key]))
        thresholds[key] = normalized[key]
    normalized["thresholds"] = thresholds

    quality = deep_merge_dict(DEFAULT_GUARD_CONFIG["quality"], normalized.get("quality") or {})
    quality["enabled"] = _as_bool(quality.get("enabled", True))
    quality["generic_template_penalty"] = _as_bool(quality.get("generic_template_penalty", True))
    quality["repetition_penalty_enabled"] = _as_bool(quality.get("repetition_penalty_enabled", True))
    quality["require_scene_for_humor"] = _as_bool(quality.get("require_scene_for_humor", True))
    quality["min_quality_score"] = float(quality.get("min_quality_score", normalized["min_quality_score"]))
    normalized["min_quality_score"] = quality["min_quality_score"]
    quality["max_prompt_copy_ratio"] = float(quality.get("max_prompt_copy_ratio", 0.45))
    quality["min_length_by_type"] = {
        str(key): int(value)
        for key, value in (quality.get("min_length_by_type") or {}).items()
    }
    for list_key in ["generic_template_terms", "old_memory_risk_terms", "scene_terms", "humor_scene_terms"]:
        quality[list_key] = normalize_string_list(quality.get(list_key))
    normalized["quality"] = quality

    severity = normalized.get("severity") if isinstance(normalized.get("severity"), dict) else {}
    normalized["severity"] = {
        "high": normalize_string_list(severity.get("high")),
        "medium": normalize_string_list(severity.get("medium")),
    }
    normalized["extra_forbidden_terms"] = normalize_string_list(normalized.get("extra_forbidden_terms"))
    normalized["term_aliases"] = normalize_aliases(normalized.get("term_aliases"))
    audit = deep_merge_dict(DEFAULT_GUARD_CONFIG["audit"], normalized.get("audit") or {})
    audit["include_guard_table"] = _as_bool(audit.get("include_guard_table", True))
    audit["include_warnings"] = _as_bool(audit.get("include_warnings", True))
    normalized["audit"] = audit
    return normalized


def build_guard_forbidden_terms(contract: DiaryContract, config: dict) -> list[str]:
    authorized_text = " ".join([contract.raw_prompt, contract.main_topic, *getattr(contract, "topic_terms", [])])
    terms = [*getattr(contract, "forbidden_drift_topics", []), *config.get("extra_forbidden_terms", [])]
    aliases = config.get("term_aliases") or {}
    return [
        term
        for term in unique_preserve_order([str(item).strip() for item in terms])
        if term and not contains_term(authorized_text, term, aliases)
    ]


def score_topic(text: str, contract: DiaryContract, config: dict) -> tuple[float, list[str], list[str], bool]:
    terms = important_topic_terms(contract)
    aliases = config.get("term_aliases") or {}
    if not terms:
        return 70.0, [], [], False

    topic_hits = [term for term in terms if contains_term(text, term, aliases)]
    missing_topic_terms = [term for term in terms if term not in topic_hits]
    hit_rate = len(topic_hits) / max(1, len(terms))

    paragraphs = split_paragraphs(text)
    if paragraphs:
        paragraph_hits = sum(1 for paragraph in paragraphs if any(contains_term(paragraph, term, aliases) for term in terms))
        paragraph_rate = paragraph_hits / len(paragraphs)
    else:
        paragraph_rate = 0.0

    tail = tail_text(text)
    tail_topic_ok = any(contains_term(tail, term, aliases) for term in terms)
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


def find_forbidden_hits(text: str, forbidden_drift_topics: list[str], config: dict) -> list[dict]:
    hits = []
    aliases = config.get("term_aliases") or {}
    for term in unique_preserve_order(forbidden_drift_topics):
        term = str(term).strip()
        if not is_searchable_forbidden_term(term):
            continue
        positions = find_positions_with_aliases(text, term, aliases)
        if not positions:
            continue
        hits.append(
            {
                "term": term,
                "count": len(positions),
                "severity": severity_for_term(term, config),
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


def check_quality(text: str, contract: DiaryContract, config: dict) -> tuple[float, list[str]]:
    quality_config = config.get("quality") or {}
    if not _as_bool(quality_config.get("enabled", True)):
        return 100.0, []

    warnings = []
    score = 100.0
    stripped = text.strip()
    text_len = len(compact(stripped))
    aliases = config.get("term_aliases") or {}

    if not stripped:
        return 0.0, ["empty_output"]

    min_length = int((quality_config.get("min_length_by_type") or {}).get(contract.prompt_type, 70))
    if text_len < min_length:
        warnings.append(f"too_short:{text_len}<{min_length}")
        score -= 25 if text_len < min_length * 0.6 else 15

    copy_ratio = prompt_copy_ratio(stripped, contract.raw_prompt)
    max_copy_ratio = float(quality_config.get("max_prompt_copy_ratio", 0.45))
    if len(compact(contract.raw_prompt)) >= 12 and copy_ratio > max_copy_ratio:
        warnings.append(f"prompt_copy_ratio_high={copy_ratio:.2f}")
        score -= 30 if copy_ratio > 0.75 else 20

    if _as_bool(quality_config.get("generic_template_penalty", True)):
        generic_hits = [term for term in quality_config.get("generic_template_terms", []) if term and term in stripped]
        if generic_hits:
            warnings.append(f"generic_template:{'/'.join(generic_hits[:4])}")
            score -= min(35, 10 * len(generic_hits))

    mixed_english = unexpected_english_tokens(stripped, contract)
    if mixed_english:
        warnings.append(f"mixed_english_token:{'/'.join(mixed_english[:3])}")
        score -= 10

    scene_terms = quality_config.get("scene_terms") or []
    scene_hits = [term for term in scene_terms if contains_term(stripped, term, aliases)]
    if text_len >= 45 and len(scene_hits) == 0:
        warnings.append("low_detail:no_scene_action")
        score -= 18

    if is_humor_contract(contract) and _as_bool(quality_config.get("require_scene_for_humor", True)):
        humor_terms = quality_config.get("humor_scene_terms") or []
        humor_hits = [term for term in humor_terms if contains_term(stripped, term, aliases)]
        if len(scene_hits) < 1 or len(humor_hits) < 1:
            warnings.append("humor_missing_scene_or_tucao")
            score -= 20

    if contract.prompt_type in {"keyword", "short_daily"} or getattr(contract, "length_hint", "") == "short":
        authorized_text = " ".join([contract.raw_prompt, contract.main_topic, *getattr(contract, "topic_terms", [])])
        old_hits = [
            term
            for term in quality_config.get("old_memory_risk_terms", [])
            if contains_term(stripped, term, aliases) and not contains_term(authorized_text, term, aliases)
        ]
        if old_hits:
            warnings.append(f"short_prompt_old_memory_risk:{'/'.join(old_hits[:5])}")
            score -= 25

    if _as_bool(quality_config.get("repetition_penalty_enabled", True)):
        repetition_warning = detect_repetition_warning(stripped, contract)
        if repetition_warning:
            warnings.append(repetition_warning)
            score -= 15

    return _clamp(score), unique_preserve_order(warnings)


def decide(
    final_score: float,
    topic_score: float,
    drift_score: float,
    format_score: float,
    language_score: float,
    quality_score: float,
    warnings: list[str],
    config: dict,
) -> GuardDecision:
    if "empty_output" in warnings:
        return "fail"
    if quality_score < 35 and final_score >= 50:
        return "revise"
    if (
        final_score >= config["min_final_score"]
        and topic_score >= config["min_topic_score"]
        and drift_score >= config["min_drift_score"]
        and format_score >= config["min_format_score"]
        and language_score >= config["min_language_score"]
    ):
        if warnings or quality_score < config["min_quality_score"]:
            if config.get("retry_on_warnings") and quality_score < config["min_quality_score"]:
                return "revise"
            return "pass_with_warnings" if config.get("pass_with_warnings", True) else "pass"
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
    quality_score: float,
    missing_topic_terms: list[str],
    forbidden_hits: list[dict],
    format_hits: list[str],
    language_noise_hits: list[str],
    quality_warnings: list[str],
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
    if quality_score < config["min_quality_score"] or quality_warnings:
        reasons.append(f"质量警告：{', '.join(quality_warnings[:8]) or '质量分偏低'}")
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
    quality_warnings: list[str],
) -> str:
    forbidden_terms = [hit["term"] for hit in forbidden_hits]
    important_terms = important_topic_terms(contract)
    required_n = min(3, max(1, len(important_terms)))
    lines = [
        "请重写为一篇新的日记正文。",
        f"必须围绕 MAIN_TOPIC：{contract.main_topic}",
        f"必须自然出现以下主题锚点中的至少 {required_n} 个：{', '.join(important_terms[:8]) or '本次主题'}",
        "允许围绕 MAIN_TOPIC 做主旨内的合理想象展开：补现场动作、内心 OS、一句小对话、轻微夸张比喻或直接相关的小后果。",
        "不允许跳到股票/算法/考试/旧聊天/旧人物长故事/未授权地点，也不要从当前 prompt 扩成完全无关的历史回忆。",
        "删除所有禁区内容，不要解释，不要列点，不要输出标题。",
        "不要只复述 MAIN_TOPIC，不要升华成大道理，不要写成总结报告。",
    ]
    if is_humor_contract(contract):
        lines.append("如果原 prompt 有搞笑/难蚌/离谱/吐槽感，重写后必须保留笑点锐度和场景感。")
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
    if quality_warnings:
        lines.append(f"需要改善的质量问题：{', '.join(quality_warnings[:8])}")
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


def severity_for_term(term: str, config: dict) -> str:
    severity = config.get("severity") or {}
    if term in severity.get("high", set()):
        return "high"
    if term in severity.get("medium", set()):
        return "medium"
    return "low"


def find_positions_with_aliases(text: str, term: str, aliases: dict[str, list[str]] | None = None) -> list[int]:
    positions = find_positions(text, term)
    for alias in (aliases or {}).get(term, []):
        positions.extend(find_positions(text, alias))
    return sorted(set(positions))


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


def contains_term(text: str, term: str, aliases: dict[str, list[str]] | None = None) -> bool:
    if not term:
        return False
    compact_text = compact(text).lower()
    compact_term = compact(term).lower()
    if compact_term in compact_text:
        return True
    return any(compact(alias).lower() in compact_text for alias in (aliases or {}).get(term, []))


def prompt_copy_ratio(text: str, raw_prompt: str) -> float:
    prompt = compact_for_overlap(raw_prompt)
    output = compact_for_overlap(text)
    if len(prompt) < 8 or not output:
        return 0.0
    if prompt in output:
        return 1.0
    return longest_common_substring_len(prompt, output) / max(1, len(prompt))


def longest_common_substring_len(a: str, b: str) -> int:
    previous = [0] * (len(b) + 1)
    best = 0
    for char_a in a:
        current = [0]
        for index_b, char_b in enumerate(b, start=1):
            value = previous[index_b - 1] + 1 if char_a == char_b else 0
            current.append(value)
            if value > best:
                best = value
        previous = current
    return best


def detect_repetition_warning(text: str, contract: DiaryContract) -> str:
    sentences = [item.strip() for item in re.split(r"[。！？!?；;\n]+", text) if item.strip()]
    if len(sentences) >= 4:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.65:
            return "repetition:sentence_loop"
    terms = important_topic_terms(contract)[:3]
    if len(text) >= 120 and terms:
        counts = [text.count(term) for term in terms if len(term) >= 2]
        if counts and max(counts) >= max(5, len(text) // 45):
            return "repetition:topic_term_overused"
    return ""


def is_humor_contract(contract: DiaryContract) -> bool:
    styles = set(getattr(contract, "style_hints", []) or [])
    risks = set(getattr(contract, "risk_tags", []) or [])
    return bool(styles & HUMOR_HINTS or "humor_or_absurd_prompt" in risks)


def unexpected_english_tokens(text: str, contract: DiaryContract) -> list[str]:
    tokens = re.findall(r"[A-Za-z]{5,}", text)
    if not tokens:
        return []
    authorized_text = " ".join([contract.raw_prompt, contract.main_topic, *getattr(contract, "topic_terms", [])])
    authorized = {token.lower() for token in re.findall(r"[A-Za-z]{2,}", authorized_text)}
    allowlist = {"markdown", "html", "http", "https"}
    unexpected = []
    for token in tokens:
        normalized = token.lower()
        if normalized in authorized or normalized in allowlist:
            continue
        unexpected.append(token)
    return unique_preserve_order(unexpected)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def compact_for_overlap(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?；;：:（）()“”\"'‘’《》<>【】\[\]-]+", "", str(text))


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def normalize_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if str(item)]


def normalize_aliases(value) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    aliases: dict[str, list[str]] = {}
    for key, items in value.items():
        aliases[str(key)] = normalize_string_list(items)
    return aliases


def deep_merge_dict(base: dict, overrides: dict | None) -> dict:
    merged = dict(base)
    if not isinstance(overrides, dict):
        return merged
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def summarize_debug(debug_root: str | Path, output_jsonl: str | None = None) -> list[dict]:
    root = Path(debug_root)
    rows = []
    for guard_path in sorted(root.rglob("09_guard.json")):
        try:
            data = json.loads(guard_path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"debug_dir": str(guard_path.parent), "error": str(exc)})
            continue
        prompt_path = guard_path.parent / "00_raw_input.txt"
        prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else ""
        selected_guard = selected_guard_from_summary(data)
        rows.append(
            {
                "prompt": prompt,
                "decision": selected_guard.get("decision", data.get("decision", "")),
                "final_score": selected_guard.get("final_score"),
                "topic_score": selected_guard.get("topic_score"),
                "drift_score": selected_guard.get("drift_score"),
                "format_score": selected_guard.get("format_score"),
                "language_score": selected_guard.get("language_score"),
                "quality_score": selected_guard.get("quality_score"),
                "retry_count": data.get("retry_count", 0),
                "warnings": selected_guard.get("warnings") or selected_guard.get("quality_warnings") or [],
                "debug_dir": str(guard_path.parent),
            }
        )
    print_guard_summary(rows)
    if output_jsonl:
        path = Path(output_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def selected_guard_from_summary(data: dict) -> dict:
    selected_attempt = data.get("selected_attempt")
    for attempt in data.get("attempts") or []:
        if attempt.get("attempt") == selected_attempt:
            return attempt.get("guard") or {}
    if data.get("attempts"):
        return (data["attempts"][0] or {}).get("guard") or {}
    return data


def print_guard_summary(rows: list[dict]) -> None:
    header = "prompt                         decision             final topic drift format lang quality retry warnings"
    print(header)
    print("-" * len(header))
    for row in rows:
        prompt = (row.get("prompt") or row.get("debug_dir") or "")[:28]
        warnings = ",".join(row.get("warnings") or [])[:34]
        print(
            f"{prompt:<28} "
            f"{str(row.get('decision', '')):<20} "
            f"{_fmt_score(row.get('final_score')):<5} "
            f"{_fmt_score(row.get('topic_score')):<5} "
            f"{_fmt_score(row.get('drift_score')):<5} "
            f"{_fmt_score(row.get('format_score')):<6} "
            f"{_fmt_score(row.get('language_score')):<4} "
            f"{_fmt_score(row.get('quality_score')):<7} "
            f"{row.get('retry_count', 0)!s:<5} "
            f"{warnings}"
        )


def _fmt_score(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.0f}"
    except (TypeError, ValueError):
        return str(value)


def _smoke_cases() -> list[dict]:
    return [
        {
            "name": "normal_pass",
            "prompt": "山东深夜在宿舍开启红烧肉和葱油饼制作模式，结果全寝室在醋香中被迫围观。",
            "text": "山东晚上在宿舍做红烧肉和葱油饼，醋香直接铺满了整个寝室。大家被迫围观，我脑子里只有一句：这锅怕不是酸菜鱼副本。",
            "expect": {"pass", "pass_with_warnings"},
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
        {
            "name": "quality_warning",
            "prompt": "山东深夜在宿舍开启红烧肉和葱油饼制作模式，结果全寝室在醋香中被迫围观。",
            "text": "山东深夜在宿舍开启红烧肉和葱油饼制作模式，结果全寝室在醋香中被迫围观。这让我意识到生活很有趣。",
            "expect": {"pass_with_warnings", "revise"},
        },
    ]


def run_smoke_cases() -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="DiaryGuard rule smoke test and debug summarizer")
    parser.add_argument("--summarize-debug", dest="summarize_debug", default=None)
    parser.add_argument("--output-jsonl", dest="output_jsonl", default=None)
    args = parser.parse_args()
    if args.summarize_debug:
        summarize_debug(args.summarize_debug, args.output_jsonl)
        return
    run_smoke_cases()


if __name__ == "__main__":
    main()
