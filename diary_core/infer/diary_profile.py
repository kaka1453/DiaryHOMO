from __future__ import annotations

from typing import Any
import re

from diary_core.infer.prompt_contract import DiaryContract


DEFAULT_DIARY_PROFILE_CONFIG = {
    "enabled": False,
    "profile_path": "config/diary_profile.yaml",
    "profile": {},
}

FACT_RELEVANCE_KEYWORDS = {
    "identity": ["自我", "自己", "外貌", "身高", "普通", "平凡", "儒雅", "温和"],
    "ability_and_history": ["学习", "学校", "大学", "双非", "鼠专", "考研", "NUS", "导数", "高数", "课程", "图书馆", "树冠", "报告"],
    "temperament": ["焦虑", "抑郁", "害怕", "决策", "胆子", "无畏", "慕强", "摆烂", "躺平", "自嘲"],
    "relationships": ["恋爱", "表白", "喜欢", "爱你", "感情", "女生", "SQ", "苏颀", "周瑜", "朋友", "室友"],
    "interests": ["股票", "钱", "金钱", "赚钱", "交易", "科技", "AI", "Cursor", "工具", "游戏", "抽卡"],
}

HISTORY_TRIGGER_TERMS = [
    "NUS",
    "鼠专",
    "SQ",
    "苏颀",
    "周瑜",
    "kaka",
    "hb",
    "山东",
    "树冠",
    "圣地",
    "考研",
    "股票",
    "高考",
    "恋爱",
]


def normalize_diary_profile_config(config: dict | None) -> dict:
    normalized = dict(DEFAULT_DIARY_PROFILE_CONFIG)
    if isinstance(config, dict):
        normalized.update(config)
    normalized["enabled"] = _as_bool(normalized.get("enabled", False))
    normalized["profile"] = normalized.get("profile") if isinstance(normalized.get("profile"), dict) else {}
    return normalized


def select_profile_attachments(contract: DiaryContract, diary_profile_config: dict | None) -> dict:
    config = normalize_diary_profile_config(diary_profile_config)
    profile = config.get("profile") or {}
    if not config["enabled"] or not profile:
        return {
            "status": "disabled" if not config["enabled"] else "empty_profile",
            "items": [],
            "debug": {
                "enabled": config["enabled"],
                "profile_path": config.get("profile_path"),
                "decisions": [{"section": "diary_profile", "injected": False, "reason": "disabled_or_empty"}],
                "matched_glossary_terms": [],
                "total_render_chars": 0,
                "max_render_chars": 0,
            },
        }

    injection = profile.get("injection") or {}
    max_items = int(injection.get("max_items_per_prompt") or 5)
    max_render_chars = int(injection.get("max_render_chars") or 1800)
    priority = injection.get("priority") or ["style_card", "term_glossary", "fallback_policy", "fact_card", "history_summary"]
    source_text = _source_text(contract)
    decisions = []

    candidates: dict[str, dict] = {}
    _maybe_add_style_card(candidates, decisions, profile)
    matched_terms = _matched_glossary_terms(profile, source_text)
    _maybe_add_term_glossary(candidates, decisions, matched_terms)
    _maybe_add_fallback_policy(candidates, decisions, profile, contract)
    _maybe_add_fact_card(candidates, decisions, profile, source_text)
    _maybe_add_history_summary(candidates, decisions, profile, source_text, matched_terms)

    ordered = []
    for section in priority:
        if section in candidates:
            ordered.append(candidates[section])
    for section, item in candidates.items():
        if section not in priority:
            ordered.append(item)

    selected = _fit_budget(ordered[:max_items], max_render_chars)
    total_chars = sum(item.get("render_chars", 0) for item in selected)
    return {
        "status": "enabled",
        "items": selected,
        "debug": {
            "enabled": True,
            "profile_path": config.get("profile_path"),
            "version": profile.get("version"),
            "source_note": profile.get("source_note"),
            "decisions": decisions,
            "matched_glossary_terms": [item["term"] for item in matched_terms],
            "matched_glossary_aliases": {
                item["term"]: item.get("matched_aliases", [])
                for item in matched_terms
            },
            "total_render_chars": total_chars,
            "max_render_chars": max_render_chars,
            "selected_sections": [item["section"] for item in selected],
        },
    }


def _maybe_add_style_card(candidates: dict, decisions: list[dict], profile: dict) -> None:
    style_card = profile.get("style_card") or {}
    if not style_card:
        decisions.append({"section": "style_card", "injected": False, "reason": "missing_style_card"})
        return
    lines = []
    lines.extend(_prefixed(style_card.get("core_style"), "风格"))
    phrase_bank = style_card.get("phrase_bank") or {}
    common_phrases = phrase_bank.get("common") or []
    if common_phrases:
        lines.append("口癖少量自然使用：" + "、".join(str(item) for item in common_phrases[:9]))
    lines.extend(_prefixed(style_card.get("scene_writing_moves"), "写法"))
    lines.extend(_prefixed(style_card.get("anti_generic"), "避免"))
    length_style = style_card.get("length_style") or {}
    for key in ["short_prompt", "normal_prompt", "humor_prompt"]:
        if length_style.get(key):
            lines.append(f"{key}: {length_style[key]}")
    candidates["style_card"] = _item(
        section="style_card",
        title="STYLE_CARD",
        content=lines,
        reason="always_style_card",
    )
    decisions.append({"section": "style_card", "injected": True, "reason": "always_style_card"})


def _maybe_add_term_glossary(candidates: dict, decisions: list[dict], matched_terms: list[dict]) -> None:
    if not matched_terms:
        decisions.append({"section": "term_glossary", "injected": False, "reason": "no_glossary_term_matched"})
        return
    lines = []
    for item in matched_terms[:8]:
        aliases = item.get("aliases") or []
        alias_text = f" aliases={','.join(aliases)}" if aliases else ""
        lines.append(f"{item['term']}{alias_text}: {item['meaning']}")
    candidates["term_glossary"] = _item(
        section="term_glossary",
        title="TERM_GLOSSARY",
        content=lines,
        reason="glossary_when_term_matched",
        meta={"matched_terms": [item["term"] for item in matched_terms]},
    )
    decisions.append(
        {
            "section": "term_glossary",
            "injected": True,
            "reason": "glossary_when_term_matched",
            "matched_terms": [item["term"] for item in matched_terms],
        }
    )


def _maybe_add_fallback_policy(candidates: dict, decisions: list[dict], profile: dict, contract: DiaryContract) -> None:
    fallback = profile.get("fallback_policy") or {}
    if not fallback:
        decisions.append({"section": "fallback_policy", "injected": False, "reason": "missing_fallback_policy"})
        return
    reason = "default_missing_facts_policy"
    if getattr(contract, "length_hint", "") == "short" or "short_input_expansion_risk" in set(getattr(contract, "risk_tags", [])):
        reason = "short_prompt_missing_facts_policy"
    elif getattr(contract, "forbidden_fact_types", []):
        reason = "fact_boundary_missing_facts_policy"
    lines = []
    lines.extend(_prefixed(fallback.get("principle"), "原则"))
    lines.extend(_prefixed(fallback.get("allowed_when_missing_facts"), "允许"))
    lines.extend(_prefixed(fallback.get("forbidden_when_missing_facts"), "禁止"))
    lines.extend(_prefixed(fallback.get("quality_floor"), "质量底线"))
    candidates["fallback_policy"] = _item(
        section="fallback_policy",
        title="MISSING_FACTS_FALLBACK_POLICY",
        content=lines,
        reason=reason,
    )
    decisions.append({"section": "fallback_policy", "injected": True, "reason": reason})


def _maybe_add_fact_card(candidates: dict, decisions: list[dict], profile: dict, source_text: str) -> None:
    fact_card = profile.get("fact_card") or {}
    if not fact_card:
        decisions.append({"section": "fact_card", "injected": False, "reason": "missing_fact_card"})
        return
    relevant_sections = []
    for section, keywords in FACT_RELEVANCE_KEYWORDS.items():
        if any(_contains(source_text, keyword) for keyword in keywords):
            relevant_sections.append(section)
    if not relevant_sections:
        decisions.append({"section": "fact_card", "injected": False, "reason": "no_profile_relevant_topic"})
        return
    lines = []
    for section in relevant_sections:
        lines.extend(_prefixed(fact_card.get(section), section))
    lines.extend(_prefixed(fact_card.get("usage_limits"), "使用限制"))
    candidates["fact_card"] = _item(
        section="fact_card",
        title="FACT_CARD",
        content=lines,
        reason="fact_card_when_profile_relevant",
        meta={"relevant_sections": relevant_sections},
    )
    decisions.append(
        {
            "section": "fact_card",
            "injected": True,
            "reason": "fact_card_when_profile_relevant",
            "relevant_sections": relevant_sections,
        }
    )


def _maybe_add_history_summary(
    candidates: dict,
    decisions: list[dict],
    profile: dict,
    source_text: str,
    matched_terms: list[dict],
) -> None:
    history = profile.get("history_summary") or {}
    if not history:
        decisions.append({"section": "history_summary", "injected": False, "reason": "missing_history_summary"})
        return
    matched_history = [term for term in HISTORY_TRIGGER_TERMS if _contains(source_text, term)]
    matched_profile_terms = [item["term"] for item in matched_terms if item.get("term") in HISTORY_TRIGGER_TERMS]
    if not matched_history and not matched_profile_terms:
        decisions.append({"section": "history_summary", "injected": False, "reason": "no_old_topic_or_proper_noun"})
        return
    lines = []
    lines.extend(_prefixed(history.get("stable_arcs"), "背景"))
    lines.extend(_prefixed(history.get("use_rules"), "使用规则"))
    candidates["history_summary"] = _item(
        section="history_summary",
        title="HISTORY_SUMMARY",
        content=lines,
        reason="history_summary_when_prompt_mentions_old_topics",
        meta={"matched_history_terms": _unique([*matched_history, *matched_profile_terms])},
    )
    decisions.append(
        {
            "section": "history_summary",
            "injected": True,
            "reason": "history_summary_when_prompt_mentions_old_topics",
            "matched_history_terms": _unique([*matched_history, *matched_profile_terms]),
        }
    )


def _matched_glossary_terms(profile: dict, source_text: str) -> list[dict]:
    terms = ((profile.get("term_glossary") or {}).get("terms") or {})
    matched = []
    for term, data in terms.items():
        aliases = [str(item) for item in (data.get("aliases") or [])]
        probes = [str(term), *aliases]
        matched_aliases = [probe for probe in probes if _contains(source_text, probe)]
        if not matched_aliases:
            continue
        matched.append(
            {
                "term": str(term),
                "aliases": aliases,
                "matched_aliases": matched_aliases,
                "meaning": str(data.get("meaning") or ""),
            }
        )
    return matched


def _fit_budget(items: list[dict], max_render_chars: int) -> list[dict]:
    selected = []
    used = 0
    for item in items:
        remaining = max_render_chars - used
        if remaining <= 0:
            break
        item = dict(item)
        render_text = str(item.get("render_text") or "")
        if len(render_text) > remaining:
            render_text = render_text[: max(0, remaining - 12)].rstrip() + "\n- ...已按预算截断"
            item["truncated"] = True
        item["render_text"] = render_text
        item["render_chars"] = len(render_text)
        selected.append(item)
        used += item["render_chars"]
    return selected


def _item(section: str, title: str, content: list[str], reason: str, meta: dict | None = None) -> dict:
    lines = [str(item).strip() for item in content if str(item).strip()]
    render_text = "\n".join(f"- {line}" for line in lines)
    return {
        "type": "diary_profile",
        "section": section,
        "title": title,
        "content": lines,
        "render_text": render_text,
        "render_chars": len(render_text),
        "reason": reason,
        "meta": meta or {},
    }


def _prefixed(values, prefix: str) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    return [f"{prefix}: {value}" for value in values if str(value).strip()]


def _source_text(contract: DiaryContract) -> str:
    return " ".join(
        [
            contract.raw_prompt,
            contract.main_topic,
            " ".join(getattr(contract, "topic_terms", []) or []),
        ]
    )


def _unique(items: list[str]) -> list[str]:
    result = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def _contains(text: str, keyword: str) -> bool:
    if not keyword:
        return False
    return _compact(keyword) in _compact(text)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
