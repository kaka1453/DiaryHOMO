from __future__ import annotations

from diary_core.infer.product_content import product_value, product_wording

import argparse
from dataclasses import asdict, dataclass
import json
import re


SHORT_PROMPT_MAX_CHARS = 12


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
        config.get("drift_topic_groups") or product_value('prompt_contract', 'DRIFT_TOPIC_GROUPS'),
    )
    extra_forbidden = config.get("forbidden_drift_topics") or config.get("forbidden_topics") or []
    forbidden_drift_topics = _unique_preserve_order([*forbidden_drift_topics, *extra_forbidden])
    format_forbidden = list(config.get("format_forbidden") or product_value('prompt_contract', 'DEFAULT_FORMAT_FORBIDDEN'))
    forbidden_topics = _unique_preserve_order([*forbidden_drift_topics, *format_forbidden])
    forbidden_fact_types = list(config.get("forbidden_fact_types") or product_value('prompt_contract', 'DEFAULT_FORBIDDEN_FACT_TYPES'))
    style_hints = _unique_preserve_order(
        [
            *(config.get("style_hints") or product_value('prompt_contract', 'DEFAULT_STYLE_HINTS')),
            *(intent.get("style_hints") or []),
        ]
    )

    if intent["is_short_prompt"]:
        topic_lock = (
            f"{product_wording('prompt_contract', 'build_contract', 0)}{main_topic}{product_wording('prompt_contract', 'build_contract', 1)}{_join_terms(topic_terms)}{product_wording('prompt_contract', 'build_contract', 2)}"
        )
        uncertainty_policy = product_wording('prompt_contract', 'build_contract', 3)
    else:
        topic_lock = (
            f"{product_wording('prompt_contract', 'build_contract', 4)}{main_topic}{product_wording('prompt_contract', 'build_contract', 5)}{_join_terms(topic_terms)}{product_wording('prompt_contract', 'build_contract', 6)}"
        )
        uncertainty_policy = product_wording('prompt_contract', 'build_contract', 7)

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
    for prompt_type, pattern in ((kind, re.compile(pattern)) for kind, pattern in product_value('prompt_contract', 'CONTRACT_PATTERNS')):
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
    if prompt.endswith(("?", "？")) or re.match(product_wording('prompt_contract', '_classify_prompt', 1), prompt):
        return "question"
    if match_info:
        return match_info["prompt_type"]
    compact = _compact(prompt)
    if compact in set(product_value('prompt_contract', 'SHORT_DAILY_PROMPTS')):
        return "short_daily"
    if _looks_like_opening(prompt):
        return "opening"
    if _looks_like_keyword(prompt):
        return "keyword"
    if re.search(product_wording('prompt_contract', '_classify_prompt', 0), prompt):
        return "instruction"
    if len(prompt) <= SHORT_PROMPT_MAX_CHARS:
        return "short_daily"
    return "opening"


def _infer_main_topic(prompt: str, prompt_type: str, match_info: dict, style_hints: list[str]) -> str:
    if not prompt:
        return product_wording('prompt_contract', '_infer_main_topic', 0)
    if prompt_type == "keyword":
        alias = product_value('prompt_contract', 'KEYWORD_ALIASES').get(_compact(prompt).lower(), prompt)
        return f"{alias}{product_wording('prompt_contract', '_infer_main_topic', 1)}"
    if prompt_type == "short_daily":
        return f"{prompt.strip('。.!！?？')}{product_wording('prompt_contract', '_infer_main_topic', 2)}"
    if prompt_type == "question":
        return f"{product_wording('prompt_contract', '_infer_main_topic', 3)}{prompt}{product_wording('prompt_contract', '_infer_main_topic', 4)}"

    topic = match_info.get("topic") if match_info else ""
    if not topic:
        topic = prompt
    topic = _clean_topic(topic)
    if not topic and match_info.get("topic") in {product_wording('prompt_contract', '_infer_main_topic', 7), product_wording('prompt_contract', '_infer_main_topic', 8)}:
        topic = product_wording('prompt_contract', '_infer_main_topic', 5)
    if not topic:
        topic = prompt[:80]

    style_topic_terms = [item for item in style_hints if item and item not in topic]
    if style_topic_terms:
        topic = f"{topic}{product_wording('prompt_contract', '_infer_main_topic', 6)}{'、'.join(style_topic_terms[:3])}"
    return topic[:120]


def _extract_allowed_facts(prompt: str, topic_terms: list[str] | None = None) -> list[str]:
    parts = re.split(r"[\s，。！？、,.!?；;：:\n]+", prompt)
    facts = list(topic_terms or [])
    for part in parts:
        item = part.strip()
        if not item or item in set(product_value('prompt_contract', 'STOPWORDS')):
            continue
        if 1 <= len(item) <= 24 and item not in facts:
            facts.append(item)
    return facts[:12]


def _extract_style_hints(prompt: str, match_info: dict | None = None) -> list[str]:
    hints = []
    style = (match_info or {}).get("style")
    if style:
        hints.extend(_split_terms(style))
    for keyword in product_value('prompt_contract', 'STYLE_KEYWORDS'):
        if keyword.lower() in prompt.lower():
            hints.append(keyword)
    if _has_humor_cue(prompt):
        hints.extend([product_wording('prompt_contract', '_extract_style_hints', 0), product_wording('prompt_contract', '_extract_style_hints', 1), product_wording('prompt_contract', '_extract_style_hints', 2), product_wording('prompt_contract', '_extract_style_hints', 3)])
    return _unique_preserve_order(
        [_clean_term(item) for item in hints if _is_valid_style_hint(_clean_term(item))]
    )


def _extract_topic_terms(prompt: str, main_topic: str) -> list[str]:
    compact_prompt = _compact(prompt).lower()
    if compact_prompt in product_value('prompt_contract', 'KEYWORD_ALIASES'):
        return _unique_preserve_order(_split_terms(product_value('prompt_contract', 'KEYWORD_ALIASES')[compact_prompt]))[:12]

    candidates = []
    candidates.extend(_extract_short_anchor_terms(prompt))
    candidates.extend(_split_terms(main_topic))
    candidates.extend(_extract_quoted_terms(prompt))
    source = f"{prompt} {main_topic}"
    for keyword in product_value('prompt_contract', 'DOMAIN_TOPIC_KEYWORDS'):
        if _contains_term(source, keyword):
            candidates.append(keyword)
    for keyword in product_value('prompt_contract', 'STYLE_KEYWORDS'):
        if keyword.lower() in prompt.lower():
            candidates.append(keyword)
    for group_terms in product_value('prompt_contract', 'DRIFT_TOPIC_GROUPS').values():
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


def _extract_short_anchor_terms(prompt: str) -> list[str]:
    anchors = []
    if product_wording('prompt_contract', '_extract_short_anchor_terms', 0) in prompt:
        anchors.append(product_wording('prompt_contract', '_extract_short_anchor_terms', 55))
    fake_work = re.search(product_wording('prompt_contract', '_extract_short_anchor_terms', 1), prompt)
    if fake_work:
        anchors.append(fake_work.group(0))
    if re.search(product_wording('prompt_contract', '_extract_short_anchor_terms', 2), prompt):
        anchors.append(product_wording('prompt_contract', '_extract_short_anchor_terms', 56))
    if product_wording('prompt_contract', '_extract_short_anchor_terms', 3) in prompt:
        anchors.append(product_wording('prompt_contract', '_extract_short_anchor_terms', 57))
    if product_wording('prompt_contract', '_extract_short_anchor_terms', 4) in prompt:
        anchors.append(product_wording('prompt_contract', '_extract_short_anchor_terms', 58))
    if product_wording('prompt_contract', '_extract_short_anchor_terms', 5) in prompt:
        anchors.append(product_wording('prompt_contract', '_extract_short_anchor_terms', 59))
    if product_wording('prompt_contract', '_extract_short_anchor_terms', 6) in prompt:
        anchors.append(product_wording('prompt_contract', '_extract_short_anchor_terms', 60))

    cue_to_anchor = {
        product_wording('prompt_contract', '_extract_short_anchor_terms', 7): product_wording('prompt_contract', '_extract_short_anchor_terms', 23),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 8): product_wording('prompt_contract', '_extract_short_anchor_terms', 24),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 9): product_wording('prompt_contract', '_extract_short_anchor_terms', 25),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 10): product_wording('prompt_contract', '_extract_short_anchor_terms', 26),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 11): product_wording('prompt_contract', '_extract_short_anchor_terms', 27),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 12): product_wording('prompt_contract', '_extract_short_anchor_terms', 28),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 13): product_wording('prompt_contract', '_extract_short_anchor_terms', 29),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 14): product_wording('prompt_contract', '_extract_short_anchor_terms', 30),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 15): product_wording('prompt_contract', '_extract_short_anchor_terms', 31),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 16): product_wording('prompt_contract', '_extract_short_anchor_terms', 32),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 17): product_wording('prompt_contract', '_extract_short_anchor_terms', 33),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 18): product_wording('prompt_contract', '_extract_short_anchor_terms', 34),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 19): product_wording('prompt_contract', '_extract_short_anchor_terms', 35),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 20): product_wording('prompt_contract', '_extract_short_anchor_terms', 36),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 21): product_wording('prompt_contract', '_extract_short_anchor_terms', 37),
        product_wording('prompt_contract', '_extract_short_anchor_terms', 22): product_wording('prompt_contract', '_extract_short_anchor_terms', 38),
    }
    for cue, anchor in cue_to_anchor.items():
        if cue in prompt:
            anchors.append(anchor)

    for term in [product_wording('prompt_contract', '_extract_short_anchor_terms', 39), product_wording('prompt_contract', '_extract_short_anchor_terms', 40), product_wording('prompt_contract', '_extract_short_anchor_terms', 41), "kaka", "LWZ", "SQ", product_wording('prompt_contract', '_extract_short_anchor_terms', 42), product_wording('prompt_contract', '_extract_short_anchor_terms', 43), product_wording('prompt_contract', '_extract_short_anchor_terms', 44), product_wording('prompt_contract', '_extract_short_anchor_terms', 45), product_wording('prompt_contract', '_extract_short_anchor_terms', 46), product_wording('prompt_contract', '_extract_short_anchor_terms', 47), product_wording('prompt_contract', '_extract_short_anchor_terms', 48), product_wording('prompt_contract', '_extract_short_anchor_terms', 49), "NPC", product_wording('prompt_contract', '_extract_short_anchor_terms', 50)]:
        if term in prompt:
            anchors.append(term)
    for term in [product_wording('prompt_contract', '_extract_short_anchor_terms', 51), product_wording('prompt_contract', '_extract_short_anchor_terms', 52), product_wording('prompt_contract', '_extract_short_anchor_terms', 53), product_wording('prompt_contract', '_extract_short_anchor_terms', 54)]:
        if term in prompt:
            anchors.append(term)
    return _unique_preserve_order(anchors)


def _build_forbidden_drift_topics(prompt: str, topic_terms: list[str], groups: dict[str, list[str]]) -> list[str]:
    forbidden = []
    prompt_terms = f"{prompt} {' '.join(topic_terms)}"
    for group_name, terms in groups.items():
        if group_name != "format_drift" and _group_is_authorized(prompt_terms, terms):
            continue
        forbidden.extend(terms)
    return _unique_preserve_order(forbidden)


def _group_is_authorized(prompt_terms: str, terms: list[str]) -> bool:
    return any(_contains_term(prompt_terms, term) for term in terms if not term.startswith((product_wording('prompt_contract', '_group_is_authorized', 0), product_wording('prompt_contract', '_group_is_authorized', 1))))


def _infer_length_hint(prompt: str, prompt_type: str) -> str:
    if prompt_type in {"keyword", "short_daily", "question"}:
        return "short"
    if prompt_type == "opening" and len(prompt.strip()) <= SHORT_PROMPT_MAX_CHARS:
        return "short"
    if prompt_type == "opening":
        return "medium"
    if len(prompt) > 60:
        return "medium-long"
    return "medium"


def _infer_risk_tags(prompt: str, prompt_type: str, topic_terms: list[str]) -> list[str]:
    tags = []
    if prompt_type in {"keyword", "short_daily"} or (prompt_type == "opening" and len(prompt.strip()) <= SHORT_PROMPT_MAX_CHARS):
        tags.append("short_input_expansion_risk")
    if prompt_type == "opening" and len(prompt.strip()) <= SHORT_PROMPT_MAX_CHARS:
        tags.append("fragment_opening_expansion_risk")
    if re.search(r"[A-Za-z]{2,}", prompt):
        tags.append("multilingual_trigger")
    if any(_contains_term(prompt, term) for term in [product_wording('prompt_contract', '_infer_risk_tags', 1), product_wording('prompt_contract', '_infer_risk_tags', 2)]):
        tags.append("romance_or_adult_style")
    if _contains_term(prompt, product_wording('prompt_contract', '_infer_risk_tags', 0)):
        tags.append("adult_content_boundary")
    if any(_contains_term(prompt, term) for term in [product_wording('prompt_contract', '_infer_risk_tags', 3), product_wording('prompt_contract', '_infer_risk_tags', 4), product_wording('prompt_contract', '_infer_risk_tags', 5), product_wording('prompt_contract', '_infer_risk_tags', 6)]):
        tags.append("political_fiction")
    if any(_contains_term(prompt, term) for term in [product_wording('prompt_contract', '_infer_risk_tags', 7), product_wording('prompt_contract', '_infer_risk_tags', 8)]):
        tags.append("fictional_setup")
    if any(term.lower() in {"cat", "kaka", "sq", "nus"} for term in topic_terms):
        tags.append("named_keyword")
    if _has_humor_cue(prompt) or any(term in {product_wording('prompt_contract', '_infer_risk_tags', 9), product_wording('prompt_contract', '_infer_risk_tags', 10), product_wording('prompt_contract', '_infer_risk_tags', 11), product_wording('prompt_contract', '_infer_risk_tags', 12), product_wording('prompt_contract', '_infer_risk_tags', 13), product_wording('prompt_contract', '_infer_risk_tags', 14), product_wording('prompt_contract', '_infer_risk_tags', 15), product_wording('prompt_contract', '_infer_risk_tags', 16), product_wording('prompt_contract', '_infer_risk_tags', 17)} for term in topic_terms):
        tags.append("humor_or_absurd_prompt")
    return _unique_preserve_order(tags)


def _has_humor_cue(prompt: str) -> bool:
    return any(cue.lower() in prompt.lower() for cue in product_value('prompt_contract', 'HUMOR_CUE_WORDS'))


def _looks_like_keyword(prompt: str) -> bool:
    if re.search(r"[，。！？,.!?；;：:\s]", prompt):
        return False
    compact = _compact(prompt)
    if compact.lower() in product_value('prompt_contract', 'KEYWORD_ALIASES'):
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,10}", compact):
        return True
    return len(compact) <= 5


def _looks_like_opening(prompt: str) -> bool:
    if len(prompt) > 40:
        return False
    if prompt.endswith(("，", "：", ":")):
        return True
    if prompt.startswith((product_wording('prompt_contract', '_looks_like_opening', 0), product_wording('prompt_contract', '_looks_like_opening', 1), product_wording('prompt_contract', '_looks_like_opening', 2), product_wording('prompt_contract', '_looks_like_opening', 3))) and not prompt.endswith(("。", "！", "？", ".", "!", "?")):
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
    term = re.sub(r"^(我在|我|在)", "", term)
    term = re.sub(r"(日记|片段|记录|生活)$", "", term)
    term = re.sub(r"(待着|待一会|待了会|待)$", "", term)
    return term.strip(" -_[]【】()（）“”\"'‘’。，,；;：:!?！？")


def _is_valid_topic_term(term: str) -> bool:
    if not term or term in set(product_value('prompt_contract', 'STOPWORDS')):
        return False
    if any(marker in term for marker in ["相关的当下", "当下感受", "回应近况"]):
        return False
    if len(term) > 18:
        return False
    if re.fullmatch(r"\d+", term):
        return False
    return True


def _is_valid_style_hint(term: str) -> bool:
    if not term or term in set(product_value('prompt_contract', 'STOPWORDS')):
        return False
    if term in product_value('prompt_contract', 'STYLE_KEYWORDS'):
        return True
    if any(keyword in term for keyword in product_value('prompt_contract', 'STYLE_KEYWORDS')):
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
    from diary_core.config.assets import add_asset_root, product_root
    from diary_core.infer.product_content import load_product_content, using_product_content
    from lm_trainer.config.common import asset_path

    parser = argparse.ArgumentParser(description="Analyze product prompt structure")
    add_asset_root(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="Analyze a single synthetic prompt")
    source.add_argument("--from-output-md", help="Read an output file under runs/")
    args = parser.parse_args()
    root = product_root(args.asset_root)
    prompts = [args.prompt] if args.prompt else extract_prompts_from_output_md(
        asset_path(root, args.from_output_md, "--from-output-md", "runs")
    )
    with using_product_content(load_product_content(root)):
        contracts = [build_contract(prompt) for prompt in prompts]
    print(json.dumps({"count": len(contracts), "types": [contract.prompt_type for contract in contracts]}))


if __name__ == "__main__":
    main()
