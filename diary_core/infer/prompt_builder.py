from __future__ import annotations

from diary_core.infer.product_content import product_value, product_wording

from diary_core.infer.prompt_contract import DiaryContract


def build_messages(contract: DiaryContract, attachments: dict | None = None, config: dict | None = None) -> list[dict]:
    config = config or {}
    attachments = attachments or {}
    system_prompt = config.get("system_prompt") or product_value('prompt_builder', 'DEFAULT_SYSTEM_PROMPT')
    user_parts = [
        "[RAW_PROMPT]",
        contract.raw_prompt.strip(),
        "",
        "[MAIN_TOPIC]",
        contract.main_topic,
        "",
        "[TOPIC_TERMS]",
        _format_list(getattr(contract, "topic_terms", [])),
        "",
        "[MUST_INCLUDE_TERMS]",
        _format_list(_must_include_terms(contract)),
        "",
        "[SCENE_BEATS]",
        _format_list(_scene_beats(contract)),
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
        _format_list(select_prompt_forbidden_topics(contract)),
        "",
        "[FORMAT_FORBIDDEN]",
        _format_list(getattr(contract, "format_forbidden", [])),
        "",
        "[STYLE_EXECUTION]",
        _style_execution_rule(contract),
        "",
        "[ANTI_GENERIC_RULE]",
        _anti_generic_rule(contract),
        "",
        "[SELF_CHECK_BEFORE_OUTPUT]",
        _self_check_rule(contract),
        "",
        "[DRIFT_AVOIDANCE_RULE]",
        _drift_avoidance_rule(contract),
        "",
        "[OLD_MEMORY_BAN]",
        _old_memory_ban_rule(contract),
        "",
        "[FACT_BOUNDARY_HINT]",
        _fact_boundary_hint(contract),
        "",
    ]
    attachment_blocks = format_attachment_blocks(attachments)
    if attachment_blocks:
        user_parts.extend(["[PROFILE_AND_CONTEXT_ATTACHMENTS]", attachment_blocks, ""])
    user_parts.extend(
        [
            "[OUTPUT]",
            product_wording('prompt_builder', 'build_messages', 0),
        ]
    )
    user_content = "\n".join(user_parts)
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


def format_attachment_blocks(attachments: dict) -> str:
    if not attachments or not attachments.get("items"):
        return ""
    profile_lines = []
    context_lines = []
    for item in attachments["items"]:
        if item.get("type") == "diary_profile":
            title = item.get("title") or item.get("section") or "PROFILE_ATTACHMENT"
            render_text = str(item.get("render_text") or "").strip()
            if render_text:
                profile_lines.append(f"[{title}]\n{render_text}")
            continue
        item_type = item.get("type", "context")
        content = item.get("content")
        if isinstance(content, list):
            content_text = "；".join(str(value) for value in content[-6:])
        else:
            content_text = str(content or "").strip()
        if content_text:
            context_lines.append(f"- {item_type}: {content_text}")
    blocks = []
    blocks.extend(profile_lines)
    if context_lines:
        blocks.append("[CONTEXT_ATTACHMENTS]\n" + "\n".join(context_lines))
    return "\n\n".join(blocks)


def _format_attachments(attachments: dict) -> str:
    return format_attachment_blocks(attachments)


def select_prompt_forbidden_topics(contract: DiaryContract, max_items: int = 24) -> list[str]:
    topic_text = _compact(" ".join([contract.raw_prompt, contract.main_topic, " ".join(getattr(contract, "topic_terms", []))]))
    groups = product_value('prompt_builder', "FORBIDDEN_TOPIC_GROUPS")
    selected = []
    for label, keywords in groups:
        if any(_contains_compact(topic_text, keyword) for keyword in keywords):
            continue
        selected.append(label)

    explicit = [
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 0),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 1),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 2),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 3),
        "NUS",
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 4),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 5),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 6),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 7),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 8),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 9),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 10),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 11),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 12),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 13),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 14),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 15),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 16),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 17),
        product_wording('prompt_builder', 'select_prompt_forbidden_topics', 18),
    ]
    for item in explicit:
        if len(selected) >= max_items:
            break
        if _contains_compact(topic_text, item):
            continue
        if item in getattr(contract, "forbidden_drift_topics", []) and item not in selected:
            selected.append(item)
    return selected[:max_items]


def _fact_boundary_hint(contract: DiaryContract) -> str:
    fact_types = getattr(contract, "forbidden_fact_types", [])
    weak_hint = (
        product_wording('prompt_builder', '_fact_boundary_hint', 0)
    )
    if not fact_types:
        return weak_hint
    return f"{weak_hint}{product_wording('prompt_builder', '_fact_boundary_hint', 1)}{_join_inline(fact_types[:4])}{product_wording('prompt_builder', '_fact_boundary_hint', 2)}"


def _style_execution_rule(contract: DiaryContract) -> str:
    styles = set(getattr(contract, "style_hints", []))
    risk_tags = set(getattr(contract, "risk_tags", []))
    humor_styles = {product_wording('prompt_builder', '_style_execution_rule', 1), product_wording('prompt_builder', '_style_execution_rule', 2), product_wording('prompt_builder', '_style_execution_rule', 3), product_wording('prompt_builder', '_style_execution_rule', 4), product_wording('prompt_builder', '_style_execution_rule', 5), product_wording('prompt_builder', '_style_execution_rule', 6), product_wording('prompt_builder', '_style_execution_rule', 7), product_wording('prompt_builder', '_style_execution_rule', 8), product_wording('prompt_builder', '_style_execution_rule', 9), product_wording('prompt_builder', '_style_execution_rule', 10), product_wording('prompt_builder', '_style_execution_rule', 11), product_wording('prompt_builder', '_style_execution_rule', 12), product_wording('prompt_builder', '_style_execution_rule', 13)}
    if styles & humor_styles or "humor_or_absurd_prompt" in risk_tags:
        return "\n".join(
            [
                product_wording('prompt_builder', '_style_execution_rule', 15),
                product_wording('prompt_builder', '_style_execution_rule', 16),
                product_wording('prompt_builder', '_style_execution_rule', 17),
                product_wording('prompt_builder', '_style_execution_rule', 18),
                product_wording('prompt_builder', '_style_execution_rule', 19),
                product_wording('prompt_builder', '_style_execution_rule', 20),
                product_wording('prompt_builder', '_style_execution_rule', 21),
                product_wording('prompt_builder', '_style_execution_rule', 22),
                product_wording('prompt_builder', '_style_execution_rule', 23),
            ]
        )
    if getattr(contract, "length_hint", "") == "short":
        return product_wording('prompt_builder', '_style_execution_rule', 14)
    return product_wording('prompt_builder', '_style_execution_rule', 0)


def _anti_generic_rule(contract: DiaryContract) -> str:
    return "\n".join(
        [
            product_wording('prompt_builder', '_anti_generic_rule', 0),
            product_wording('prompt_builder', '_anti_generic_rule', 1),
            product_wording('prompt_builder', '_anti_generic_rule', 2),
            product_wording('prompt_builder', '_anti_generic_rule', 3),
            product_wording('prompt_builder', '_anti_generic_rule', 4),
        ]
    )


def _drift_avoidance_rule(contract: DiaryContract) -> str:
    lines = [
        product_wording('prompt_builder', '_drift_avoidance_rule', 0),
        product_wording('prompt_builder', '_drift_avoidance_rule', 1),
        product_wording('prompt_builder', '_drift_avoidance_rule', 2),
        product_wording('prompt_builder', '_drift_avoidance_rule', 3),
    ]
    if getattr(contract, "length_hint", "") == "short":
        lines.append(product_wording('prompt_builder', '_drift_avoidance_rule', 4))
    return "\n".join(lines)


def _self_check_rule(contract: DiaryContract) -> str:
    terms = _join_inline(getattr(contract, "topic_terms", [])[:6]) or contract.main_topic
    must_terms = _join_inline(_must_include_terms(contract))
    return "\n".join(
        [
            f"{product_wording('prompt_builder', '_self_check_rule', 0)}{terms}。",
            f"{product_wording('prompt_builder', '_self_check_rule', 1)}{must_terms or product_wording('prompt_builder', '_self_check_rule', 2)}{product_wording('prompt_builder', '_self_check_rule', 3)}",
            product_wording('prompt_builder', '_self_check_rule', 4),
            product_wording('prompt_builder', '_self_check_rule', 5),
        ]
    )


def _old_memory_ban_rule(contract: DiaryContract) -> str:
    topic_text = _compact(" ".join([contract.raw_prompt, contract.main_topic, " ".join(getattr(contract, "topic_terms", []))]))
    authorized_terms = _authorized_old_memory_aliases(topic_text)
    forbidden = [
        term
        for term in product_value('prompt_builder', 'OLD_MEMORY_FORBIDDEN_TERMS')
        if term not in authorized_terms and not _contains_compact(topic_text, term)
    ]
    if not forbidden:
        return product_wording('prompt_builder', '_old_memory_ban_rule', 2)
    return (
        f"{product_wording('prompt_builder', '_old_memory_ban_rule', 0)}{_join_inline(forbidden[:20])}{product_wording('prompt_builder', '_old_memory_ban_rule', 1)}"
    )


def _authorized_old_memory_aliases(topic_text: str) -> set[str]:
    authorized = set()
    if _contains_compact(topic_text, product_wording('prompt_builder', '_authorized_old_memory_aliases', 0)) or _contains_compact(topic_text, product_wording('prompt_builder', '_authorized_old_memory_aliases', 1)):
        authorized.add(product_wording('prompt_builder', '_authorized_old_memory_aliases', 2))
    return authorized


def _must_include_terms(contract: DiaryContract) -> list[str]:
    terms = [term for term in getattr(contract, "topic_terms", []) if 1 <= len(term) <= 12]
    if not terms:
        return []
    return terms[:4]


def _scene_beats(contract: DiaryContract) -> list[str]:
    prompt = contract.raw_prompt.strip()
    terms = [term for term in getattr(contract, "topic_terms", []) if term]
    if getattr(contract, "length_hint", "") == "short":
        if _contains_compact(prompt, "cat"):
            return [
                product_wording('prompt_builder', '_scene_beats', 18),
                product_wording('prompt_builder', '_scene_beats', 19),
                product_wording('prompt_builder', '_scene_beats', 20),
            ]
        if _contains_compact(prompt, product_wording('prompt_builder', '_scene_beats', 5)):
            return [
                product_wording('prompt_builder', '_scene_beats', 21),
                product_wording('prompt_builder', '_scene_beats', 22),
                product_wording('prompt_builder', '_scene_beats', 23),
            ]
        if _contains_compact(prompt, product_wording('prompt_builder', '_scene_beats', 24)) and _contains_compact(prompt, product_wording('prompt_builder', '_scene_beats', 25)):
            return [
                product_wording('prompt_builder', '_scene_beats', 26),
                product_wording('prompt_builder', '_scene_beats', 27),
                product_wording('prompt_builder', '_scene_beats', 28),
            ]
        return [
            f"{product_wording('prompt_builder', '_scene_beats', 6)}{contract.main_topic}{product_wording('prompt_builder', '_scene_beats', 7)}",
            f"{product_wording('prompt_builder', '_scene_beats', 8)}{_join_inline(terms[:5]) or product_wording('prompt_builder', '_scene_beats', 9)}。",
            product_wording('prompt_builder', '_scene_beats', 10),
        ]

    humor = "humor_or_absurd_prompt" in set(getattr(contract, "risk_tags", []))
    clauses = _prompt_clauses(prompt)
    if humor:
        if _contains_compact(prompt, product_wording('prompt_builder', '_scene_beats', 29)) and _contains_compact(prompt, product_wording('prompt_builder', '_scene_beats', 30)):
            return [
                product_wording('prompt_builder', '_scene_beats', 31),
                product_wording('prompt_builder', '_scene_beats', 32),
                product_wording('prompt_builder', '_scene_beats', 33),
            ]
        if _contains_compact(prompt, product_wording('prompt_builder', '_scene_beats', 34)) and _contains_compact(prompt, product_wording('prompt_builder', '_scene_beats', 35)):
            return [
                product_wording('prompt_builder', '_scene_beats', 36),
                product_wording('prompt_builder', '_scene_beats', 37),
                product_wording('prompt_builder', '_scene_beats', 38),
            ]
        if _contains_compact(prompt, product_wording('prompt_builder', '_scene_beats', 39)) and _contains_compact(prompt, product_wording('prompt_builder', '_scene_beats', 40)):
            return [
                product_wording('prompt_builder', '_scene_beats', 41),
                product_wording('prompt_builder', '_scene_beats', 42),
                product_wording('prompt_builder', '_scene_beats', 43),
            ]
        if _contains_compact(prompt, product_wording('prompt_builder', '_scene_beats', 11)):
            return [
                product_wording('prompt_builder', '_scene_beats', 44),
                product_wording('prompt_builder', '_scene_beats', 45),
                product_wording('prompt_builder', '_scene_beats', 46),
            ]
        return [
            f"{product_wording('prompt_builder', '_scene_beats', 12)}{(clauses[0] if clauses else contract.main_topic)}。",
            f"{product_wording('prompt_builder', '_scene_beats', 13)}{_join_inline(terms[:6]) or contract.main_topic}。",
            product_wording('prompt_builder', '_scene_beats', 14),
        ]

    if clauses:
        if _contains_compact(prompt, product_wording('prompt_builder', '_scene_beats', 47)) and _contains_compact(prompt, product_wording('prompt_builder', '_scene_beats', 48)):
            return [
                product_wording('prompt_builder', '_scene_beats', 49),
                product_wording('prompt_builder', '_scene_beats', 50),
                product_wording('prompt_builder', '_scene_beats', 51),
            ]
        beats = [f"{product_wording('prompt_builder', '_scene_beats', 15)}{clauses[0]}{product_wording('prompt_builder', '_scene_beats', 16)}"]
        if len(clauses) > 1:
            beats.append(f"{product_wording('prompt_builder', '_scene_beats', 52)}{clauses[1]}{product_wording('prompt_builder', '_scene_beats', 53)}")
        else:
            beats.append(f"{product_wording('prompt_builder', '_scene_beats', 54)}{_join_inline(terms[:5]) or contract.main_topic}{product_wording('prompt_builder', '_scene_beats', 55)}")
        beats.append(product_wording('prompt_builder', '_scene_beats', 17))
        return beats

    return [
        f"{product_wording('prompt_builder', '_scene_beats', 0)}{contract.main_topic}{product_wording('prompt_builder', '_scene_beats', 1)}",
        f"{product_wording('prompt_builder', '_scene_beats', 2)}{_join_inline(terms[:6]) or product_wording('prompt_builder', '_scene_beats', 3)}。",
        product_wording('prompt_builder', '_scene_beats', 4),
    ]


def _join_inline(items: list[str]) -> str:
    return "、".join(items)


def _prompt_clauses(prompt: str) -> list[str]:
    cleaned = prompt.strip(" \n\t。.!！?？")
    cleaned = cleaned.replace("写一篇", "").replace("日记", "")
    parts = []
    for part in cleaned.replace("，", ",").replace("。", ",").replace("；", ",").split(","):
        item = part.strip(" “”，,。.!！?？")
        if item and len(item) <= 42:
            parts.append(item)
    return parts[:3]


def _compact(text: str) -> str:
    return "".join(str(text).split()).lower()


def _contains_compact(text: str, keyword: str) -> bool:
    return _compact(keyword) in text


def _fallback_render_messages(messages: list[dict]) -> str:
    chunks = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        chunks.append(f"[{role}]\n{content}")
    chunks.append("[ASSISTANT]\n")
    return "\n\n".join(chunks)
