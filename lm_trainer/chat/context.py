"""Render and budget native prompts without changing the session history."""
# ruff: noqa: BLE001

from __future__ import annotations

import re
from dataclasses import dataclass

from lm_trainer.template_policy import PROTOCOL_MARKERS, chat_template_kwargs

from .memory import (
    REFERENCE_CLOSE,
    REFERENCE_OPEN,
    ContextCard,
    audit,
    reference_section,
)
from .types import CardAudit, ChatError, Message, RoleConfig

# Additional non-chat protocols are fail-closed, never interpreted by this runtime.
HIDDEN_PROTOCOL = (
    *PROTOCOL_MARKERS,
    "<analysis>",
    "</analysis>",
    "<reasoning>",
    "</reasoning>",
    "<tool_use>",
    "</tool_use>",
    "<function_call>",
    "</function_call>",
    "<|recipient|>",
)
PROTOCOL = (
    *HIDDEN_PROTOCOL,
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|start|>",
    "<|end|>",
)


@dataclass(frozen=True, slots=True)
class BuiltContext:
    input_ids: list[int]
    card_audit: tuple[CardAudit, ...]
    dropped_turns: int


class ContextBuilder:
    def __init__(self, *, max_cards: int = 3, card_token_limit: int = 400):
        if any(type(x) is not int or x < 0 for x in (max_cards, card_token_limit)):
            raise ChatError("INVALID_CONFIG")
        self.max_cards = max_cards
        self.card_token_limit = card_token_limit

    @staticmethod
    def render(tokenizer, policy, messages, *, for_training=False):
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": policy["role_map"][r], "content": c} for r, c in messages],
                tokenize=False,
                **chat_template_kwargs(policy, for_training=for_training),
            )
            if not isinstance(rendered, str) or not rendered:
                raise ValueError
            return rendered
        except Exception:
            raise ChatError("TEMPLATE_UNSUPPORTED") from None

    @classmethod
    def verify_template(cls, tokenizer, template, policy):
        # Require audited template bytes plus a plain assistant prefix. This is not a
        # sandbox for arbitrary Jinja templates, tool modes or reasoning grammars.
        if policy["enable_thinking"] is not False and re.search(
            r"enable_thinking|<think>|<analysis>|<reasoning>", template, re.IGNORECASE
        ):
            raise ChatError("TEMPLATE_UNSUPPORTED")
        messages = [
            ("system", "ChatProbeSystem"),
            ("user", "ChatProbeQuestion"),
            ("assistant", "ChatProbeAnswer"),
            ("user", "ChatProbeFollowup"),
        ]
        before = cls.render(tokenizer, policy, messages, for_training=True)
        after = cls.render(tokenizer, policy, messages)
        suffix = after[len(before) :] if after.startswith(before) else ""
        visible_suffix = suffix
        if policy["enable_thinking"] is False:
            # Some native templates prefill a *closed, empty* think block in non-thinking mode.
            visible_suffix = re.sub(r"<think>\s*</think>(?=\s*$)", "", suffix)
        if (
            not visible_suffix.strip()
            or any(p.lower() in visible_suffix.lower() for p in HIDDEN_PROTOCOL)
            or any(p in suffix for p in policy["stop_tokens"])
            or any(before.count(text) != 1 for _, text in messages)
            or sorted(before.index(text) for _, text in messages)
            != [before.index(text) for _, text in messages]
        ):
            raise ChatError("TEMPLATE_UNSUPPORTED")

    @classmethod
    def encode(cls, tokenizer, policy, messages):
        rendered = cls.render(tokenizer, policy, messages)
        try:
            ids = tokenizer(
                rendered, add_special_tokens=False, truncation=False, verbose=False
            )["input_ids"]
            if not ids:
                raise ValueError
            return ids
        except Exception:
            raise ChatError("TEMPLATE_UNSUPPORTED") from None

    def build(
        self,
        tokenizer,
        policy,
        *,
        role: RoleConfig,
        history: tuple[Message, ...],
        user_text: str,
        cards: tuple[ContextCard, ...],
        audit_entries: tuple[CardAudit, ...],
        context_limit: int,
        max_new_tokens: int,
    ) -> BuiltContext:
        markers = {
            p.lower()
            for p in (
                *policy["control_tokens"],
                *PROTOCOL,
                REFERENCE_OPEN,
                REFERENCE_CLOSE,
            )
        }
        control_ids = set(policy["control_token_ids"])

        def safe(text):
            try:
                return not (
                    any(p in text.lower() for p in markers)
                    or control_ids.intersection(
                        tokenizer(text, add_special_tokens=False, verbose=False)[
                            "input_ids"
                        ]
                    )
                )
            except Exception:
                return False

        if not safe(role.system_prompt):
            raise ChatError("TEMPLATE_UNSUPPORTED")
        if not safe(user_text):
            raise ChatError("UNSAFE_CONTENT")
        limit = context_limit - max_new_tokens
        current = ("user", user_text)
        system = role.system_prompt
        base = self.encode(tokenizer, policy, [("system", system), current])
        if len(base) > limit:
            raise ChatError("CONTEXT_LIMIT")
        accepted = []
        entries = list(audit_entries)
        ids = base
        for card in cards:
            if not all(
                map(
                    safe,
                    (*card.facts, card.event_time, card.recorded_at, card.uncertainty),
                )
            ):
                entries.append(audit(card, "reserved_content"))
                continue
            if len(accepted) >= self.max_cards:
                entries.append(audit(card, "card_count"))
                continue
            candidate = role.system_prompt + reference_section([*accepted, card])
            if not safe(
                candidate.replace(REFERENCE_OPEN, "").replace(REFERENCE_CLOSE, "")
            ):
                entries.append(audit(card, "reserved_content"))
                continue
            candidate_ids = self.encode(
                tokenizer, policy, [("system", candidate), current]
            )
            if len(candidate_ids) - len(base) > self.card_token_limit:
                entries.append(audit(card, "card_tokens"))
                continue
            if len(candidate_ids) > limit:
                entries.append(audit(card, "context_budget"))
                continue
            # A trusted, audited template still must actually include its reference block.
            if candidate not in self.render(
                tokenizer, policy, [("system", candidate), current]
            ):
                raise ChatError("TEMPLATE_UNSUPPORTED")
            entries.append(audit(card, "injected"))
            accepted.append(card)
            system, ids = candidate, candidate_ids

        if len(history) % 2 or any(
            m.role != ("user" if i % 2 == 0 else "assistant")
            for i, m in enumerate(history)
        ):
            raise ChatError("INVALID_INPUT")
        retained = 0
        for i in range(len(history) - 2, -1, -2):
            candidate_ids = self.encode(
                tokenizer,
                policy,
                [
                    ("system", system),
                    *((m.role, m.content) for m in history[i:]),
                    current,
                ],
            )
            if len(candidate_ids) > limit:
                break
            retained += 1
            ids = candidate_ids
        return BuiltContext(ids, tuple(entries), len(history) // 2 - retained)
