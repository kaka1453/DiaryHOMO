"""Render and budget native prompts without changing the session history."""
# ruff: noqa: BLE001

from __future__ import annotations

import re
from contextlib import nullcontext
from dataclasses import dataclass

from lm_trainer.conversation import ConversationError
from lm_trainer.conversation.renderer import check_visible_text, render_native
from lm_trainer.template_policy import PROTOCOL_MARKERS
from lm_trainer.train.encoding import (
    EncodingError,
    conversation_identity,
    encode_conversation_prefix,
)

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
INFERENCE_CONTROLS = (*PROTOCOL, REFERENCE_OPEN, REFERENCE_CLOSE)


def check_inference_text(text, *, tokenizer=None, policy=None, application=False):
    controls = () if policy is None else policy["control_tokens"]
    try:
        check_visible_text(
            text, (*INFERENCE_CONTROLS, *controls), application=application
        )
        if tokenizer is not None and set(policy["control_token_ids"]).intersection(
            tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
        ):
            raise ConversationError("control IDs")
    except Exception:
        raise ChatError(
            "TEMPLATE_UNSUPPORTED" if application else "UNSAFE_CONTENT"
        ) from None


@dataclass(frozen=True, slots=True)
class BuiltContext:
    input_ids: list[int]
    card_audit: tuple[CardAudit, ...]
    dropped_turns: int
    conversation_renderer: dict[str, str] | None = None
    injected_tokens: int = 0


class ContextBuilder:
    def __init__(self, *, max_cards: int = 3, card_token_limit: int = 400):
        if any(type(x) is not int or x < 0 for x in (max_cards, card_token_limit)):
            raise ChatError("INVALID_CONFIG")
        self.max_cards = max_cards
        self.card_token_limit = card_token_limit

    @staticmethod
    def render(tokenizer, policy, messages, *, for_training=False):
        try:
            rendered = render_native(
                tokenizer,
                policy,
                [{"role": policy["role_map"][r], "content": c} for r, c in messages],
                for_training=for_training,
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
        return cls._encode_rendered(tokenizer, cls.render(tokenizer, policy, messages))

    @staticmethod
    def _encode_rendered(tokenizer, rendered):
        try:
            ids = tokenizer(
                rendered, add_special_tokens=False, truncation=False, verbose=False
            )["input_ids"]
            if not ids:
                raise ValueError
            return ids
        except Exception:
            raise ChatError("TEMPLATE_UNSUPPORTED") from None

    @staticmethod
    def build_conversation(
        tokenizer,
        policy,
        observation,
        *,
        target_person,
        context_limit,
        max_new_tokens,
    ) -> BuiltContext:
        """Stateless direct conversation path; no synthetic user or injected cards."""
        if (
            type(context_limit) is not int
            or type(max_new_tokens) is not int
            or context_limit < 1
            or max_new_tokens < 1
        ):
            raise ChatError("INVALID_INPUT")
        try:
            ids = encode_conversation_prefix(
                tokenizer, policy, observation, target_person
            )
        except ConversationError as exc:
            code = (
                "UNSAFE_CONTENT"
                if str(exc).startswith("reserved control")
                else "INVALID_INPUT"
            )
            raise ChatError(code) from None
        except EncodingError:
            raise ChatError("TEMPLATE_UNSUPPORTED") from None
        native_limit = getattr(tokenizer, "model_max_length", None)
        limit = (
            min(context_limit, native_limit)
            if type(native_limit) is int
            else context_limit
        )
        if len(ids) + max_new_tokens > limit:
            raise ChatError("CONTEXT_LIMIT")
        return BuiltContext(ids, (), 0, conversation_identity(policy))

    def build_inference_v1(
        self,
        tokenizer,
        policy,
        *,
        role,
        messages,
        observation,
        target_person,
        cards,
        audit_entries,
        context_limit,
        max_new_tokens,
        card_guard=None,
    ) -> BuiltContext:
        """Budget complete explicit input first. Only optional references may be omitted."""
        if (messages is None) == (observation is None):
            raise ChatError("INVALID_INPUT")
        if any(type(n) is not int or n < 1 for n in (context_limit, max_new_tokens)):
            raise ChatError("INVALID_INPUT")
        if policy["system_prompt"] not in (None, role.system_prompt):
            raise ChatError("TEMPLATE_UNSUPPORTED")
        native_limit = getattr(tokenizer, "model_max_length", None)
        limit = (
            min(context_limit, native_limit)
            if type(native_limit) is int
            else context_limit
        )

        def safe(text, *, application=False):
            check_inference_text(
                text, tokenizer=tokenizer, policy=policy, application=application
            )

        safe(role.system_prompt, application=True)
        for text in (
            (event.content for event in observation.events if event.selection == "text")
            if observation is not None
            else (text for _, text in messages)
        ):
            safe(text)

        def encode(system):
            if observation is not None:
                try:
                    return encode_conversation_prefix(
                        tokenizer,
                        {**policy, "system_prompt": system},
                        observation,
                        target_person,
                    )
                except ConversationError as exc:
                    code = (
                        "UNSAFE_CONTENT"
                        if str(exc).startswith("reserved control")
                        else "INVALID_INPUT"
                    )
                    raise ChatError(code) from None
                except EncodingError:
                    raise ChatError("TEMPLATE_UNSUPPORTED") from None
            rendered = self.render(tokenizer, policy, [("system", system), *messages])
            if rendered.count(system) < 1 or (
                REFERENCE_OPEN in system
                and (
                    rendered.count(REFERENCE_OPEN) != 1
                    or rendered.count(REFERENCE_CLOSE) != 1
                )
            ):
                raise ChatError("TEMPLATE_UNSUPPORTED")
            return self._encode_rendered(tokenizer, rendered)

        base = ids = encode(role.system_prompt)
        if len(base) + max_new_tokens > limit:
            raise ChatError("CONTEXT_LIMIT")
        guard = (
            card_guard(cards)
            if cards and card_guard is not None
            else nullcontext((cards, ()))
        )
        with guard as (approved, denied):
            accepted, entries = [], [*audit_entries, *denied]
            for card in approved:
                if len(accepted) >= self.max_cards:
                    entries.append(audit(card, "card_count"))
                    continue
                try:
                    for text in (
                        *card.facts,
                        card.event_time,
                        card.recorded_at,
                        card.uncertainty,
                    ):
                        safe(text)
                    system = role.system_prompt + reference_section([*accepted, card])
                    safe(
                        system.replace(REFERENCE_OPEN, "").replace(REFERENCE_CLOSE, ""),
                        application=True,
                    )
                except ChatError:
                    entries.append(audit(card, "reserved_content"))
                    continue
                try:
                    candidate = encode(system)
                except ChatError:
                    # The required prefix already passed proof; an optional card may
                    # introduce a native boundary the tokenizer cannot align.
                    entries.append(audit(card, "card_template"))
                    continue
                if len(candidate) - len(base) > self.card_token_limit:
                    reason = "card_tokens"
                elif len(candidate) + max_new_tokens > limit:
                    reason = "context_budget"
                else:
                    accepted.append(card)
                    ids, reason = candidate, "injected"
                entries.append(audit(card, reason))
            return BuiltContext(
                ids,
                tuple(entries),
                0,
                conversation_identity(policy) if observation is not None else None,
                max(0, len(ids) - len(base)),
            )

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
