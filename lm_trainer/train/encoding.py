"""Encode one independent document or prove native-template target token spans."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from lm_trainer.conversation import (
    RENDERER_VERSION,
    ConversationError,
    render_observation,
)
from lm_trainer.conversation.renderer import (
    check_visible_text,
    render_native,
    target_turn,
)
from lm_trainer.template_policy import PROTOCOL_MARKERS, ROLE_MAP

IGNORE = -100


class EncodingError(ValueError):
    """Static diagnostic safe to include without leaking tokenizer inputs."""


@dataclass(frozen=True, slots=True)
class TokenSpan:
    start: int
    end: int
    kind: str
    source_id: str | None = None


@dataclass(frozen=True)
class Encoded:
    input_ids: list[int]
    labels: list[int]
    roles: list[str]
    body_targets: int
    eos_targets: int = 0
    truncated: bool = False
    spans: tuple[TokenSpan, ...] = ()
    prefix_ids: tuple[int, ...] = ()
    focus_ids: tuple[str, ...] = ()
    renderer_version: str | None = None
    template_hash: str | None = None
    stop_hash: str | None = None


def _ids(tokenizer, text):
    return list(
        tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    )


def encode_text(tokenizer, text, plan):
    limit = plan["max_length"]
    if plan["objective"] == "legacy":
        raw = list(tokenizer(text, truncation=False)["input_ids"])
        ids = (
            list(tokenizer(text, truncation=True, max_length=limit)["input_ids"])
            if len(raw) > limit
            else raw
        )
        if len(ids) < 2:
            raise EncodingError("legacy sample has no next-token targets")
        # Legacy uses the tokenizer's automatic specials and truncation.
        return Encoded(
            ids,
            ids.copy(),
            ["legacy"] * len(ids),
            len(ids) - 1,
            truncated=len(raw) > limit,
        )

    body = _ids(tokenizer, text)
    if not body:
        raise EncodingError("CPT tokenizer produced no document tokens")
    reserved = {tokenizer.bos_token_id, tokenizer.eos_token_id} - {None}
    if reserved.intersection(body):
        raise EncodingError("CPT text contains reserved document boundary tokens")
    prefix, suffix = [], []
    for key, dest in (("bos", prefix), ("eos", suffix)):
        if plan[f"add_{key}"]:
            token = getattr(tokenizer, f"{key}_token_id")
            if token is None:
                raise EncodingError(
                    f"CPT requested {key.upper()} but tokenizer has no such token"
                )
            dest.append(token)
    ids = prefix + body + suffix
    if len(ids) > limit:
        raise EncodingError("CPT document exceeds max_length; not truncated")
    body_targets = len(body) - (not prefix)
    if not (body_targets or suffix):
        raise EncodingError("CPT sample has no next-token targets")
    if not body_targets and not plan["allow_eos_only"]:
        raise EncodingError("CPT sample has only EOS supervision")
    return Encoded(
        ids,
        [IGNORE, *ids[1:]],
        ["bos"] * len(prefix) + ["cpt"] * len(body) + ["eos"] * len(suffix),
        body_targets,
        len(suffix),
    )


def _project_messages(row, policy):
    messages = [
        {"role": ROLE_MAP[m["role"]], "content": m["content"]} for m in row["messages"]
    ]
    mask = list(row["target_mask"])
    system = policy["system_prompt"]
    if messages[0]["role"] == "system":
        if system is None or messages[0]["content"] != system:
            raise EncodingError(
                "only the configured application system prompt is permitted"
            )
    elif system is not None:
        messages.insert(0, {"role": "system", "content": system})
        mask.insert(0, False)
    if any(
        control in m["content"]
        for m in messages
        for control in policy["control_tokens"]
    ):
        raise EncodingError("message contains a reserved template control token")
    return messages, mask


def _literal_spans(tokenizer, messages, rendered, kwargs):
    """Differential render: prove content slots by substitution, never search body text.

    Probe strings must occur once, in order. Replacing them with *all* original
    content must reconstruct the complete native render byte-for-byte. The
    skeleton therefore cannot depend on lost/hidden/trimmed message content.
    """
    prefix = "__lm_trainer_content_slot_"
    while prefix in rendered or any(prefix in m["content"] for m in messages):
        prefix += "_"
    markers = [f"{prefix}{i:08d}__" for i in range(len(messages))]
    probes = [
        {"role": m["role"], "content": marker} for m, marker in zip(messages, markers)
    ]
    probe = tokenizer.apply_chat_template(probes, tokenize=False, **kwargs)
    fragments, cursor = [], 0
    for marker in markers:
        if probe.count(marker) != 1 or (pos := probe.find(marker, cursor)) < cursor:
            raise EncodingError(
                "native template does not preserve ordered literal message slots"
            )
        fragments.append(probe[cursor:pos])
        cursor = pos + len(marker)
    fragments.append(probe[cursor:])
    check = (
        "".join(frag + m["content"] for frag, m in zip(fragments, messages))
        + fragments[-1]
    )
    if check != rendered:
        raise EncodingError(
            "native template transforms content or uses unprovable message spans"
        )
    for frag in fragments:
        # Some native templates emit an empty think prefix; only accept it with
        # explicit non-thinking selection, and never label it.
        visible = frag
        if kwargs.get("enable_thinking") is False:
            visible = re.sub(r"<think>\s*</think>", "", visible)
        if any(marker in visible for marker in PROTOCOL_MARKERS):
            raise EncodingError(
                "native template emitted unsupported reasoning or tool controls"
            )
    return fragments


def encode_sft(tokenizer, row, template, policy, plan):
    messages, mask = _project_messages(row, policy)
    kwargs = policy["training_template_kwargs"]
    # Even templates with a generation mask require exact per-message attribution;
    # official mask positives alone cannot identify history or separate stops.
    use_official = re.search(r"{%[-+]?\s*generation\s*[-+]?%}", template) is not None
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        native = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            **({"return_assistant_tokens_mask": True} if use_official else {}),
            **kwargs,
        )
        fragments = _literal_spans(tokenizer, messages, rendered, kwargs)
    except EncodingError:
        raise
    except Exception:  # noqa: BLE001 -- Jinja/tokenizer errors can contain raw input.
        raise EncodingError(
            "native template could not render or align the chat"
        ) from None
    full = list(native["input_ids"])
    if len(full) > plan["max_length"]:
        raise EncodingError("SFT context exceeds max_length; no target truncated")
    ids, labels, roles = [], [], []
    bodies, stops = 0, 0
    official = native.get("assistant_masks") if use_official else None
    if use_official and (official is None or len(official) != len(full)):
        raise EncodingError("native template assistant mask unavailable")

    control_ids = set(policy["control_token_ids"])

    def add(text, role, supervised=False, *, content=False):
        tokens = _ids(tokenizer, text) if text else []
        if content and any(token in control_ids for token in tokens):
            raise EncodingError(
                "message tokenizes to a reserved template control token"
            )
        ids.extend(tokens)
        roles.extend([role] * len(tokens))
        labels.extend(tokens if supervised else [IGNORE] * len(tokens))
        return len(tokens)

    add(fragments[0], "template")
    for i, (message, selected) in enumerate(zip(messages, mask)):
        role = message["role"]
        start = len(ids)
        count = add(
            message["content"],
            role if not selected else "assistant_target",
            selected,
            content=True,
        )
        if selected:
            if not start or not count:
                raise EncodingError("SFT target has no predictable visible body tokens")
            if official is not None and not all(official[start : start + count]):
                raise EncodingError(
                    "native assistant mask disagrees with target body span"
                )
            bodies += count
        suffix = fragments[i + 1]
        stop = next(
            (
                s
                for s in sorted(policy["stop_tokens"], key=len, reverse=True)
                if suffix.startswith(s)
            ),
            None,
        )
        if selected and stop is None:
            raise EncodingError(
                "native target turn lacks an immediate configured stop token"
            )
        if stop is not None:
            supervised = selected and plan["supervise_target_eos"]
            count = add(stop, f"{role}_stop", supervised)
            if supervised:
                stops += count
            suffix = suffix[len(stop) :]
        add(suffix, "template")
    if ids != full:
        raise EncodingError(
            "native token boundaries do not align with literal slots (possible cross-boundary merge)"
        )
    if not bodies:
        raise EncodingError("SFT sample has no next-token targets")
    return Encoded(full, labels, roles, bodies, stops)


def conversation_identity(policy):
    stops = list(zip(policy["stop_tokens"], policy["stop_token_ids"], strict=True))
    return {
        "renderer_version": RENDERER_VERSION,
        "template_hash": policy["template_hash"],
        "stop_hash": hashlib.sha256(
            json.dumps(stops, ensure_ascii=False).encode()
        ).hexdigest(),
    }


def _conversation_layout(tokenizer, policy, turns, *, training, official=False):
    """Prove literal slots, then prove their decomposition against all native IDs."""
    messages = [turn.message() for turn in turns]
    kwargs = policy[
        "training_template_kwargs" if training else "inference_template_kwargs"
    ]
    try:
        for message, turn in zip(messages, turns):
            # Raw pieces were checked by the renderer; this catches controls that
            # could arise across piece boundaries and configured tokenizer specials.
            folded = message["content"].casefold()
            if any(t.casefold() in folded for t in policy["control_tokens"]):
                raise ConversationError(
                    "reserved control token in conversation content"
                )
            if turn.role == "system":
                check_visible_text(message["content"], application=True)
        rendered = render_native(tokenizer, policy, messages, for_training=training)
        native = render_native(
            tokenizer,
            policy,
            messages,
            for_training=training,
            tokenize=True,
            return_dict=True,
            **({"return_assistant_tokens_mask": True} if official else {}),
        )
        fragments = _literal_spans(tokenizer, messages, rendered, kwargs)
        hidden = re.compile(
            r"<\/?(?:analysis|reasoning|tool_use|function|parameter)|<\|recipient\|>",
            re.IGNORECASE,
        )
        if any(hidden.search(fragment) for fragment in fragments):
            raise EncodingError(
                "native template emitted unsupported reasoning or tool controls"
            )
        full = list(native["input_ids"])
        ids, spans = [], []
        controls = set(policy["control_token_ids"]) | set(
            getattr(tokenizer, "all_special_ids", ())
        )
        controls.add(getattr(tokenizer, "unk_token_id", None))

        def append(text, kind, source_id=None, *, content=False):
            if not text:
                return
            tokens = _ids(tokenizer, text)
            if not tokens:
                raise EncodingError("message span has no tokens")
            if content and controls.intersection(tokens):
                raise ConversationError(
                    "reserved control or unknown token ID in conversation content"
                )
            start = len(ids)
            ids.extend(tokens)
            spans.append(TokenSpan(start, len(ids), kind, source_id))

        append(fragments[0], "template")
        for i, turn in enumerate(turns):
            for piece in turn.pieces:
                append(piece.text, piece.kind, piece.source_id, content=True)
            suffix = fragments[i + 1]
            target = training and i == len(turns) - 1
            if target:
                stop = next(
                    (s for s in policy["stop_tokens"] if suffix.startswith(s)), None
                )
                if stop is None or _ids(tokenizer, stop) != [
                    policy["stop_token_ids"][policy["stop_tokens"].index(stop)]
                ]:
                    raise EncodingError(
                        "native target turn lacks an immediate configured atomic stop"
                    )
                append(stop, "target_stop")
                suffix = suffix[len(stop) :]
            append(suffix, "template")
        if not full or ids != full:
            raise EncodingError(
                "native token boundaries do not align with source spans (cross-boundary merge)"
            )
        if official:
            mask = native.get("assistant_masks")
            if (
                mask is None
                or len(mask) != len(full)
                or any(
                    not all(mask[span.start : span.end])
                    for span in spans
                    if span.kind == "target_body"
                )
            ):
                raise EncodingError(
                    "native assistant mask disagrees with target body spans"
                )
        return full, tuple(spans), rendered
    except (EncodingError, ConversationError):
        raise
    except Exception:  # noqa: BLE001 -- tokenizer/Jinja errors can contain untrusted bodies.
        raise EncodingError(
            "native template could not render or align conversation spans"
        ) from None


def _observed_prefix(tokenizer, policy, observation, target_person):
    turns = render_observation(observation, target_person, policy["system_prompt"])
    ids, _, rendered = _conversation_layout(tokenizer, policy, turns, training=False)
    try:
        plain = render_native(
            tokenizer, policy, [turn.message() for turn in turns], for_training=True
        )
    except Exception:  # noqa: BLE001
        raise EncodingError(
            "native template could not render observed conversation"
        ) from None
    suffix = rendered[len(plain) :] if rendered.startswith(plain) else ""
    if not suffix.strip() or any(stop in suffix for stop in policy["stop_tokens"]):
        raise EncodingError("missing or unsupported native assistant generation prompt")
    return turns, ids, rendered


def _verify_native_binding(tokenizer, policy, template=None):
    try:
        native = tokenizer.get_chat_template()
        matched = (
            (template is None or native == template)
            and (hashlib.sha256(native.encode()).hexdigest() == policy["template_hash"])
            and all(
                _ids(tokenizer, stop) == [token_id]
                for stop, token_id in zip(
                    policy["stop_tokens"], policy["stop_token_ids"], strict=True
                )
            )
        )
    except Exception:  # noqa: BLE001 -- never expose tokenizer configuration text.
        matched = False
    if not matched:
        raise EncodingError(
            "native template or stop binding does not match resolved policy"
        )


def encode_conversation_prefix(
    tokenizer, policy, observation, target_person, *, max_length=None
):
    """Native direct inference IDs (caller supplies a resolved template policy)."""
    _verify_native_binding(tokenizer, policy)
    _, ids, _ = _observed_prefix(tokenizer, policy, observation, target_person)
    if max_length is not None and len(ids) > max_length:
        raise EncodingError(
            "conversation prefix exceeds max_length; no history truncated"
        )
    return ids


def verify_training_prefix(encoded, input_ids, *, identity=None):
    """Compare independent direct inference output to the proved training cut."""
    start = next((s.start for s in encoded.spans if s.kind == "target_body"), 0)
    if (
        not start
        or start != len(encoded.prefix_ids)
        or tuple(input_ids) != encoded.prefix_ids
        or tuple(encoded.input_ids[:start]) != encoded.prefix_ids
    ):
        raise EncodingError("training and direct inference token prefixes disagree")
    if identity is not None and any(
        identity.get(k) != getattr(encoded, k)
        for k in ("renderer_version", "template_hash", "stop_hash")
    ):
        raise EncodingError(
            "training and direct inference renderer/template/stop identities disagree"
        )


def encode_conversation(tokenizer, window, template, policy, plan):
    """Compile one independent sequence per opportunity; no state or loss is shared."""
    _verify_native_binding(tokenizer, policy, template)
    results = []
    official = re.search(r"{%[-+]?\s*generation\s*[-+]?%}", template) is not None
    identity = conversation_identity(policy)
    native_limit = getattr(tokenizer, "model_max_length", None)
    limit = (
        min(plan["max_length"], native_limit)
        if type(native_limit) is int
        else plan["max_length"]
    )
    for events in window.opportunities:
        history, prefix, prompt = _observed_prefix(
            tokenizer,
            policy,
            window.observation(events[0].source_id),
            window.target_person,
        )
        target = target_turn(events)
        full, spans, rendered = _conversation_layout(
            tokenizer,
            policy,
            (*history, target),
            training=True,
            official=official,
        )
        if len(full) > limit:
            raise EncodingError("SFT context exceeds max_length; no target truncated")
        cut = next(s.start for s in spans if s.kind == "target_body")
        # Reading the actual literal slot boundary uses known segment lengths, not
        # a search for (possibly repeated) assistant text.
        if not cut or not rendered[: len(prompt)] == prompt or cut != len(prefix):
            raise EncodingError("training and direct inference token prefixes disagree")
        labels, roles = [IGNORE] * len(full), ["template"] * len(full)
        bodies = stops = 0
        for span in spans:
            a, b = span.start, span.end
            roles[a:b] = [span.kind] * (b - a)
            if span.kind == "target_body" or (
                span.kind == "target_stop" and plan["supervise_target_eos"]
            ):
                labels[a:b] = full[a:b]
                bodies += (b - a) if span.kind == "target_body" else 0
                stops += (b - a) if span.kind == "target_stop" else 0
        if not bodies or labels[0] != IGNORE:
            raise EncodingError("SFT target has no predictable visible body tokens")
        encoded = Encoded(
            full,
            labels,
            roles,
            bodies,
            stops,
            spans=spans,
            prefix_ids=tuple(full[:cut]),
            focus_ids=tuple(e.source_id for e in events),
            **identity,
        )
        verify_training_prefix(encoded, prefix)
        results.append(encoded)
    return tuple(results)
