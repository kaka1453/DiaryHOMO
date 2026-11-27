"""Shared native chat-template policy. No model or tokenizer imports."""

from __future__ import annotations

import hashlib
import re

POLICY_SCHEMA = "lm_trainer.template-policy.v1"
ROLE_MAP = {role: role for role in ("system", "user", "assistant")}
# Unsupported protocol markers are rejected, not removed from message text.
PROTOCOL_MARKERS = (
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<|channel|>",
    "<|message|>",
)


def validate_policy(value):
    allowed = {
        "version",
        "template_hash",
        "stop_tokens",
        "enable_thinking",
        "system_prompt",
        "control_tokens",
    }
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(
            "SFT requires a native template_policy with explicit stop_tokens"
        )
    version = value.get("version", POLICY_SCHEMA)
    digest = value.get("template_hash")
    stops = value.get("stop_tokens")
    controls = value.get("control_tokens", [])
    thinking = value.get("enable_thinking")
    system = value.get("system_prompt")
    if not isinstance(version, str) or not re.fullmatch(r"[\w.:-]{1,160}", version):
        raise ValueError("invalid template_policy version")
    if digest is not None and (
        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)
    ):
        raise ValueError("template_hash must be SHA-256")
    for tokens in (stops, controls):
        if (
            not isinstance(tokens, list)
            or not all(isinstance(t, str) and t.strip() for t in tokens)
            or len(set(tokens)) != len(tokens)
        ):
            raise ValueError(
                "template_policy token lists require unique nonempty strings"
            )
    if not stops or (thinking is not None and thinking is not False):
        raise ValueError(
            "template_policy requires stop_tokens and supports only visible replies"
        )
    if system is not None and (not isinstance(system, str) or not system.strip()):
        raise ValueError("template_policy.system_prompt must be nonempty text")
    return {
        "version": version,
        "template_hash": digest.lower() if digest else None,
        "stop_tokens": stops,
        "enable_thinking": thinking,
        "system_prompt": system,
        "control_tokens": controls,
    }


def chat_template_kwargs(policy, *, for_training):
    """Call apply_chat_template with these flags; never add specials a second time."""
    kwargs = {"add_generation_prompt": not for_training}
    if policy["enable_thinking"] is False:
        kwargs["enable_thinking"] = False
    return kwargs


def resolve_policy(tokenizer, policy):
    policy = validate_policy(policy)
    try:
        template = tokenizer.get_chat_template()
    except (AttributeError, ValueError):
        raise ValueError("tokenizer has no unambiguous native chat template") from None
    if not isinstance(template, str) or not template:
        raise ValueError("tokenizer has no unambiguous native chat template")
    digest = hashlib.sha256(template.encode()).hexdigest()
    if policy["template_hash"] not in (None, digest):
        raise ValueError("native chat template SHA-256 mismatch")
    if policy["enable_thinking"] is False and not re.search(
        r"\benable_thinking\b", template
    ):
        raise ValueError("native template does not expose enable_thinking")
    # Explicit stops are atomic registered tokens, not arbitrary content strings.
    special = set(tokenizer.all_special_tokens)
    added_vocab = tokenizer.get_added_vocab()
    added = set(added_vocab)
    stops = policy["stop_tokens"]
    stop_ids = []
    for stop in stops:
        tokens = tokenizer(stop, add_special_tokens=False)["input_ids"]
        if stop not in special | added or len(tokens) != 1:
            raise ValueError("native turn stop must be an atomic registered token")
        stop_ids.extend(tokens)
    # Added base-model tokens often are controls even if not flagged 'special'.
    controls = special | {t for t in added if t.startswith(("<|", "[INST", "[/INST"))}
    controls.update(stops + policy["control_tokens"] + list(PROTOCOL_MARKERS))
    control_ids = {added_vocab[token] for token in controls if token in added_vocab}
    control_ids.update(stop_ids)
    return template, {
        **policy,
        "template_hash": digest,
        "schema": POLICY_SCHEMA,
        "role_map": ROLE_MAP.copy(),
        "visible_reply_only": True,
        "stop_token_ids": stop_ids,
        "control_tokens": sorted(controls),
        "control_token_ids": sorted(
            control_ids - ({tokenizer.unk_token_id} - set(stop_ids))
        ),
        "training_template_kwargs": chat_template_kwargs(policy, for_training=True),
        "inference_template_kwargs": chat_template_kwargs(policy, for_training=False),
    }


def public_policy(policy):
    """Reports describe but do not print configured application prompt text."""
    if policy is None:
        return None
    result = dict(policy)
    system = result.pop("system_prompt", None)
    result["system_prompt_sha256"] = (
        hashlib.sha256(system.encode()).hexdigest() if system else None
    )
    return result
