"""Task C v1 values. Importing this module never initializes a model backend."""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from typing import Literal

FinishReason = Literal["eos", "length", "stop", "cancel", "error"]
ErrorCode = Literal[
    "INVALID_CONFIG",
    "INVALID_INPUT",
    "UNKNOWN_MODEL",
    "UNKNOWN_ROLE",
    "UNKNOWN_SESSION",
    "SESSION_BUSY",
    "MODEL_UNAVAILABLE",
    "ARTIFACT_MISMATCH",
    "TEMPLATE_UNSUPPORTED",
    "CONTEXT_LIMIT",
    "UNSAFE_CONTENT",
    "DEVICE_BUSY",
    "OUT_OF_MEMORY",
    "LOAD_FAILED",
    "GENERATION_FAILED",
    "UNSAFE_OUTPUT",
    "SAVE_FAILED",
    "AUTH_REQUIRED",
    "FORBIDDEN",
    "UNSUPPORTED_FIELD",
    "PAYLOAD_TOO_LARGE",
    "QUEUE_FULL",
    "TIMEOUT",
    "CANCELLED",
    "REQUEST_NOT_FOUND",
    "SERVICE_UNAVAILABLE",
]

_MESSAGES: dict[ErrorCode, str] = {
    "INVALID_CONFIG": "Invalid chat configuration.",
    "INVALID_INPUT": "Invalid chat input.",
    "UNKNOWN_MODEL": "Model ID is not registered.",
    "UNKNOWN_ROLE": "Role ID is not configured.",
    "UNKNOWN_SESSION": "Session does not exist.",
    "SESSION_BUSY": "Session already has an active request or model switch.",
    "MODEL_UNAVAILABLE": "Local model, tokenizer, device or adapter is unavailable.",
    "ARTIFACT_MISMATCH": "Adapter failed the pinned artifact or compatibility checks.",
    "TEMPLATE_UNSUPPORTED": "Native visible-reply template or role binding failed validation.",
    "CONTEXT_LIMIT": "System instructions and current input do not fit; shorten the input or generation budget.",
    "UNSAFE_CONTENT": "Input contains a reserved template or protocol marker.",
    "DEVICE_BUSY": "Another runtime in this process has an active model on this GPU.",
    "OUT_OF_MEMORY": "Insufficient memory; no alternate model was loaded.",
    "LOAD_FAILED": "The local inference backend could not be loaded.",
    "GENERATION_FAILED": "The inference backend failed to generate a reply.",
    "UNSAFE_OUTPUT": "Generation produced a control or unsupported protocol block.",
    "SAVE_FAILED": "History could not be saved to a private output directory.",
    "AUTH_REQUIRED": "A valid service credential is required.",
    "FORBIDDEN": "The credential does not authorize this operation.",
    "UNSUPPORTED_FIELD": "The request uses an unsupported field or mode.",
    "PAYLOAD_TOO_LARGE": "The request body exceeds the service limit.",
    "QUEUE_FULL": "The local inference queue is full.",
    "TIMEOUT": "The local inference request exceeded its deadline.",
    "CANCELLED": "The request was cancelled.",
    "REQUEST_NOT_FOUND": "No matching active request exists.",
    "SERVICE_UNAVAILABLE": "The local inference service is not ready.",
}


class ChatError(Exception):
    """Public errors are fixed and never include exception details from the backend."""

    def __init__(self, code: ErrorCode):
        self.code = code
        super().__init__(_MESSAGES[code])


def _positive_finite(value: object) -> bool:
    # math.isfinite raises OverflowError for arbitrarily large Python integers.
    return (
        type(value) in (int, float)
        and 0 < value <= sys.float_info.max
        and math.isfinite(value)
    )


def valid_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[\w.:-]{1,160}", value) is not None


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    seed: int | None = None

    def __post_init__(self):
        if (
            type(self.max_new_tokens) is not int
            or self.max_new_tokens < 1
            or type(self.do_sample) is not bool
            or type(self.top_k) is not int
            or self.top_k < 0
            or (
                self.seed is not None
                and (type(self.seed) is not int or not 0 <= self.seed < 2**63)
            )
            or not all(
                _positive_finite(x)
                for x in (self.temperature, self.top_p, self.repetition_penalty)
            )
            or self.top_p > 1
        ):
            raise ChatError("INVALID_CONFIG")


@dataclass(frozen=True, slots=True)
class RoleConfig:
    role_id: str
    namespace: str
    version: str
    system_prompt: str = field(repr=False)

    def __post_init__(self):
        if not all(map(valid_id, (self.role_id, self.namespace, self.version))) or not (
            isinstance(self.system_prompt, str) and self.system_prompt.strip()
        ):
            raise ChatError("INVALID_CONFIG")


@dataclass(frozen=True, slots=True)
class ModelInfo:
    model_id: str
    kind: Literal["base", "adapter"]
    available: bool
    fixture_only: bool | None
    template_hash: str | None = None
    reason: ErrorCode | None = None
    supports_streaming: bool = True


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: str = field(repr=False)
    session_id: str
    turn_id: str
    timestamp: str
    model_id: str
    role_id: str
    role_version: str
    template_hash: str | None


@dataclass(frozen=True, slots=True)
class SessionInfo:
    session_id: str
    model_id: str
    role_id: str
    role_version: str
    namespace: str
    settings: GenerationSettings
    messages: tuple[Message, ...]
    active_request_id: str | None


@dataclass(frozen=True, slots=True)
class CancelResult:
    session_id: str
    request_id: str
    accepted: bool


@dataclass(frozen=True, slots=True)
class CardAudit:
    card_id: str
    version: int
    reason: str  # injected, memory_disabled, namespace, scope, status, stale, ...


@dataclass(frozen=True, slots=True)
class ActivationInfo:
    model_id: str
    load_ms: float | None
    warmup_ms: float | None
    resident_vram_mb: float | None


@dataclass(frozen=True, slots=True)
class ChatResult:
    request_id: str
    session_id: str
    turn_id: str
    model_id: str
    role_id: str
    template_hash: str | None
    fixture_only: bool | None
    text: str = field(repr=False)
    finish_reason: FinishReason
    committed: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_ms: float | None = None
    tokens_per_second: float | None = None
    latency_ms: float | None = None
    peak_vram_mb: float | None = None
    error_code: ErrorCode | None = None
    activation: ActivationInfo | None = None
    card_audit: tuple[CardAudit, ...] = ()
    dropped_turns: int | None = None


@dataclass(frozen=True, slots=True)
class ChatEvent:
    kind: Literal["delta", "done", "cancel", "error"]
    request_id: str
    session_id: str
    turn_id: str
    delta: str = field(default="", repr=False)
    result: ChatResult | None = None
