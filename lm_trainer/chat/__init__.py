"""Public, headless Task C v1 API. Backend imports are lazy."""

from .context import ContextBuilder
from .memory import (
    CardPermissions,
    ContextCard,
    ManualCardProvider,
    MemoryProvider,
    NoMemoryProvider,
)
from .registry import ModelRegistry, ModelSpec, load_registry, load_roles
from .runtime import ChatRuntime, ChatRuntimeProtocol, ChatStream, MockChatRuntime
from .types import (
    ActivationInfo,
    CancelResult,
    CardAudit,
    ChatError,
    ChatEvent,
    ChatResult,
    ErrorCode,
    FinishReason,
    GenerationSettings,
    Message,
    ModelInfo,
    RoleConfig,
    SessionInfo,
)

__all__ = [
    "ActivationInfo",
    "CancelResult",
    "CardAudit",
    "CardPermissions",
    "ChatError",
    "ChatEvent",
    "ChatResult",
    "ChatRuntime",
    "ChatRuntimeProtocol",
    "ChatStream",
    "ContextBuilder",
    "ContextCard",
    "ErrorCode",
    "FinishReason",
    "GenerationSettings",
    "ManualCardProvider",
    "MemoryProvider",
    "Message",
    "MockChatRuntime",
    "ModelInfo",
    "ModelRegistry",
    "ModelSpec",
    "NoMemoryProvider",
    "RoleConfig",
    "SessionInfo",
    "load_registry",
    "load_roles",
]
