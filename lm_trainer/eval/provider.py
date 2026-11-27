"""Runtime providers: the single dependency-injection point for Task D.

``RuntimeProvider`` is the only seam between the evaluation/UI layer and Task C.
Swapping ``MockRuntimeProvider`` for ``RegistryRuntimeProvider`` changes the
backend from offline mock to real local inference without touching any caller.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from lm_trainer.chat import (
    ChatRuntime,
    ChatRuntimeProtocol,
    ManualCardProvider,
    MemoryProvider,
    MockChatRuntime,
    ModelInfo,
    ModelRegistry,
    RoleConfig,
    load_registry,
    load_roles,
)

MOCK = "mock"
REAL = "real_inference"


class RuntimeProvider:
    mode = "abstract"

    def list_models(self) -> list[ModelInfo]:
        raise NotImplementedError

    def build(self, model_id: str) -> ChatRuntimeProtocol:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"mode": self.mode, "type": type(self).__name__}


class MockRuntimeProvider(RuntimeProvider):
    """Offline provider built on Task C's ``MockChatRuntime``.

    One shared runtime instance holds all sessions. Rebuilding a runtime per
    request would drop session history, so ``build()`` always returns the same
    object and only validates the model id.
    """

    mode = MOCK

    def __init__(
        self,
        models: tuple[ModelInfo, ...],
        roles: tuple[RoleConfig, ...],
        *,
        responder: Callable[[str, RoleConfig, tuple, str], str],
        memory_provider: MemoryProvider | None = None,
    ):
        # A real ManualCardProvider keeps all namespace/scope/ACL/revocation
        # decisions inside Task C. Passing ``memory_cards=None`` selects nothing,
        # which is exactly the "cards off" default.
        memory_provider = memory_provider or ManualCardProvider()
        self.models = tuple(models)
        self.roles = tuple(roles)
        self._memory_provider = memory_provider
        # ``MockChatRuntime`` routes every request through one responder and
        # carries no model id, so a shared Mock provider can only serve one
        # active model at a time. Headless evaluation creates a provider per
        # model; the interactive Mock UI pins this to the first model.
        self._responder_model = self.models[0].model_id
        self._responder = responder
        # MockChatRuntime's responder signature carries no model id, so route
        # by the active model the runtime reports at request time. That value is
        # per-runtime (shared), so a Mock provider is intended for a single
        # interactive operator or for headless serial evaluation, not for
        # concurrent multi-model sessions.
        self._runtime = MockChatRuntime(
            self.models,
            self.roles,
            responder=self._route,
            memory_provider=memory_provider,
        )

    def _route(self, role, history, text):
        # ``active_model()`` acquires the model lock, which the caller already
        # holds while generating: taking it again would deadlock the request
        # thread. The responder therefore runs without consulting runtime state.
        return self._responder(self.models[0].model_id, role, history, text)

    def list_models(self) -> list[ModelInfo]:
        return list(self.models)

    def build(self, model_id: str) -> ChatRuntimeProtocol:
        if not any(m.model_id == model_id for m in self.models):
            raise KeyError(model_id)
        # Task C keeps authoritative history and the model lock inside the
        # runtime. Returning the same instance is what makes multi-turn state
        # survive across browser requests.
        return self._runtime

    def describe(self) -> dict:
        return {"mode": self.mode, "type": type(self).__name__, "shared": True}


class RegistryRuntimeProvider(RuntimeProvider):
    """Real local inference: one Task C runtime per model, released after use."""

    mode = REAL

    def __init__(
        self,
        models_config: Path | str,
        roles_config: Path | str,
        *,
        allow_fixture: bool = False,
        base_override: Path | None = None,
        device_override: str | None = None,
        memory_provider: MemoryProvider | None = None,
    ):
        self._models_config = Path(models_config)
        self._roles_config = Path(roles_config)
        self._allow_fixture = bool(allow_fixture)
        self._base_override = base_override
        self._device_override = device_override
        self._memory_provider = memory_provider or ManualCardProvider()
        self.roles: tuple[RoleConfig, ...] = load_roles(roles_config)
        self._registry = self._load()

    def _load(self) -> ModelRegistry:
        return load_registry(
            self._models_config,
            allow_fixture=self._allow_fixture,
            base_override=self._base_override,
            device_override=self._device_override,
        )

    def list_models(self) -> list[ModelInfo]:
        return self._registry.list_models()

    def build(self, model_id: str) -> ChatRuntimeProtocol:
        spec = self._registry.spec(model_id)
        single = ModelRegistry([spec], allow_fixture=self._allow_fixture)
        return ChatRuntime(
            single, self.roles, memory_provider=self._memory_provider
        )

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "type": type(self).__name__,
            "models_config": self._models_config.name,
            "roles_config": self._roles_config.name,
            "allow_fixture": self._allow_fixture,
        }


def manual_cards_provider() -> ManualCardProvider:
    """The only card provider used by Task D; Task C enforces namespace/ACL."""

    return ManualCardProvider()


def planned_models(provider: RuntimeProvider, requested: Sequence[str]) -> list[ModelInfo]:
    """Return planned models with availability, preserving the requested order."""

    available = {m.model_id: m for m in provider.list_models()}
    result = []
    for model_id in requested:
        info = available.get(model_id)
        if info is None:
            result.append(
                ModelInfo(model_id, "adapter", False, None, None, "UNKNOWN_MODEL")
            )
        else:
            result.append(info)
    return result
