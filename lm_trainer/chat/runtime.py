"""In-memory sessions and a serialized active-model lifecycle."""
# ruff: noqa: BLE001

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from queue import SimpleQueue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from .backend import HFBackend, Trace
from .context import ContextBuilder
from .memory import ContextCard, MemoryProvider, NoMemoryProvider, audit
from .registry import ModelRegistry
from .types import (
    ActivationInfo,
    CancelResult,
    ChatError,
    ChatEvent,
    ChatResult,
    GenerationSettings,
    Message,
    ModelInfo,
    RoleConfig,
    SessionInfo,
    valid_id,
)


def _now():
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class _Request:
    started: float = field(default_factory=perf_counter)
    request_id: str = field(default_factory=lambda: uuid4().hex)
    turn_id: str = field(default_factory=lambda: uuid4().hex)
    cancellation: Event = field(default_factory=Event)


@dataclass(slots=True)
class _Session:
    session_id: str
    model_id: str
    role: RoleConfig
    settings: GenerationSettings
    messages: tuple[Message, ...] = ()
    request: _Request | None = None
    switching: bool = False

    def info(self):
        return SessionInfo(
            self.session_id,
            self.model_id,
            self.role.role_id,
            self.role.version,
            self.role.namespace,
            self.settings,
            self.messages,
            self.request.request_id if self.request else None,
        )


class ChatStream(Iterator[ChatEvent]):
    """A single-consumer iterator. The request ID is available before the first delta."""

    def __init__(self, request, cancel, queue, worker):
        self.request_id = request.request_id
        self._cancel = cancel
        self._cancelled = request.cancellation
        self._queue = queue
        self._worker = worker
        self._finished = False

    def __next__(self) -> ChatEvent:
        if self._finished:
            raise StopIteration
        while True:
            event = self._queue.get()
            if event.kind != "delta":
                self._finished = True
                self._worker.join()
                return event
            if not self._cancelled.is_set():
                return event

    def close(self):
        if not self._finished:
            self._cancel(self.request_id)
            self._worker.join()
            self._finished = True
            while not self._queue.empty():
                self._queue.get_nowait()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ChatRuntimeProtocol(Protocol):
    def list_models(self) -> list[ModelInfo]: ...
    def create_session(
        self, model_id: str, role_id: str, settings: GenerationSettings | None = None
    ) -> SessionInfo: ...
    def get_session(self, session_id: str) -> SessionInfo: ...
    def send(
        self,
        session_id: str,
        user_text: str,
        memory_cards: list[ContextCard] | None = None,
    ) -> ChatResult: ...
    def stream(
        self,
        session_id: str,
        user_text: str,
        memory_cards: list[ContextCard] | None = None,
    ) -> ChatStream: ...
    def cancel(self, session_id: str, request_id: str) -> CancelResult: ...
    def clear_session(self, session_id: str) -> SessionInfo: ...
    def switch_model(
        self, session_id: str, model_id: str, *, new_session: bool = True
    ) -> SessionInfo: ...


class ChatRuntime:
    def __init__(
        self,
        registry: ModelRegistry,
        roles: tuple[RoleConfig, ...] | list[RoleConfig],
        *,
        memory_provider: MemoryProvider | None = None,
        context_builder: ContextBuilder | None = None,
    ):
        self._init(
            HFBackend(registry),
            registry.list_models,
            lambda mid: registry.spec(mid).info(),
            roles,
            memory_provider,
            context_builder,
        )

    def _init(
        self, backend, list_models, describe, roles, memory_provider, context_builder
    ):
        self._backend = backend
        self._list_models = list_models
        self._describe = describe
        self._roles = {role.role_id: role for role in roles}
        if len(self._roles) != len(roles):
            raise ChatError("INVALID_CONFIG")
        self._memory = (
            memory_provider if memory_provider is not None else NoMemoryProvider()
        )
        self._builder = (
            context_builder if context_builder is not None else ContextBuilder()
        )
        self._sessions: dict[str, _Session] = {}
        self._state = Lock()
        self._model_lock = Lock()

    def list_models(self) -> list[ModelInfo]:
        return self._list_models()

    def _session(self, session_id):
        if not valid_id(session_id) or session_id not in self._sessions:
            raise ChatError("UNKNOWN_SESSION")
        return self._sessions[session_id]

    def _new_session(self, model_id, role, settings):
        s = _Session(uuid4().hex, model_id, role, settings)
        self._sessions[s.session_id] = s
        return s.info()

    def create_session(
        self, model_id: str, role_id: str, settings: GenerationSettings | None = None
    ) -> SessionInfo:
        self._describe(model_id)
        if not valid_id(role_id) or role_id not in self._roles:
            raise ChatError("UNKNOWN_ROLE")
        if settings is not None and not isinstance(settings, GenerationSettings):
            raise ChatError("INVALID_CONFIG")
        with self._state:
            return self._new_session(
                model_id, self._roles[role_id], settings or GenerationSettings()
            )

    def get_session(self, session_id: str) -> SessionInfo:
        with self._state:
            return self._session(session_id).info()

    @staticmethod
    def _require_idle(s):
        if s.request is not None or s.switching:
            raise ChatError("SESSION_BUSY")

    def clear_session(self, session_id: str) -> SessionInfo:
        with self._state:
            s = self._session(session_id)
            self._require_idle(s)
            s.messages = ()
            return s.info()

    def active_model(self) -> ActivationInfo | None:
        with self._model_lock:
            return self._backend.active

    def unload(self):
        with self._model_lock:
            try:
                self._backend.unload()
            except Exception:
                raise ChatError("LOAD_FAILED") from None

    def switch_model(
        self, session_id: str, model_id: str, *, new_session: bool = True
    ) -> SessionInfo:
        self._describe(model_id)
        if type(new_session) is not bool:
            raise ChatError("INVALID_INPUT")
        with self._state:
            s = self._session(session_id)
            self._require_idle(s)
            s.switching = True
        try:
            with self._model_lock:
                self._backend.activate(model_id)
                with self._state:
                    if new_session:
                        return self._new_session(model_id, s.role, s.settings)
                    if s.model_id != model_id:
                        s.messages = ()
                        s.model_id = model_id
                    return s.info()
        finally:
            with self._state:
                s.switching = False

    def _reserve(self, session_id, text, cards):
        if isinstance(cards, (list, tuple)):
            cards = tuple(cards)
        if (
            not isinstance(text, str)
            or not text.strip()
            or (
                cards is not None
                and (
                    not isinstance(cards, (list, tuple))
                    or len(cards) > 128
                    or not all(isinstance(c, ContextCard) for c in cards)
                )
            )
        ):
            raise ChatError("INVALID_INPUT")
        with self._state:
            s = self._session(session_id)
            self._require_idle(s)
            s.request = request = _Request()
            return s, request, tuple(cards) if cards is not None else ()

    def cancel(self, session_id: str, request_id: str) -> CancelResult:
        with self._state:
            r = self._session(session_id).request
            accepted = r is not None and r.request_id == request_id
            if accepted:
                r.cancellation.set()
            return CancelResult(session_id, request_id, accepted)

    def send(
        self,
        session_id: str,
        user_text: str,
        memory_cards: list[ContextCard] | None = None,
    ) -> ChatResult:
        s, request, cards = self._reserve(session_id, user_text, memory_cards)
        return self._run(s, request, user_text, cards)

    def stream(
        self,
        session_id: str,
        user_text: str,
        memory_cards: list[ContextCard] | None = None,
    ) -> ChatStream:
        s, request, cards = self._reserve(session_id, user_text, memory_cards)
        queue = SimpleQueue()

        def run():
            def emit(delta):
                queue.put(
                    ChatEvent(
                        "delta", request.request_id, session_id, request.turn_id, delta
                    )
                )

            result = self._run(s, request, user_text, cards, emit)
            kind = {"cancel": "cancel", "error": "error"}.get(
                result.finish_reason, "done"
            )
            queue.put(
                ChatEvent(
                    kind, request.request_id, session_id, request.turn_id, result=result
                )
            )

        worker = Thread(target=run, name="lm-trainer-chat", daemon=True)
        try:
            worker.start()
        except RuntimeError:
            with self._state:
                s.request = None
            raise ChatError("GENERATION_FAILED") from None
        return ChatStream(
            request, lambda rid: self.cancel(session_id, rid), queue, worker
        )

    def _run(
        self, s, request, user_text, cards, publish: Callable[[str], None] | None = None
    ):
        trace = Trace(request.started)
        cancellation = request.cancellation
        info = self._describe(s.model_id)
        error = None
        text, reason = "", "cancel"

        def emit(chunk):
            if (
                publish is not None
                and info.supports_streaming
                and not cancellation.is_set()
            ):
                if trace.ttft_ms is None:
                    trace.ttft_ms = (perf_counter() - trace.started) * 1000
                publish(chunk)

        try:
            if not cancellation.is_set():
                selected, entries = self._memory.select(
                    cards, role=s.role, session_id=s.session_id
                )
                trace.card_audit = entries
                acquired = False
                while not cancellation.is_set() and not acquired:
                    acquired = self._model_lock.acquire(timeout=0.025)
                if acquired:
                    try:
                        if not cancellation.is_set():
                            if (
                                not self._backend.active
                                or self._backend.active.model_id != s.model_id
                            ):
                                trace.activation = self._backend.activate(s.model_id)
                            if not cancellation.is_set():
                                text, reason = self._backend.respond(
                                    role=s.role,
                                    history=s.messages,
                                    text=user_text,
                                    cards=selected,
                                    audit_entries=entries,
                                    settings=s.settings,
                                    builder=self._builder,
                                    cancel=cancellation,
                                    emit=emit,
                                    trace=trace,
                                )
                    finally:
                        self._model_lock.release()
        except ChatError as exc:
            reason, error = "error", exc.code
        except Exception:
            reason, error = "error", "GENERATION_FAILED"
        except BaseException:
            if publish is None:
                # Propagate process-control exceptions to synchronous callers, after release.
                with self._state:
                    s.request = None
                raise
            # A background worker must still deliver a terminal event to its consumer.
            reason, error = "error", "GENERATION_FAILED"

        with self._state:
            if cancellation.is_set():
                reason, error = "cancel", None
            committed = reason not in ("error", "cancel") and bool(text.strip())
            if committed:
                s.messages += tuple(
                    Message(
                        role,
                        content,
                        s.session_id,
                        request.turn_id,
                        _now(),
                        s.model_id,
                        s.role.role_id,
                        s.role.version,
                        info.template_hash,
                    )
                    for role, content in (("user", user_text), ("assistant", text))
                )
            s.request = None
        return ChatResult(
            request.request_id,
            s.session_id,
            request.turn_id,
            info.model_id,
            s.role.role_id,
            info.template_hash,
            info.fixture_only,
            text if reason not in ("error", "cancel") else "",
            reason,
            committed,
            trace.input_tokens,
            trace.output_tokens,
            trace.ttft_ms,
            trace.tokens_per_second,
            (perf_counter() - trace.started) * 1000,
            trace.peak_vram_mb,
            error,
            trace.activation,
            tuple(
                replace(x, reason="not_sent")
                if x.reason == "injected" and trace.input_tokens is None
                else x
                for x in trace.card_audit
            ),
            trace.dropped_turns,
        )

    def save_session(self, session_id: str, output_dir: Path | str) -> Path:
        """Explicit export only, into an existing private directory owned by the caller."""
        info = self.get_session(session_id)
        dirname = Path(output_dir).absolute()
        filename = f"chat-{info.session_id}-{uuid4().hex}.json"
        try:
            fd = os.open(dirname, os.O_DIRECTORY | os.O_NOFOLLOW | os.O_RDONLY)
            try:
                mode = os.fstat(fd)
                if mode.st_uid != os.getuid() or stat.S_IMODE(mode.st_mode) != 0o700:
                    raise PermissionError
                out = os.open(
                    filename,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=fd,
                )
                with os.fdopen(out, "w", encoding="utf-8") as handle:
                    json.dump(asdict(info), handle, ensure_ascii=False)
            finally:
                os.close(fd)
        except (OSError, ValueError, TypeError):
            raise ChatError("SAVE_FAILED") from None
        return dirname / filename


class _MockBackend:
    def __init__(self, lookup, responder):
        self.lookup = lookup
        self.responder = responder
        self.active = None

    def activate(self, model_id):
        info = self.lookup(model_id)
        if not info.available:
            raise ChatError(info.reason or "MODEL_UNAVAILABLE")
        self.active = ActivationInfo(model_id, None, None, None)
        return self.active

    def unload(self):
        self.active = None

    def respond(self, *, role, history, text, cards, trace, **_):
        trace.card_audit += tuple(audit(c, "not_sent") for c in cards)
        reply = self.responder(role, history, text)
        if not isinstance(reply, str):
            raise ChatError("GENERATION_FAILED")
        return reply, "eos"


class MockChatRuntime(ChatRuntime):
    """Same session/request semantics, no model packages, no simulated token metrics or deltas."""

    def __init__(
        self,
        models: tuple[ModelInfo, ...] | list[ModelInfo],
        roles: tuple[RoleConfig, ...] | list[RoleConfig],
        *,
        responder: Callable[[RoleConfig, tuple[Message, ...], str], str] | None = None,
        memory_provider: MemoryProvider | None = None,
    ):
        infos = {m.model_id: replace(m, supports_streaming=False) for m in models}
        if len(infos) != len(models):
            raise ChatError("INVALID_CONFIG")

        def lookup(model_id):
            if not valid_id(model_id) or model_id not in infos:
                raise ChatError("UNKNOWN_MODEL")
            return infos[model_id]

        self._init(
            _MockBackend(lookup, responder or (lambda *_: "Mock reply.")),
            lambda: list(infos.values()),
            lookup,
            roles,
            memory_provider,
            None,
        )
