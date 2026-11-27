"""Versioned stateless inference using the existing headless model slot."""
# ruff: noqa: BLE001

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, replace
from functools import partial
from queue import SimpleQueue
from threading import Condition, Lock, Thread
from time import perf_counter
from typing import Literal, Protocol
from uuid import uuid4

from lm_trainer.conversation.contract import Observation

from .backend import Trace
from .context import check_inference_text
from .memory import ContextCard, ManualCardProvider, audit
from .runtime import ChatRuntime, MockChatRuntime
from .types import (
    CardAudit,
    ChatError,
    ErrorCode,
    FinishReason,
    GenerationSettings,
    RoleConfig,
    valid_id,
)

AUDIT_LOG = logging.getLogger("lm_trainer.inference.audit")


def error_status(code: ErrorCode) -> int:
    return {
        "AUTH_REQUIRED": 401,
        "FORBIDDEN": 403,
        "UNKNOWN_MODEL": 404,
        "REQUEST_NOT_FOUND": 404,
        "PAYLOAD_TOO_LARGE": 413,
        "QUEUE_FULL": 429,
        "CANCELLED": 409,
        "TIMEOUT": 504,
        "CONTEXT_LIMIT": 422,
        "UNSAFE_CONTENT": 422,
        "MODEL_UNAVAILABLE": 503,
        "ARTIFACT_MISMATCH": 503,
        "TEMPLATE_UNSUPPORTED": 503,
        "DEVICE_BUSY": 503,
        "OUT_OF_MEMORY": 503,
        "LOAD_FAILED": 503,
        "SERVICE_UNAVAILABLE": 503,
        "GENERATION_FAILED": 500,
        "UNSAFE_OUTPUT": 500,
    }.get(code, 400)


def write_audit(**metadata):
    AUDIT_LOG.info("%s", json.dumps(metadata, ensure_ascii=True, allow_nan=False))


@dataclass(frozen=True, slots=True)
class ScopeBindingV1:
    scope: str
    role_id: str
    input_kind: Literal["messages", "observation"] = "messages"
    conversation_id: str | None = None
    target_person: str | None = None
    conversation_mode: Literal["direct", "group"] | None = None

    def __post_init__(self):
        if not all(map(valid_id, (self.scope, self.role_id))) or not (
            (
                self.input_kind == "messages"
                and self.conversation_id is None
                and self.target_person is None
                and self.conversation_mode is None
            )
            or (
                self.input_kind == "observation"
                and all(map(valid_id, (self.conversation_id, self.target_person)))
                and self.conversation_mode in ("direct", "group")
            )
        ):
            raise ChatError("INVALID_CONFIG")


@dataclass(frozen=True, slots=True)
class InferenceRequestV1:
    model_id: str
    scope: str
    messages: tuple[tuple[str, str], ...] | None = field(default=None, repr=False)
    observation: Observation | None = field(default=None, repr=False)
    settings: GenerationSettings | None = None

    def __post_init__(self):
        if (
            not valid_id(self.model_id)
            or not valid_id(self.scope)
            or (self.messages is None) == (self.observation is None)
            or (
                self.settings is not None
                and not isinstance(self.settings, GenerationSettings)
            )
            or (
                self.observation is not None
                and not isinstance(self.observation, Observation)
            )
        ):
            raise ChatError("INVALID_INPUT")
        if self.messages is not None and (
            not isinstance(self.messages, tuple)
            or len(self.messages) % 2 == 0
            or any(
                not isinstance(m, tuple)
                or len(m) != 2
                or m[0] != ("assistant" if i % 2 else "user")
                or not isinstance(m[1], str)
                or not m[1].strip()
                for i, m in enumerate(self.messages)
            )
        ):
            raise ChatError("INVALID_INPUT")


@dataclass(frozen=True, slots=True)
class AuthorizedContextV1:
    """Only an authorized scope and its already-observed text (never the source catalog)."""

    request_id: str
    model_id: str
    binding: ScopeBindingV1
    role: RoleConfig
    observed_texts: tuple[str, ...] = field(repr=False)


class InferenceMemoryProviderV1(Protocol):
    def select(
        self,
        context: AuthorizedContextV1,
    ) -> tuple[tuple[ContextCard, ...], tuple[CardAudit, ...]]: ...


@dataclass(frozen=True, slots=True)
class InferenceResultV1:
    request_id: str
    model_id: str
    role_id: str
    role_version: str
    template_hash: str | None
    template_version: str | None
    renderer_version: str | None
    text: str = field(repr=False)
    finish_reason: FinishReason
    error_code: ErrorCode | None
    input_tokens: int | None
    output_tokens: int | None
    ttft_ms: float | None
    tokens_per_second: float | None
    latency_ms: float
    peak_vram_mb: float | None
    queue_wait_ms: float
    cancellation_count: int
    card_audit: tuple[CardAudit, ...]


@dataclass(frozen=True, slots=True)
class InferenceEventV1:
    kind: Literal["delta", "done", "cancel", "error"]
    request_id: str
    delta: str = field(default="", repr=False)
    result: InferenceResultV1 | None = None


class InferenceHandleV1:
    """Single-consumer stream; cancellation is cooperative and idempotent."""

    def __init__(self, timeout):
        self.request_id = uuid4().hex
        self.started = perf_counter()
        self.deadline = self.started + timeout
        self._events = SimpleQueue()
        self._lock = Lock()
        self._reason: ErrorCode | None = None
        self._finished = False
        self._consumed = False

    def _check_deadline(self):
        if (
            self._reason is None
            and not self._finished
            and perf_counter() >= self.deadline
        ):
            self._reason = "TIMEOUT"
        return self._reason

    def is_set(self):
        with self._lock:
            return self._check_deadline() is not None

    def timeout_pending(self):
        with self._lock:
            return not self._finished and self._check_deadline() == "TIMEOUT"

    def cancel(self) -> bool:
        with self._lock:
            if self._check_deadline() is not None or self._finished:
                return False
            self._reason = "CANCELLED"
            return True

    def _finish(self):
        with self._lock:
            code = self._check_deadline()
            self._finished = True
            return code

    def next_event(self, timeout: float | None = None) -> InferenceEventV1:
        if self._consumed:
            raise StopIteration
        while True:
            event = self._events.get(timeout=timeout)
            if event.kind != "delta":
                self._consumed = True
                return event
            if not self.is_set():
                return event

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_event()

    def close(self):
        self.cancel()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


_AUDIT_REASONS = frozenset(
    {
        "namespace",
        "scope",
        "active",
        "revoked",
        "superseded",
        "read_denied",
        "stale",
        "duplicate_version",
        "not_sent",
        "reserved_content",
        "card_count",
        "card_tokens",
        "card_template",
        "context_budget",
        "memory_disabled",
        "injected",
    }
)


class InferenceServiceV1:
    """One fixed model, one active call, bounded FIFO admission. No session records."""

    def __init__(
        self,
        runtime: ChatRuntime,
        model_id: str,
        bindings: tuple[ScopeBindingV1, ...],
        *,
        memory_provider: InferenceMemoryProviderV1 | None = None,
        max_queued: int = 4,
        timeout_seconds: float = 60,
        max_output_tokens: int = 512,
        defaults: GenerationSettings | None = None,
        test_mode: bool = False,
    ):
        self._runtime = runtime
        self._info = runtime._describe(model_id)
        self._bindings = {b.scope: b for b in bindings}
        if (
            not self._bindings
            or len(self._bindings) != len(bindings)
            or any(b.role_id not in runtime._roles for b in bindings)
            or type(max_queued) is not int
            or not 0 <= max_queued <= 64
            or type(timeout_seconds) not in (int, float)
            or not 0 < timeout_seconds <= 3600
            or type(max_output_tokens) is not int
            or not 1 <= max_output_tokens <= 8192
            or type(test_mode) is not bool
        ):
            raise ChatError("INVALID_CONFIG")
        if (
            (self._info.fixture_only is not False and not test_mode)
            or (isinstance(runtime, MockChatRuntime) and not test_mode)
            or (
                self._info.kind != "adapter"
                and not (test_mode and self._info.fixture_only is True)
            )
        ):
            raise ChatError("ARTIFACT_MISMATCH")
        self.model_id = model_id
        self.defaults = GenerationSettings() if defaults is None else defaults
        self._limit = max_output_tokens
        self.validate_settings(self.defaults)
        self._max_admitted = max_queued + 1
        self._timeout = timeout_seconds
        self._provider = memory_provider
        self._jobs: dict[str, tuple[InferenceHandleV1, object | None, Thread]] = {}
        self._cv = Condition()
        self._lifecycle = Lock()
        self._ready = False
        self._closed = False
        self._start_error: ErrorCode | None = None
        self._versions: dict[str, tuple[str | None, str | None]] = {}

    def validate_settings(self, settings):
        if (
            not isinstance(settings, GenerationSettings)
            or settings.max_new_tokens > self._limit
            or settings.temperature > 2
            or settings.top_k > 2048
            or settings.repetition_penalty > 2
            or (settings.seed is not None and not settings.do_sample)
        ):
            raise ChatError("INVALID_INPUT")

    def binding(self, scope):
        if not valid_id(scope) or scope not in self._bindings:
            raise ChatError("FORBIDDEN")
        return self._bindings[scope]

    @property
    def scopes(self):
        return frozenset(self._bindings)

    def _verify_binding(self, binding):
        return self._runtime._backend.verify_inference_v1(
            self._runtime._roles[binding.role_id],
            structured=binding.input_kind == "observation",
        )

    def _activate(self):
        backend = self._runtime._backend
        backend.activate(self.model_id)
        try:
            return {b.scope: self._verify_binding(b) for b in self._bindings.values()}
        except BaseException:
            try:
                backend.unload()
            except Exception:  # noqa: S110
                pass
            raise

    def start(self):
        """Warm and check the sole pinned backend. A failed start needs a new service."""
        with self._lifecycle:
            if self._closed:
                raise ChatError("SERVICE_UNAVAILABLE")
            if self._start_error is not None:
                raise ChatError(self._start_error)
            if self._ready:
                return
            runtime = self._runtime
            with runtime._model_lock:
                if runtime._inference_pin is not None:
                    self._start_error = "DEVICE_BUSY"
                    raise ChatError(self._start_error)
                try:
                    self._versions = self._activate()
                except (ChatError, MemoryError) as exc:
                    self._start_error = (
                        exc.code if isinstance(exc, ChatError) else "OUT_OF_MEMORY"
                    )
                except Exception:
                    self._start_error = "LOAD_FAILED"
                if self._start_error is not None:
                    raise ChatError(self._start_error)
                runtime._inference_pin = self.model_id
            with self._cv:
                self._ready = True

    def close(self):
        with self._lifecycle:
            with self._cv:
                if self._closed:
                    return
                self._closed = True
                jobs = tuple(self._jobs.values())
                for handle, _, _ in jobs:
                    handle.cancel()
                self._cv.notify_all()
            for _, _, thread in jobs:
                thread.join()
            if self._ready:
                with self._runtime._model_lock:
                    self._runtime._inference_pin = None
                    try:
                        self._runtime._backend.unload()
                    except Exception:
                        raise ChatError("LOAD_FAILED") from None

    def require_ready(self):
        with self._cv:
            if not self._ready or self._closed:
                raise ChatError(self._start_error or "SERVICE_UNAVAILABLE")

    def submit(
        self, request: InferenceRequestV1, *, stream: bool = False, owner=None
    ) -> InferenceHandleV1:
        if not isinstance(request, InferenceRequestV1) or type(stream) is not bool:
            raise ChatError("INVALID_INPUT")
        if request.model_id != self.model_id:
            raise ChatError("UNKNOWN_MODEL")
        binding = self.binding(request.scope)
        if binding.input_kind == "observation":
            if (
                request.observation is None
                or request.observation.conversation.id != binding.conversation_id
                or request.observation.conversation.mode != binding.conversation_mode
            ):
                raise ChatError("FORBIDDEN")
        elif request.messages is None:
            raise ChatError("FORBIDDEN")
        settings = request.settings or self.defaults
        self.validate_settings(settings)
        handle = InferenceHandleV1(self._timeout)
        with self._cv:
            if not self._ready or self._closed:
                raise ChatError(self._start_error or "SERVICE_UNAVAILABLE")
            if len(self._jobs) >= self._max_admitted:
                raise ChatError("QUEUE_FULL")
            worker = Thread(
                target=self._run,
                args=(handle, request, binding, settings, stream),
                name="lm-trainer-inference",
                daemon=True,
            )
            self._jobs[handle.request_id] = handle, owner, worker
            try:
                worker.start()
            except RuntimeError:
                del self._jobs[handle.request_id]
                raise ChatError("SERVICE_UNAVAILABLE") from None
            self._cv.notify_all()
        return handle

    def generate(self, request: InferenceRequestV1) -> InferenceResultV1:
        with self.submit(request) as handle:
            return handle.next_event().result

    def stream(self, request: InferenceRequestV1) -> InferenceHandleV1:
        return self.submit(request, stream=True)

    def cancel(self, request_id: str, *, owner=None) -> bool:
        with self._cv:
            job = self._jobs.get(request_id)
            if job is None or job[1] != owner:
                return False
            accepted = job[0].cancel()
            self._cv.notify_all()
            return accepted

    def _wait_turn(self, handle):
        with self._cv:
            while not handle.is_set():
                if next(iter(self._jobs)) == handle.request_id:
                    return True
                self._cv.wait(timeout=0.025)
        return False

    def _cards(self, handle, request, binding, role):
        if self._provider is None:
            return (), (), None
        texts = (
            tuple(text for _, text in request.messages)
            if request.messages is not None
            else tuple(
                e.content for e in request.observation.events if e.selection == "text"
            )
        )
        for text in texts:
            check_inference_text(text)
        context = AuthorizedContextV1(
            handle.request_id, self.model_id, binding, role, texts
        )
        cards, entries = self._provider.select(context)
        guard = getattr(self._provider, "prompt_guard", None)
        if (
            not isinstance(cards, tuple)
            or len(cards) > 64
            or not all(isinstance(c, ContextCard) and c.version < 2**63 for c in cards)
            or not isinstance(entries, tuple)
            or len(entries) > 128
            or not all(
                isinstance(e, CardAudit)
                and e.reason in _AUDIT_REASONS - {"injected"}
                and valid_id(e.card_id)
                and type(e.version) is int
                and 0 < e.version < 2**63
                for e in entries
            )
        ):
            raise ChatError("GENERATION_FAILED")
        # A role can be bound to several tenants/groups. A role-wide card alone
        # does not establish permission for this particular inference scope.
        scoped = tuple(c for c in cards if c.scope == "session_only")
        unscoped = tuple(audit(c, "scope") for c in cards if c.scope != "session_only")
        selected, rejected = ManualCardProvider().select(
            scoped, role=role, session_id=binding.scope
        )
        return (
            selected,
            entries + unscoped + rejected,
            None if guard is None else partial(guard, context),
        )

    def _run(self, handle, request, binding, settings, stream):
        trace = Trace(handle.started)
        role = self._runtime._roles[binding.role_id]
        template_version, renderer_version = self._versions[binding.scope]
        took_slot = None
        reason, error, text = "error", "GENERATION_FAILED", ""

        def emit(delta):
            if (
                stream
                and self._info.supports_streaming
                and delta
                and not handle.is_set()
            ):
                if trace.ttft_ms is None:
                    trace.ttft_ms = (perf_counter() - handle.started) * 1000
                handle._events.put(
                    InferenceEventV1("delta", handle.request_id, delta=delta)
                )

        try:
            if self._wait_turn(handle) and self._runtime._acquire_model(handle):
                took_slot = perf_counter()
                try:
                    if not handle.is_set():
                        backend = self._runtime._backend
                        if (
                            not backend.active
                            or backend.active.model_id != self.model_id
                        ) and self._activate() != self._versions:
                            backend.unload()
                            raise ChatError("ARTIFACT_MISMATCH")
                        if not handle.is_set():
                            cards, trace.card_audit, card_guard = self._cards(
                                handle, request, binding, role
                            )
                            if not handle.is_set():
                                text, reason = backend.respond_inference_v1(
                                    role=role,
                                    messages=request.messages,
                                    observation=request.observation,
                                    target_person=binding.target_person,
                                    cards=cards,
                                    card_guard=card_guard,
                                    audit_entries=trace.card_audit,
                                    settings=settings,
                                    builder=self._runtime._builder,
                                    cancel=handle,
                                    emit=emit,
                                    trace=trace,
                                )
                                error = None
                finally:
                    self._runtime._model_lock.release()
        except ChatError as exc:
            error = exc.code
        except MemoryError:
            error = "OUT_OF_MEMORY"
        except BaseException:
            # Even process-control exceptions in a daemon worker must release admission.
            error = "GENERATION_FAILED"
        stopped = handle._finish()
        if stopped is not None:
            reason, error = ("cancel" if stopped == "CANCELLED" else "error"), stopped
        elif reason == "cancel":
            error = "CANCELLED"
        ended = perf_counter()
        result = InferenceResultV1(
            handle.request_id,
            self.model_id,
            role.role_id,
            role.version,
            self._info.template_hash,
            template_version,
            renderer_version,
            text if reason not in ("cancel", "error") else "",
            reason,
            error,
            trace.input_tokens,
            trace.output_tokens,
            trace.ttft_ms,
            trace.tokens_per_second,
            (ended - handle.started) * 1000,
            trace.peak_vram_mb,
            ((took_slot or ended) - handle.started) * 1000,
            int(error in ("CANCELLED", "TIMEOUT")),
            tuple(
                replace(a, reason="not_sent")
                if a.reason == "injected" and trace.input_tokens is None
                else a
                for a in trace.card_audit
            ),
        )
        kind = {"error": "error", "cancel": "cancel"}.get(reason, "done")
        try:
            write_audit(
                request_id=result.request_id,
                status=error_status(error) if error else 200,
                error_code=error,
                model_id=result.model_id,
                role_id=result.role_id,
                role_version=result.role_version,
                template_hash=result.template_hash,
                template_version=template_version,
                renderer_version=renderer_version,
                **result_metrics(result),
                reference_tokens=trace.reference_tokens
                if trace.input_tokens is not None
                else None,
                card_audit=[
                    {"version": a.version, "reason": a.reason}
                    for a in result.card_audit
                ],
            )
        finally:
            with self._cv:
                del self._jobs[handle.request_id]
                self._cv.notify_all()
            handle._events.put(InferenceEventV1(kind, handle.request_id, result=result))


def result_metrics(result):
    values = {
        key: getattr(result, key)
        for key in (
            "input_tokens",
            "output_tokens",
            "ttft_ms",
            "tokens_per_second",
            "latency_ms",
            "peak_vram_mb",
            "queue_wait_ms",
            "cancellation_count",
        )
    }
    return {
        k: None if isinstance(v, float) and not math.isfinite(v) else v
        for k, v in values.items()
    }
