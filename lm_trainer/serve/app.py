"""Narrow OpenAI-shaped transport for the versioned local inference service."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from queue import Empty
from time import perf_counter, time
from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, Response, StreamingResponse

from lm_trainer.chat.inference import (
    InferenceRequestV1,
    InferenceServiceV1,
    error_status,
    result_metrics,
    write_audit,
)
from lm_trainer.chat.types import ChatError, valid_id
from lm_trainer.conversation import ConversationError, observe, parse_catalog

_FIELDS = frozenset(
    {
        "model",
        "messages",
        "observation",
        "stream",
        "n",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "seed",
    }
)
_DISCONNECTED = object()
_SOURCE_FIELDS = {"schema", "fixture_only", "conversations"}
_CONVERSATION_FIELDS = {"id", "event_group", "mode", "events", "relations"}
_EVENT_FIELDS = {
    "source_id",
    "position",
    "speaker",
    "person",
    "timestamp",
    "kind",
    "body_available",
    "readable",
    "selection",
    "content",
    "available_for_training",
    "selection_basis",
    "witness_source_ids",
    "visible_from_position",
    "display_name",
}


@dataclass(frozen=True, slots=True)
class APIGrantV1:
    principal_id: str
    allowed_models: frozenset[str]
    allowed_scopes: frozenset[str]

    def __post_init__(self):
        if not valid_id(self.principal_id) or any(
            not isinstance(group, frozenset)
            or not group
            or not all(map(valid_id, group))
            for group in (self.allowed_models, self.allowed_scopes)
        ):
            raise ChatError("INVALID_CONFIG")


def _token_ok(token):
    return (
        isinstance(token, str)
        and 16 <= len(token) <= 512
        and token.isascii()
        and all(33 <= ord(c) <= 126 for c in token)
    )


def _authorize(request, credentials):
    auth, api_keys = (
        request.headers.getlist("authorization"),
        request.headers.getlist("x-api-key"),
    )
    if len(auth) + len(api_keys) != 1:
        raise ChatError("AUTH_REQUIRED")
    if auth:
        scheme, _, token = auth[0].partition(" ")
        if scheme.lower() != "bearer":
            raise ChatError("AUTH_REQUIRED")
    else:
        token = api_keys[0]
    if not _token_ok(token):
        raise ChatError("AUTH_REQUIRED")
    supplied = hashlib.sha256(token.encode("ascii")).digest()
    matching = None
    for digest, grant in credentials:
        if hmac.compare_digest(digest, supplied):
            matching = grant
    if matching is None:
        raise ChatError("AUTH_REQUIRED")
    return matching


def _pairs(pairs):
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("duplicate field")
    return obj


def _number(raw):
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return value


async def _read_json(request, limit):
    if (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        != "application/json"
    ):
        raise ChatError("INVALID_INPUT")
    if request.headers.get("content-encoding", "identity").lower() != "identity":
        raise ChatError("UNSUPPORTED_FIELD")
    lengths = request.headers.getlist("content-length")
    if lengths:
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal():
            raise ChatError("INVALID_INPUT")
        if len(lengths[0]) > 12 or int(lengths[0]) > limit:
            raise ChatError("PAYLOAD_TOO_LARGE")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > limit:
            raise ChatError("PAYLOAD_TOO_LARGE")
        body.extend(chunk)
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_float=_number,
            parse_constant=_number,
        )
    except (ValueError, RecursionError):
        raise ChatError("INVALID_INPUT") from None
    if not isinstance(value, dict):
        raise ChatError("INVALID_INPUT")
    return value


def _check_source_fields(source):
    # Only an HTTP allowlist; shape, relations, provenance and rendering remain Task A's.
    def check(obj, allowed):
        if not isinstance(obj, dict):
            raise ChatError("INVALID_INPUT")
        if set(obj) - allowed:
            raise ChatError("UNSUPPORTED_FIELD")

    check(source, _SOURCE_FIELDS)
    conversations = source.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ChatError("INVALID_INPUT")
    if len(conversations) != 1:
        raise ChatError("FORBIDDEN")
    for conversation in conversations:
        check(conversation, _CONVERSATION_FIELDS)
        if not isinstance(conversation.get("events"), list):
            raise ChatError("INVALID_INPUT")
        for event in conversation["events"]:
            check(event, _EVENT_FIELDS)


def _payload(data, grant, scope, service):
    if set(data) - _FIELDS or (
        "n" in data and (type(data["n"]) is not int or data["n"] != 1)
    ):
        raise ChatError("UNSUPPORTED_FIELD")
    if "model" not in data or not valid_id(data["model"]):
        raise ChatError("INVALID_INPUT")
    if data["model"] != service.model_id:
        raise ChatError("UNKNOWN_MODEL")
    if data["model"] not in grant.allowed_models or scope not in grant.allowed_scopes:
        raise ChatError("FORBIDDEN")
    binding = service.binding(scope)
    stream = data.get("stream", False)
    if type(stream) is not bool or ("messages" in data) == ("observation" in data):
        raise ChatError("INVALID_INPUT")
    if binding.input_kind not in data:
        raise ChatError("FORBIDDEN")
    default = service.defaults
    if "max_tokens" in data and "max_completion_tokens" in data:
        raise ChatError("INVALID_INPUT")
    edits = {
        "max_new_tokens": data.get(
            "max_completion_tokens", data.get("max_tokens", default.max_new_tokens)
        )
    }
    if "temperature" in data:
        value = data["temperature"]
        if type(value) not in (int, float) or not 0 <= value <= 2:
            raise ChatError("INVALID_INPUT")
        edits.update(do_sample=value > 0, temperature=value or default.temperature)
        if value == 0:
            # Greedy explicitly overrides any server sampling seed.
            edits["seed"] = None
    elif "top_p" in data or "seed" in data:
        edits["do_sample"] = True
    if any(k in data for k in ("top_p", "seed")) and edits.get("do_sample") is False:
        raise ChatError("INVALID_INPUT")
    for key in ("top_p", "seed"):
        if key in data:
            if data[key] is None:
                raise ChatError("INVALID_INPUT")
            edits[key] = data[key]
    try:
        settings = replace(default, **edits)
    except ChatError:
        raise ChatError("INVALID_INPUT") from None
    service.validate_settings(settings)

    messages = observation = None
    if binding.input_kind == "messages":
        raw = data["messages"]
        if not isinstance(raw, list):
            raise ChatError("INVALID_INPUT")
        for m in raw:
            if not isinstance(m, dict):
                raise ChatError("INVALID_INPUT")
            if set(m) - {"role", "content"}:
                raise ChatError("UNSUPPORTED_FIELD")
            if m.get("role") in ("system", "developer"):
                raise ChatError("FORBIDDEN")
            if m.get("role") not in ("user", "assistant"):
                raise ChatError("UNSUPPORTED_FIELD")
        messages = tuple((m["role"], m.get("content")) for m in raw)
    else:
        raw = data["observation"]
        if (
            not isinstance(raw, dict)
            or set(raw) != {"source", "observed_ids"}
            or not isinstance(raw["observed_ids"], list)
        ):
            raise ChatError("INVALID_INPUT")
        _check_source_fields(raw["source"])
        try:
            catalog = parse_catalog(raw["source"])
            if set(catalog.conversations) != {binding.conversation_id}:
                raise ChatError("FORBIDDEN")
            observation = observe(catalog, binding.conversation_id, raw["observed_ids"])
        except (
            ConversationError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
        ):
            raise ChatError("INVALID_INPUT") from None
    return InferenceRequestV1(
        data["model"], scope, messages, observation, settings
    ), stream


def _headers(request_id):
    return {"X-Request-ID": request_id, "Cache-Control": "no-store"}


def _error(code, request_id):
    return {
        "error": {
            "code": code,
            "message": str(ChatError(code)),
            "request_id": request_id,
        }
    }


def _rejection(code, request_id, *, log=True):
    status = error_status(code)
    if log:
        write_audit(request_id=request_id, status=status, error_code=code)
    headers = _headers(request_id)
    if code == "AUTH_REQUIRED":
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(_error(code, request_id), status_code=status, headers=headers)


def _usage(result):
    a, b = result.input_tokens, result.output_tokens
    return {
        "prompt_tokens": a,
        "completion_tokens": b,
        "total_tokens": None if a is None or b is None else a + b,
    }


def _early_timeout(handle):
    return {
        **_error("TIMEOUT", handle.request_id),
        "usage": {
            k: None for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "metrics": {
            **dict.fromkeys(
                (
                    "input_tokens",
                    "output_tokens",
                    "ttft_ms",
                    "tokens_per_second",
                    "peak_vram_mb",
                    "queue_wait_ms",
                )
            ),
            "latency_ms": (perf_counter() - handle.started) * 1000,
            "cancellation_count": 1,
        },
    }


def _finish(result):
    return "length" if result.finish_reason == "length" else "stop"


async def _next(handle):
    while True:
        try:
            return await asyncio.to_thread(handle.next_event, 0.025)
        except Empty:
            if handle.timeout_pending():
                raise ChatError("TIMEOUT") from None


async def _until_disconnect(work, receive):
    async def disconnected():
        while (await receive())["type"] != "http.disconnect":
            pass

    tasks = operation, watcher = (
        asyncio.create_task(work),
        asyncio.create_task(disconnected()),
    )
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            watcher.result()
            return _DISCONNECTED
        return operation.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _frame(content):
    return (
        "data: "
        + json.dumps(content, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        + "\n\n"
    )


class _SSE(StreamingResponse):
    def __init__(self, handle, metadata):
        self._handle = handle
        super().__init__(
            self._chunks(metadata),
            media_type="text/event-stream",
            headers={**_headers(handle.request_id), "X-Accel-Buffering": "no"},
        )

    async def __call__(self, scope, receive, send):
        try:
            await _until_disconnect(self.stream_response(send), receive)
        except (ClientDisconnect, OSError):
            pass
        finally:
            self._handle.cancel()

    async def _chunks(self, metadata):
        def frame(delta, finish=None, **extra):
            return _frame(
                {
                    **metadata,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    **extra,
                }
            )

        handle = self._handle
        yield frame({"role": "assistant"})
        any_delta = False
        while True:
            try:
                event = await _next(handle)
            except ChatError:
                yield _frame(_early_timeout(handle))
                return
            if event.kind == "delta":
                any_delta = True
                yield frame({"content": event.delta})
                continue
            result = event.result
            if result.error_code:
                yield _frame(
                    {
                        **_error(result.error_code, handle.request_id),
                        "usage": _usage(result),
                        "metrics": result_metrics(result),
                    }
                )
                return
            yield frame(
                {} if any_delta or not result.text else {"content": result.text},
                _finish(result),
                usage=_usage(result),
                metrics=result_metrics(result),
            )
            yield "data: [DONE]\n\n"
            return


def create_app(
    service: InferenceServiceV1,
    credentials: dict[str, APIGrantV1],
    *,
    max_body_bytes: int = 256 * 1024,
    body_timeout_seconds: float = 10,
):
    """Create the optional local app; caller supplies trusted configuration, never request data."""
    if (
        type(max_body_bytes) is not int
        or not 1 <= max_body_bytes <= 4 * 1024 * 1024
        or type(body_timeout_seconds) not in (int, float)
        or not 0 < body_timeout_seconds <= 60
        or not credentials
        or any(
            not _token_ok(token)
            or not isinstance(grant, APIGrantV1)
            or not grant.allowed_scopes <= service.scopes
            for token, grant in credentials.items()
        )
        or len({grant.principal_id for grant in credentials.values()})
        != len(credentials)
    ):
        raise ChatError("INVALID_CONFIG")
    digests = tuple(
        (hashlib.sha256(token.encode("ascii")).digest(), grant)
        for token, grant in credentials.items()
    )

    @asynccontextmanager
    async def lifespan(_app):
        try:
            await asyncio.to_thread(service.start)
        except ChatError as exc:
            write_audit(
                request_id=uuid4().hex,
                status=error_status(exc.code),
                error_code=exc.code,
            )
        try:
            yield
        finally:
            try:
                await asyncio.to_thread(service.close)
            except ChatError as exc:
                write_audit(
                    request_id=uuid4().hex,
                    status=error_status(exc.code),
                    error_code=exc.code,
                )

    app = FastAPI(
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        redirect_slashes=False,
    )

    @app.get("/v1/models")
    async def models(request: Request):
        request_id = uuid4().hex
        try:
            grant = _authorize(request, digests)
            service.require_ready()
        except ChatError as exc:
            return _rejection(exc.code, request_id)
        data = (
            [{"id": service.model_id, "object": "model", "owned_by": "local"}]
            if service.model_id in grant.allowed_models
            else []
        )
        return JSONResponse(
            {"object": "list", "data": data}, headers=_headers(request_id)
        )

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        request_id = uuid4().hex
        try:
            grant = _authorize(request, digests)
            scopes = request.headers.getlist("x-inference-scope")
            if (
                len(scopes) != 1
                or scopes[0] not in grant.allowed_scopes
                or service.model_id not in grant.allowed_models
            ):
                raise ChatError("FORBIDDEN")
            service.require_ready()
            async with asyncio.timeout(body_timeout_seconds):
                data = await _read_json(request, max_body_bytes)
            job, stream = _payload(data, grant, scopes[0], service)
            handle = service.submit(job, stream=stream, owner=grant.principal_id)
        except ChatError as exc:
            return _rejection(exc.code, request_id)
        except TimeoutError:
            return _rejection("TIMEOUT", request_id)
        except ClientDisconnect:
            write_audit(request_id=request_id, status=499, error_code="CANCELLED")
            return Response(status_code=499, headers=_headers(request_id))
        metadata = {
            "id": "chatcmpl-" + handle.request_id,
            "created": int(time()),
            "model": service.model_id,
            "object": "chat.completion.chunk" if stream else "chat.completion",
        }
        if stream:
            return _SSE(handle, metadata)
        try:
            event = await _until_disconnect(_next(handle), request.receive)
            if event is _DISCONNECTED:
                return Response(status_code=499, headers=_headers(handle.request_id))
            result = event.result
            if result.error_code:
                return JSONResponse(
                    {
                        **_error(result.error_code, handle.request_id),
                        "usage": _usage(result),
                        "metrics": result_metrics(result),
                    },
                    status_code=error_status(result.error_code),
                    headers=_headers(handle.request_id),
                )
            return JSONResponse(
                {
                    **metadata,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": result.text},
                            "finish_reason": _finish(result),
                        }
                    ],
                    "usage": _usage(result),
                    "metrics": result_metrics(result),
                },
                headers=_headers(handle.request_id),
            )
        except ChatError:
            return JSONResponse(
                _early_timeout(handle),
                status_code=504,
                headers=_headers(handle.request_id),
            )
        finally:
            handle.cancel()

    @app.delete("/v1/requests/{target_id}")
    async def cancel(request: Request, target_id: str):
        request_id = uuid4().hex
        try:
            grant = _authorize(request, digests)
            if service.model_id not in grant.allowed_models:
                raise ChatError("FORBIDDEN")
            if not service.cancel(target_id, owner=grant.principal_id):
                raise ChatError("REQUEST_NOT_FOUND")
        except ChatError as exc:
            return _rejection(exc.code, request_id)
        return Response(status_code=202, headers=_headers(request_id))

    return app
