"""Private single-device HF runtime. The caller serializes all its model operations."""
# ruff: noqa: BLE001,S110

from __future__ import annotations

import gc
import re
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock
from time import perf_counter
from typing import Any
from weakref import WeakValueDictionary

from .context import INFERENCE_CONTROLS, PROTOCOL, ContextBuilder, check_inference_text
from .registry import ModelRegistry, VerifiedModel
from .types import (
    ActivationInfo,
    CardAudit,
    ChatError,
    FinishReason,
    GenerationSettings,
)

_DEVICE_LOCK = Lock()
_DEVICE_OWNERS = WeakValueDictionary()
_MIB = 1024**2


@dataclass(slots=True)
class Trace:
    started: float
    activation: ActivationInfo | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_ms: float | None = None
    tokens_per_second: float | None = None
    peak_vram_mb: float | None = None
    card_audit: tuple[CardAudit, ...] = ()
    dropped_turns: int | None = None
    reference_tokens: int | None = None


class VisibleStreamer:
    """HF streamer hook, with incremental UTF-8 decoding and a protocol lookbehind.

    Raw chunks never cross the event boundary. Withhold suffixes that could start
    a reserved marker, including markers spelled across multiple ordinary tokens.
    """

    def __init__(
        self,
        tokenizer,
        policy,
        native_eos,
        cancel: Event,
        emit: Callable[[str], None],
        *,
        controls=PROTOCOL,
    ):
        from tokenizers.decoders import DecodeStream

        self.tokenizer = tokenizer.backend_tokenizer
        self.decoder = DecodeStream(skip_special_tokens=False)
        self.cancel = cancel
        self.emit = emit
        self.native_eos = set(native_eos)
        self.stop_ids = set(policy["stop_token_ids"])
        self.bad_ids = set(policy["control_token_ids"]).union(tokenizer.all_special_ids)
        self.bad_ids.discard(tokenizer.unk_token_id)
        self.stops = {s.lower() for s in policy["stop_tokens"]}
        self.blocked = {
            s.lower() for s in (*policy["control_tokens"], *controls)
        } - self.stops
        self.markers = sorted(self.stops | self.blocked, key=lambda p: (-len(p), p))
        self.pattern = re.compile("|".join(map(re.escape, self.markers)), re.IGNORECASE)
        self.prefixes = {m[:i] for m in self.markers for i in range(1, len(m))}
        self.max_prefix = max(map(len, self.prefixes), default=0)
        self.prompt = True
        self.ids: list[int] = []
        self.count = 0
        self.pending = ""
        self.decoded = ""
        self.parts: list[str] = []
        self.reason: FinishReason | None = None
        self.text_stopped = False
        self.ended = False

    def _emit(self, text):
        if text and not self.cancel.is_set():
            self.parts.append(text)
            self.emit(text)

    def _feed(self, chunk):
        self.decoded += chunk
        text = self.pending + chunk
        match = self.pattern.search(text)
        if match is not None:
            self._emit(text[: match.start()])
            self.pending = ""
            if match[0].lower() in self.stops:
                self.reason = "stop"
                self.text_stopped = True
                return
            raise ChatError("UNSAFE_OUTPUT")
        held = next(
            (
                n
                for n in range(min(len(text), self.max_prefix), 0, -1)
                if text[-n:].lower() in self.prefixes
            ),
            0,
        )
        self._emit(text[:-held] if held else text)
        self.pending = text[-held:] if held else ""

    def put(self, values):
        if self.prompt:
            self.prompt = False
            return
        for token in values.flatten().tolist():
            self.count += 1
            if self.cancel.is_set() or self.reason:
                continue
            if token in self.native_eos or token in self.stop_ids:
                self.reason = "eos" if token in self.native_eos else "stop"
                continue
            if token in self.bad_ids:
                raise ChatError("UNSAFE_OUTPUT")
            self.ids.append(token)
            chunk = self.decoder.step(self.tokenizer, token)
            if chunk:
                self._feed(chunk)

    def end(self):
        self.ended = True
        if self.cancel.is_set() or self.text_stopped:
            return
        # Flush an incomplete UTF-8 tail using the backend's ordinary decoding.
        full = self.tokenizer.decode(self.ids, skip_special_tokens=False)
        if not full.startswith(self.decoded):
            raise ChatError("UNSAFE_OUTPUT")
        if full != self.decoded:
            self._feed(full[len(self.decoded) :])
        self._emit(self.pending)
        self.pending = ""

    @property
    def text(self) -> str:
        return "".join(self.parts)


@dataclass(slots=True)
class _Active:
    verified: VerifiedModel
    model: Any
    input_device: Any
    native_eos: tuple[int, ...]
    stats: ActivationInfo


def _backend_error(exc: Exception, fallback):
    import torch

    if isinstance(exc, ChatError):
        return exc.code
    if isinstance(exc, (torch.OutOfMemoryError, MemoryError)):
        return "OUT_OF_MEMORY"
    return fallback


def _eos_ids(model, tokenizer):
    result = set()
    for value in (
        model.generation_config.eos_token_id,
        model.config.eos_token_id,
        tokenizer.eos_token_id,
    ):
        if value is not None:
            result.update([value] if isinstance(value, int) else value)
    return tuple(sorted(result))


def _config(active: _Active, settings: GenerationSettings, *, seeded=False):
    from transformers import GenerationConfig

    return GenerationConfig(
        max_new_tokens=settings.max_new_tokens,
        repetition_penalty=settings.repetition_penalty,
        do_sample=settings.do_sample and not seeded,
        num_beams=1,
        num_return_sequences=1,
        use_cache=True,
        cache_implementation="dynamic",
        return_dict_in_generate=False,
        eos_token_id=sorted(
            set(active.native_eos) | set(active.verified.policy["stop_token_ids"])
        ),
        pad_token_id=active.verified.tokenizer.pad_token_id,
        **(
            {
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "top_k": settings.top_k,
            }
            if settings.do_sample and not seeded
            else {}
        ),
    )


def _seeded_sampler(device, settings):
    """Sample with a private generator, never seed or fork the process RNG."""
    import torch
    from transformers import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

    generator = torch.Generator(device=device).manual_seed(settings.seed)
    warpers = [TemperatureLogitsWarper(float(settings.temperature))]
    if settings.top_k:
        warpers.append(TopKLogitsWarper(settings.top_k))
    if settings.top_p < 1:
        warpers.append(TopPLogitsWarper(float(settings.top_p)))

    def sample(input_ids, scores):
        for warp in warpers:
            scores = warp(input_ids, scores)
        selected = torch.multinomial(scores.softmax(dim=-1), 1, generator=generator)
        # Model.generate's greedy branch then selects exactly the sampled token.
        return torch.full_like(scores, float("-inf")).scatter_(1, selected, 0.0)

    return sample


def _kwargs(torch, device, ids):
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


class HFBackend:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._active: _Active | None = None
        self._claim: str | None = None

    @property
    def active(self):
        return self._active.stats if self._active else None

    def _claim_device(self, device):
        if not device.startswith("cuda:"):
            return
        with _DEVICE_LOCK:
            if device in _DEVICE_OWNERS:
                raise ChatError("DEVICE_BUSY")
            _DEVICE_OWNERS[device] = self
            self._claim = device

    def unload(self):
        self._active = None
        device = self._claim
        try:
            gc.collect()
            if device:
                import torch

                with torch.cuda.device(device):
                    try:
                        torch.cuda.synchronize(device)
                    finally:
                        torch.cuda.empty_cache()
        finally:
            if device:
                with _DEVICE_LOCK:
                    if _DEVICE_OWNERS.get(device) is self:
                        del _DEVICE_OWNERS[device]
                self._claim = None

    def _discard_failed(self):
        try:
            self.unload()
        except Exception:
            # Preserve the original sanitized failure code even if the driver is unhealthy.
            pass

    def activate(self, model_id) -> ActivationInfo:
        if self._active is not None and model_id == self._active.stats.model_id:
            return self._active.stats
        # Validate CPU metadata first; only then release the old model and allocate the new one.
        started = perf_counter()
        verified = self.registry.verify(model_id)
        try:
            if self._active is not None:
                self.unload()
            self._claim_device(verified.spec.device)
            self._active = self._load(verified, started)
            return self._active.stats
        except Exception as exc:
            code = _backend_error(exc, "LOAD_FAILED")
        except BaseException:
            self._discard_failed()
            raise
        # Do not keep a backend exception/traceback alive during CUDA cleanup.
        self._discard_failed()
        raise ChatError(code) from None

    @staticmethod
    def _load(verified, started):
        import torch
        from transformers import AutoModelForCausalLM

        from lm_trainer.model.quantization import build_quantization_config

        spec = verified.spec
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
            spec.dtype
        ]
        qconfig = build_quantization_config(spec.quantization)
        base, loading = AutoModelForCausalLM.from_pretrained(
            spec.base_path,
            dtype=dtype,
            device_map={"": spec.device},
            use_safetensors=True,
            trust_remote_code=False,
            local_files_only=True,
            output_loading_info=True,
            **({"quantization_config": qconfig} if qconfig else {}),
        )
        if any(
            loading.get(k)
            for k in (
                "missing_keys",
                "mismatched_keys",
                "error_msgs",
                "conversion_errors",
            )
        ):
            raise ChatError("LOAD_FAILED")
        if spec.adapter_path is not None:
            from peft import PeftModel, get_peft_model_state_dict
            from safetensors import safe_open

            from lm_trainer.train.artifacts import WEIGHTS

            model = PeftModel.from_pretrained(
                base,
                spec.adapter_path,
                is_trainable=False,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            # PEFT can otherwise warn and leave an adapter partially initialized.
            loaded = get_peft_model_state_dict(model)
            with safe_open(
                spec.adapter_path / WEIGHTS, framework="pt", device="cpu"
            ) as expected:
                if set(loaded) != set(expected.keys()) or any(
                    not torch.equal(
                        t.detach().cpu(), expected.get_tensor(k).to(t.dtype)
                    )
                    for k, t in loaded.items()
                ):
                    raise ChatError("ARTIFACT_MISMATCH")
            del loaded
        else:
            model = base
        model.eval()
        model.requires_grad_(False)
        device = model.get_input_embeddings().weight.device
        if device != torch.device(spec.device):
            raise ChatError("LOAD_FAILED")
        gpu = device.type == "cuda"
        if gpu:
            torch.cuda.synchronize(device)
        warm = perf_counter()
        active = _Active(
            verified,
            model,
            device,
            _eos_ids(model, verified.tokenizer),
            ActivationInfo(spec.model_id, None, None, None),
        )
        with torch.inference_mode():
            model.generate(
                **_kwargs(torch, device, verified.warmup_ids),
                generation_config=_config(active, GenerationSettings(max_new_tokens=1)),
            )
        if gpu:
            torch.cuda.synchronize(device)
        active.stats = ActivationInfo(
            spec.model_id,
            (warm - started) * 1000,
            (perf_counter() - warm) * 1000,
            torch.cuda.memory_allocated(device) / _MIB if gpu else None,
        )
        return active

    def respond(
        self,
        *,
        role,
        history,
        text,
        cards,
        audit_entries,
        settings,
        builder: ContextBuilder,
        cancel,
        emit,
        trace: Trace,
    ) -> tuple[str, FinishReason]:
        ids = self._bind_context(
            role, history, text, cards, audit_entries, settings, builder, trace
        )
        return self._respond_ids(settings, cancel, emit, trace, ids)

    def verify_inference_v1(self, role, *, structured):
        verified = self._active.verified
        expected = verified.policy["system_prompt"]
        if expected not in (None, role.system_prompt):
            raise ChatError("TEMPLATE_UNSUPPORTED")
        if structured and verified.conversation_renderer is None:
            raise ChatError("ARTIFACT_MISMATCH")
        if (
            structured
            and expected != role.system_prompt
            and not verified.spec.fixture_only
        ):
            raise ChatError("TEMPLATE_UNSUPPORTED")
        check_inference_text(
            role.system_prompt,
            tokenizer=verified.tokenizer,
            policy=verified.policy,
            application=True,
        )
        return verified.policy["version"], (
            verified.conversation_renderer["renderer_version"] if structured else None
        )

    def respond_inference_v1(
        self,
        *,
        role,
        messages,
        observation,
        target_person,
        cards,
        audit_entries,
        settings,
        builder,
        cancel,
        emit,
        trace,
        card_guard=None,
    ):
        verified = self._active.verified
        built = builder.build_inference_v1(
            verified.tokenizer,
            verified.policy,
            role=role,
            messages=messages,
            observation=observation,
            target_person=target_person,
            cards=cards,
            audit_entries=audit_entries,
            context_limit=verified.context_limit,
            max_new_tokens=settings.max_new_tokens,
            card_guard=card_guard,
        )
        trace.card_audit = built.card_audit
        trace.reference_tokens = built.injected_tokens
        return self._respond_ids(
            settings, cancel, emit, trace, built.input_ids, inference_v1=True
        )

    def _respond_ids(self, settings, cancel, emit, trace, ids, *, inference_v1=False):
        if cancel.is_set():
            return "", "cancel"
        try:
            if inference_v1:
                return self._generate(
                    settings, cancel, emit, trace, ids, controls=INFERENCE_CONTROLS
                )
            return self._generate(settings, cancel, emit, trace, ids)
        except Exception as exc:
            code = _backend_error(exc, "GENERATION_FAILED")
        except BaseException:
            self._discard_failed()
            raise
        self._discard_failed()
        raise ChatError(code) from None

    def _bind_context(
        self, role, history, text, cards, audit_entries, settings, builder, trace
    ):
        active = self._active
        expected = active.verified.policy["system_prompt"]
        if expected is not None and role.system_prompt != expected:
            raise ChatError("TEMPLATE_UNSUPPORTED")
        built = builder.build(
            active.verified.tokenizer,
            active.verified.policy,
            role=role,
            history=history,
            user_text=text,
            cards=cards,
            audit_entries=audit_entries,
            context_limit=active.verified.context_limit,
            max_new_tokens=settings.max_new_tokens,
        )
        trace.card_audit, trace.dropped_turns = built.card_audit, built.dropped_turns
        # Keep inputs request-local; no tensors or KV cache are stored in _Active.
        # Return the prompt rather than mutating the active model.
        return built.input_ids

    def _generate(self, settings, cancel, emit, trace, ids, *, controls=PROTOCOL):
        import torch
        from transformers import (
            LogitsProcessorList,
            StoppingCriteria,
            StoppingCriteriaList,
        )

        active = self._active
        tok, policy = active.verified.tokenizer, active.verified.policy
        gate = VisibleStreamer(
            tok, policy, active.native_eos, cancel, emit, controls=controls
        )
        device = active.input_device
        gpu = device.type == "cuda"

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full(
                    (input_ids.shape[0],),
                    cancel.is_set() or gate.reason == "stop",
                    dtype=torch.bool,
                    device=input_ids.device,
                )

        seeded = settings.do_sample and settings.seed is not None
        processors = LogitsProcessorList(
            [_seeded_sampler(device, settings)] if seeded else []
        )
        if gpu:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        try:
            with torch.inference_mode():
                inputs = _kwargs(torch, device, ids)
                trace.input_tokens = len(ids)
                output = active.model.generate(
                    **inputs,
                    generation_config=_config(active, settings, seeded=seeded),
                    stopping_criteria=StoppingCriteriaList([Stop()]),
                    streamer=gate,
                    logits_processor=processors,
                )
                if not gate.ended or output.shape[-1] - len(ids) != gate.count:
                    raise ChatError("GENERATION_FAILED")
            if gpu:
                torch.cuda.synchronize(device)
        finally:
            trace.output_tokens = gate.count if trace.input_tokens is not None else None
            if trace.output_tokens:
                trace.tokens_per_second = trace.output_tokens / (perf_counter() - start)
            if gpu:
                trace.peak_vram_mb = torch.cuda.max_memory_allocated(device) / _MIB
        if cancel.is_set():
            return "", "cancel"
        return gate.text, gate.reason or "length"
