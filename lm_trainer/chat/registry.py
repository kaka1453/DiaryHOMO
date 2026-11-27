"""Trusted local registry; no GPU allocation, network download or remote code."""
# ruff: noqa: BLE001,SIM118

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lm_trainer.template_policy import resolve_policy, validate_policy
from lm_trainer.train.artifacts import (
    WEIGHTS,
    base_fingerprint,
    quantization_identity,
    read_json,
    tokenizer_hash,
    verify_artifact,
)
from lm_trainer.train.encoding import conversation_identity
from lm_trainer.train.stages import policy_identity

from .context import ContextBuilder
from .types import ChatError, ModelInfo, RoleConfig, valid_id

_SHA = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    base_path: Path
    template_policy: dict[str, Any]
    context_limit: int
    device: str = "cpu"
    dtype: str = "fp32"
    quantization: str = "none"
    base_revision: str | None = None
    adapter_path: Path | None = None
    manifest_sha256: str | None = None
    fixture_only: bool = False

    def __post_init__(self):
        try:
            policy = validate_policy(self.template_policy)
            if (
                not valid_id(self.model_id)
                or not policy["template_hash"]
                or type(self.context_limit) is not int
                or self.context_limit < 2
                or self.dtype not in ("fp32", "fp16", "bf16")
                or self.quantization not in ("none", "4bit", "8bit")
                or not isinstance(self.device, str)
                or not re.fullmatch(r"cpu|cuda:(0|[1-9][0-9]*)", self.device)
                or (self.device == "cpu" and self.quantization != "none")
                or (self.quantization == "4bit" and self.dtype != "fp16")
                or type(self.fixture_only) is not bool
                or (self.base_revision is not None and not valid_id(self.base_revision))
                or ((self.adapter_path is None) != (self.manifest_sha256 is None))
                or (
                    self.manifest_sha256 is not None
                    and not _SHA.fullmatch(self.manifest_sha256)
                )
            ):
                raise ValueError
            object.__setattr__(self, "template_policy", policy)
            object.__setattr__(self, "base_path", Path(self.base_path).absolute())
            if self.adapter_path is not None:
                object.__setattr__(
                    self, "adapter_path", Path(self.adapter_path).absolute()
                )
        except (ValueError, TypeError, OSError):
            raise ChatError("INVALID_CONFIG") from None

    def info(self, *, reason=None):
        return ModelInfo(
            self.model_id,
            "adapter" if self.adapter_path is not None else "base",
            reason is None,
            self.fixture_only,
            self.template_policy["template_hash"],
            reason,
        )


@dataclass(slots=True)
class VerifiedModel:
    spec: ModelSpec
    tokenizer: Any
    policy: dict[str, Any]
    context_limit: int
    warmup_ids: list[int]
    conversation_renderer: dict[str, str] | None = None


def _validate_base_headers(base, fingerprint):
    from safetensors import safe_open

    index = fingerprint["weight_index_sha256"]
    weight_map = (
        read_json(base / "model.safetensors.index.json")["weight_map"]
        if index
        else None
    )
    by_shard = {}
    if weight_map is not None:
        for key, filename in weight_map.items():
            by_shard.setdefault(filename, set()).add(key)
    for filename in fingerprint["weight_files"]:
        with safe_open(base / filename, framework="pt", device="cpu") as weights:
            keys = set(weights.keys())
            if not keys or (weight_map is not None and keys != by_shard[filename]):
                raise ValueError


def _validate_adapter_header(path, r, targets):
    from safetensors import safe_open

    by_module = {}
    matched_targets = set()
    with safe_open(path / WEIGHTS, framework="pt", device="cpu") as weights:
        for key in weights.keys():
            match = re.fullmatch(r"(base_model\.model\..+)\.lora_([AB])\.weight", key)
            module = match[1].removeprefix("base_model.model.") if match else ""
            matches = {t for t in targets if module == t or module.endswith("." + t)}
            tensor = weights.get_slice(key)
            shape = tensor.get_shape()
            if (
                match is None
                or not matches
                or len(shape) != 2
                or min(shape) < 1
                or tensor.get_dtype() not in ("F32", "F16", "BF16")
                or shape[0 if match[2] == "A" else 1] != r
            ):
                raise ValueError
            by_module.setdefault(match[1], set()).add(match[2])
            matched_targets.update(matches)
    if (
        not by_module
        or any(pair != {"A", "B"} for pair in by_module.values())
        or matched_targets != set(targets)
    ):
        raise ValueError


class ModelRegistry:
    def __init__(
        self,
        specs: list[ModelSpec] | tuple[ModelSpec, ...],
        *,
        allow_fixture: bool = False,
    ):
        self._specs = {spec.model_id: deepcopy(spec) for spec in specs}
        if len(self._specs) != len(specs) or type(allow_fixture) is not bool:
            raise ChatError("INVALID_CONFIG")
        self._allow_fixture = allow_fixture

    def spec(self, model_id: str) -> ModelSpec:
        if not valid_id(model_id) or model_id not in self._specs:
            raise ChatError("UNKNOWN_MODEL")
        return deepcopy(self._specs[model_id])

    def list_models(self) -> list[ModelInfo]:
        result = []
        for spec in self._specs.values():
            try:
                self.verify(spec.model_id)
            except ChatError as exc:
                result.append(spec.info(reason=exc.code))
            else:
                result.append(spec.info())
        return result

    def verify(self, model_id: str) -> VerifiedModel:
        spec = self.spec(model_id)
        if spec.fixture_only and not self._allow_fixture:
            raise ChatError("ARTIFACT_MISMATCH")
        base, adapter = spec.base_path, spec.adapter_path
        if (
            not base.is_dir()
            or not (base / "config.json").is_file()
            or (base / "adapter_config.json").exists()  # HF can auto-detect PEFT here
            or not (
                (base / "model.safetensors").is_file()
                or (base / "model.safetensors.index.json").is_file()
            )
            or (
                adapter is not None
                and not all(
                    (adapter / f).is_file()
                    for f in ("training_manifest.json", "adapter_config.json", WEIGHTS)
                )
            )
        ):
            raise ChatError("MODEL_UNAVAILABLE")
        try:
            fingerprint = base_fingerprint(
                {"model_name_or_path": base, "base_revision": spec.base_revision}
            )
        except Exception:
            raise ChatError("MODEL_UNAVAILABLE") from None
        manifest = None
        if adapter is not None:
            try:
                manifest, _ = verify_artifact(
                    adapter,
                    expected_sha256=spec.manifest_sha256,
                    allow_fixture=self._allow_fixture,
                )
                if (
                    manifest["base_model"] != fingerprint
                    or manifest["fixture_only"] != spec.fixture_only
                    or manifest["quantization"]
                    != quantization_identity({"quantization_mode": spec.quantization})
                    or manifest["compute_dtype"] != spec.dtype
                ):
                    raise ValueError
                cfg = read_json(adapter / "adapter_config.json")
                if any(
                    cfg.get(k)
                    for k in (
                        "base_model_name_or_path",
                        "auto_mapping",
                        "use_qalora",
                        "use_bdlora",
                        "alora_invocation_tokens",
                        "arrow_config",
                        "layer_replication",
                        "eva_config",
                        "corda_config",
                        "loftq_config",
                        "lora_ga_config",
                        "velora_config",
                        "monteclora_config",
                        "kasa_config",
                        "megatron_config",
                        "ensure_weight_tying",
                    )
                ):
                    raise ValueError
                _validate_adapter_header(
                    adapter,
                    manifest["lora_config"]["r"],
                    manifest["lora_config"]["target_modules"],
                )
            except Exception:
                raise ChatError("ARTIFACT_MISMATCH") from None
        try:
            import torch
            from transformers import AutoConfig, AutoTokenizer

            if (
                spec.device.startswith("cuda:")
                and int(spec.device[5:]) >= torch.cuda.device_count()
            ):
                raise ValueError
            kw = {"trust_remote_code": False, "local_files_only": True}
            _validate_base_headers(base, fingerprint)
            config = AutoConfig.from_pretrained(base, **kw)
            tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, **kw)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if not tokenizer.is_fast or tokenizer.pad_token_id is None:
                raise ValueError
            native_limit = getattr(config, "max_position_embeddings", None)
            if type(native_limit) is not int or native_limit < 2:
                raise ValueError
        except Exception:
            raise ChatError("MODEL_UNAVAILABLE") from None
        try:
            template, resolved = resolve_policy(tokenizer, spec.template_policy)
            ContextBuilder.verify_template(tokenizer, template, resolved)
            warmup = ContextBuilder.encode(
                tokenizer, resolved, [("system", "Warmup."), ("user", "Ready?")]
            )
            limit = min(spec.context_limit, native_limit, tokenizer.model_max_length)
            if len(warmup) >= limit:
                raise ValueError
        except Exception:
            raise ChatError("TEMPLATE_UNSUPPORTED") from None
        renderer = None
        if manifest is not None:
            try:
                adapter_tok = AutoTokenizer.from_pretrained(
                    adapter, use_fast=True, **kw
                )
                if (
                    tokenizer_hash(tokenizer) != manifest["tokenizer_hash"]
                    or tokenizer_hash(adapter_tok) != manifest["tokenizer_hash"]
                    or policy_identity(
                        tokenizer, {"template_policy": spec.template_policy}
                    )
                    != manifest["template_policy"]
                ):
                    raise ValueError
                if "conversation_renderer" in manifest:
                    renderer = conversation_identity(resolved)
                    if manifest["conversation_renderer"] != renderer:
                        raise ValueError
            except Exception:
                raise ChatError("ARTIFACT_MISMATCH") from None
        return VerifiedModel(spec, tokenizer, resolved, limit, warmup, renderer)


def load_roles(path: Path | str) -> tuple[RoleConfig, ...]:
    """Load only operator-controlled role configuration (never supplied by chat users)."""
    try:
        value = read_json(Path(path))
        if value["schema"] != "lm_trainer.chat-roles.v1" or not isinstance(
            value["roles"], list
        ):
            raise ValueError
        roles = tuple(RoleConfig(**config) for config in value["roles"])
        if not roles or len({role.role_id for role in roles}) != len(roles):
            raise ValueError
        return roles
    except (KeyError, ValueError, TypeError, OSError):
        raise ChatError("INVALID_CONFIG") from None


def load_registry(
    path: Path | str,
    *,
    allow_fixture: bool = False,
    base_override: Path | None = None,
    device_override: str | None = None,
) -> ModelRegistry:
    """Read operator configuration; relative paths are relative to the config file."""
    try:
        path = Path(path).absolute()
        value = read_json(path)
        if value["schema"] != "lm_trainer.chat-models.v1" or not isinstance(
            value["models"], list
        ):
            raise ValueError
        specs = []
        for entry in value["models"]:
            entry = dict(entry)
            entry["base_path"] = (
                base_override
                if base_override is not None
                else path.parent / entry["base_path"]
            )
            if device_override is not None:
                entry["device"] = device_override
            if entry.get("adapter_path") is not None:
                entry["adapter_path"] = path.parent / entry["adapter_path"]
            specs.append(ModelSpec(**entry))
        return ModelRegistry(specs, allow_fixture=allow_fixture)
    except (KeyError, ValueError, TypeError, OSError):
        raise ChatError("INVALID_CONFIG") from None
