"""Lifecycle and manifest state used by the shared training loop."""

from __future__ import annotations

import hashlib
import secrets
from pathlib import Path

from lm_trainer.template_policy import public_policy, resolve_policy
from lm_trainer.train.artifacts import (
    ADAPTER,
    CHECKPOINT,
    FORMAT,
    SCHEMA,
    base_fingerprint,
    digest,
    lora_identity,
    quantization_identity,
    require_compatible,
    tokenizer_hash,
    verify_artifact,
)
from lm_trainer.train.schema import preflight_data


def artifact_reference(runtime):
    key = {"continue_stage": "parent", "resume": "resume_from"}.get(
        runtime["init_adapter"]
    )
    if key is None:
        return None, None, None
    path = Path(runtime[key])
    kind = CHECKPOINT if key == "resume_from" else ADAPTER
    manifest, sha = verify_artifact(
        path, kind=kind, allow_fixture=runtime["data_mode"] == "fixture"
    )
    if manifest["fixture_only"] != (runtime["data_mode"] == "fixture"):
        raise ValueError("Stage and input artifact fixture scope do not match.")
    for field, expected in {
        "lora_config": lora_identity(runtime),
        "quantization": quantization_identity(runtime),
        "backend": runtime["backend"],
        "compute_dtype": runtime["compute_dtype"],
    }.items():
        if manifest[field] != expected:
            raise ValueError(f"Incompatible parent/checkpoint {field}.")
    return path, manifest, sha


def dev_preflight(runtime, validated):
    if not runtime.get("dev_manifest_path"):
        return None
    dev = preflight_data(
        {
            **runtime,
            "manifest_path": runtime["dev_manifest_path"],
            "sources": None,
            "target_loss_token_share": "natural",
            "token_budget": None,
        },
        split="dev",
    )
    if (
        validated.events & dev.events
        or validated.origin_ids & dev.origin_ids
        or validated.plan["target_person"] != dev.plan["target_person"]
    ):
        raise ValueError("Dev target mismatch or train/dev event or source_id overlap.")
    return dev


def stage_settings(runtime):
    excluded = {
        "parent",
        "resume_from",
        "init_adapter",
        "stop_after_updates",
        "parent_stage",
        "checkpoint_dir",
        "model_name_or_path",
        "tokenizer_name_or_path",
        "manifest_path",
        "dev_manifest_path",
        "eval_file_path",  # legacy asset-root metadata, never part of a stage report
        "train_config_path",
        "model_config_path",
        "data_file_path",
    }
    return {k: v for k, v in runtime.items() if k not in excluded}


def stage_config_identity(runtime):
    return digest(stage_settings(runtime))


def policy_identity(tokenizer, runtime):
    if runtime.get("template_policy") is None:
        template = getattr(tokenizer, "chat_template", None)
        policy = None
        version = "lm_trainer.no-policy.v1"
        template_sha = (
            hashlib.sha256(template.encode()).hexdigest()
            if isinstance(template, str)
            else digest(template)
            if template
            else None
        )
    else:
        _, resolved = resolve_policy(tokenizer, runtime["template_policy"])
        policy = public_policy(resolved)
        version = policy["version"]
        template_sha = policy["template_hash"]
    return {
        "version": version,
        "template_hash": template_sha,
        "policy_sha256": digest(policy),
        "policy": policy,
    }


class Stage:
    def __init__(self, runtime, validated, dev, environment, *, reference=None):
        self.runtime = runtime
        self.path, self.reference, self.reference_sha = (
            reference if reference is not None else artifact_reference(runtime)
        )
        self.resume = runtime["init_adapter"] == "resume"
        self.dev = dev
        plan = validated.plan
        settings = stage_settings(runtime)
        settings["template_policy"] = public_policy(settings.get("template_policy"))
        self.manifest = {
            "schema": SCHEMA,
            "adapter_format": FORMAT,
            "run_id": self.reference["run_id"]
            if self.resume
            else secrets.token_hex(16),
            "stage_name": runtime["name"],
            "objective": plan["objective"],
            "init_adapter": runtime["init_adapter"],
            "seed": runtime["seed"],
            "base_model": base_fingerprint(runtime),
            "lora_config": lora_identity(runtime),
            "quantization": quantization_identity(runtime),
            "compute_dtype": runtime["compute_dtype"],
            "backend": runtime["backend"],
            "data_release_id": plan["release_id"],
            "manifest_sha256": plan["manifest_sha256"],
            "split": "train",
            "data_mode": plan["data_mode"],
            "fixture_only": plan["fixture_only"],
            "parent_adapter_sha256": self.reference["parent_adapter_sha256"]
            if self.resume
            else self.reference_sha,
            "resume_checkpoint_sha256": self.reference_sha if self.resume else None,
            "stage_config_sha256": stage_config_identity(runtime),
            "training_settings": settings,
            "environment": environment,
        }

    def identify_tokenizer(self, tokenizer):
        self.manifest.update(
            tokenizer_hash=tokenizer_hash(tokenizer),
            template_policy=policy_identity(tokenizer, self.runtime),
        )
        if self.reference:
            require_compatible(self.reference, self.manifest)

    def identify_plan(self, loader, schedule):
        self.identity = {
            k: self.manifest[k]
            for k in (
                "run_id",
                "stage_name",
                "objective",
                "seed",
                "base_model",
                "tokenizer_hash",
                "template_policy",
                "lora_config",
                "quantization",
                "compute_dtype",
                "backend",
                "data_release_id",
                "manifest_sha256",
                "split",
                "fixture_only",
                "stage_config_sha256",
                "parent_adapter_sha256",
                "environment",
            )
        }
        self.identity.update(
            {
                "schedule": schedule,
                "sampler_plan_sha256": loader.batch_sampler.state_dict()["plan_key"],
                "dev_manifest_sha256": self.dev.plan["manifest_sha256"]
                if self.dev
                else None,
            }
        )
        self.manifest["resume_identity_sha256"] = digest(self.identity)
        if (
            self.resume
            and self.reference.get("resume_identity_sha256")
            != self.manifest["resume_identity_sha256"]
        ):
            raise ValueError(
                "Resume configuration, data, schedule or environment changed."
            )

    def identify_model(self, model):
        from lm_trainer.train.modeling import (
            adapter_tensors,
            check_trainable,
            frozen_fingerprint,
            tensor_fingerprint,
        )

        self.manifest["trainable_parameters"] = check_trainable(model, self.runtime)
        self.loaded_fingerprint = tensor_fingerprint(adapter_tensors(model))
        self.base_parameters = frozen_fingerprint(model)
        self.versions = {
            n: p._version for n, p in model.named_parameters() if not p.requires_grad
        }
        if self.reference and (
            self.loaded_fingerprint != self.reference["adapter_weights_sha256"]
            or self.base_parameters != self.reference.get("frozen_parameter_sha256")
        ):
            raise ValueError(
                "Starting LoRA or frozen base tensor fingerprint mismatch."
            )
        self.manifest.update(
            start_adapter_sha256=self.reference["start_adapter_sha256"]
            if self.resume
            else self.loaded_fingerprint,
            loaded_adapter_sha256=self.loaded_fingerprint,
            frozen_parameter_sha256=self.base_parameters,
        )

    def artifact_data(self, model, updates, exposure, projected, reason, *, full_check):
        from lm_trainer.train.modeling import (
            adapter_tensors,
            check_trainable,
            frozen_fingerprint,
            tensor_fingerprint,
        )

        check_trainable(model, self.runtime)
        if self.versions != {
            n: p._version for n, p in model.named_parameters() if not p.requires_grad
        }:
            raise RuntimeError("Frozen base parameter version changed during training.")
        if full_check and frozen_fingerprint(model) != self.base_parameters:
            raise RuntimeError("Frozen base parameter bytes changed during training.")
        current = tensor_fingerprint(adapter_tensors(model))
        if (
            reason not in ("checkpoint_interval", "paused")
            and current == self.manifest["start_adapter_sha256"]
        ):
            raise RuntimeError("Training did not modify LoRA weights.")
        if full_check and base_fingerprint(self.runtime) != self.manifest["base_model"]:
            raise RuntimeError("Base files changed during training.")
        if self.path and full_check:
            verify_artifact(
                self.path,
                kind=CHECKPOINT if self.resume else ADAPTER,
                allow_fixture=self.manifest["fixture_only"],
                expected_sha256=self.reference_sha,
            )
        return {
            **self.manifest,
            "optimizer_updates": len(updates),
            "stop_reason": reason,
            "adapter_weights_sha256": current,
            "planned_vs_actual_tokens": {
                "planned": projected,
                "actual": exposure.report(),
            },
        }
