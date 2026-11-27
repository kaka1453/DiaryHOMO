"""Versioned, local-only adapter interchange. Imports no model libraries."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from lm_trainer.train.performance import sha256_file

SCHEMA = "lm_trainer.adapter-artifact.v1"
MANIFEST = "training_manifest.json"
ADAPTER = "inference_adapter"
CHECKPOINT = "training_checkpoint"
FORMAT = "peft_lora_safetensors"
WEIGHTS = "adapter_model.safetensors"
STATE = "training_state.pt"
_SHA = re.compile(r"[a-f0-9]{64}")


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def read_json(path, *, payload=None):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate artifact JSON field.")
            result[key] = value
        return result

    def bad_number(_):
        raise ValueError("Nonfinite artifact JSON field.")

    return json.loads(
        path.read_bytes() if payload is None else payload,
        object_pairs_hook=unique,
        parse_constant=bad_number,
    )


def base_fingerprint(runtime):
    """Fingerprint exactly the standard local safetensors layout HF loads."""
    root = Path(runtime["model_name_or_path"])
    single, index = root / "model.safetensors", root / "model.safetensors.index.json"
    if single.is_file():
        names = [single.name]
    elif index.is_file():
        weights = read_json(index).get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Base safetensors index has no weight map.")
        if any(
            not isinstance(n, str) or not re.fullmatch(r"[\w.-]+\.safetensors", n)
            for n in weights.values()
        ):
            raise ValueError(
                "Base index must reference literal local safetensors shards."
            )
        names = sorted(set(weights.values()))
    else:
        raise ValueError(
            "A stage requires local model.safetensors or a safetensors shard index."
        )
    for name in ["config.json", *names]:
        path = root / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError("Missing or empty local base config/weight shard.")
    config = read_json(root / "config.json")
    if config.get("quantization_config"):
        raise ValueError(
            "Stages require an unquantized source base; select runtime quantization."
        )
    weights = {n: sha256_file(root / n) for n in names}
    return {
        "model_type": config.get("model_type"),
        "revision": runtime.get("base_revision"),
        "config_sha256": sha256_file(root / "config.json"),
        "weights_sha256": digest(weights),
        "weight_files": weights,
        "weight_index_sha256": sha256_file(index)
        if index.is_file() and not single.is_file()
        else None,
    }


def tokenizer_hash(tokenizer):
    if not tokenizer.is_fast:
        raise ValueError("Stage tokenizer must expose a fast tokenizer backend.")
    core = json.loads(tokenizer.backend_tokenizer.to_str())
    for key in ("padding", "truncation"):
        core.pop(key, None)  # transient encode options, not part of vocabulary identity
    return digest(
        {
            "backend": core,
            "special_tokens": tokenizer.special_tokens_map,
            "chat_template": tokenizer.chat_template,
            "padding_side": tokenizer.padding_side,
        }
    )


def lora_identity(runtime):
    return {
        key: sorted(runtime[f"lora_{key}"])
        if key == "target_modules"
        else runtime[f"lora_{key}"]
        for key in ("r", "alpha", "dropout", "bias", "target_modules")
    }


def quantization_identity(runtime):
    mode = runtime["quantization_mode"]
    return {
        "mode": mode,
        "implementation": "bitsandbytes" if mode != "none" else None,
        "quant_type": "nf4" if mode == "4bit" else None,
        "double_quant": True if mode == "4bit" else None,
    }


def inventory(directory):
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file() or not path.stat().st_size:
            raise ValueError(
                "Artifact entries must be nonempty regular files, without symlinks."
            )
        if path.name != MANIFEST:
            files[path.name] = sha256_file(path)
    return files


def verify_artifact(
    directory, *, kind=ADAPTER, allow_fixture=False, expected_sha256=None
):
    """Return (verified manifest, its byte SHA-256). Fixture is opt-in.

    Hashes detect corruption, not authorship. Consumers needing provenance pin the
    expected manifest SHA-256 from a trusted registry and use immutable storage.
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or not manifest_path.stat().st_size
    ):
        raise ValueError("Missing or unsafe artifact manifest.")
    payload = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and manifest_sha != expected_sha256:
        raise ValueError("Artifact manifest SHA-256 mismatch.")
    m = read_json(manifest_path, payload=payload)
    if (
        kind not in (ADAPTER, CHECKPOINT)
        or not isinstance(m, dict)
        or (
            m.get("schema") != SCHEMA
            or m.get("artifact_kind") != kind
            or m.get("adapter_format") != FORMAT
        )
    ):
        raise ValueError("Invalid artifact schema, kind or format.")
    for key in ("run_id", "stage_name", "data_release_id", "stop_reason"):
        if not isinstance(m.get(key), str) or not re.fullmatch(
            r"[\w.:-]{1,160}", m[key]
        ):
            raise ValueError(f"Missing or invalid artifact {key}.")
    for key in (
        "tokenizer_hash",
        "manifest_sha256",
        "stage_config_sha256",
        "start_adapter_sha256",
        "adapter_weights_sha256",
        "loaded_adapter_sha256",
        "frozen_parameter_sha256",
        "resume_identity_sha256",
    ):
        if not isinstance(m.get(key), str) or not _SHA.fullmatch(m[key]):
            raise ValueError(f"Missing or invalid artifact {key}.")
    if "data_lineage" in m:
        lineage = m["data_lineage"]
        if (
            not isinstance(lineage, list)
            or not lineage
            or any(
                not isinstance(entry, dict)
                or set(entry) != {"release_id", "manifest_sha256", "parent_version"}
                or not isinstance(entry["release_id"], str)
                or not entry["release_id"]
                or not isinstance(entry["manifest_sha256"], str)
                or not _SHA.fullmatch(entry["manifest_sha256"])
                or (
                    entry["parent_version"] is not None
                    and not isinstance(entry["parent_version"], str)
                )
                for entry in lineage
            )
            or not any(
                entry["release_id"] == m["data_release_id"]
                and entry["manifest_sha256"] == m["manifest_sha256"]
                for entry in lineage
            )
        ):
            raise ValueError("Invalid artifact data lineage.")
    for key in ("parent_adapter_sha256", "resume_checkpoint_sha256"):
        if key not in m or (
            m[key] is not None
            and (not isinstance(m[key], str) or not _SHA.fullmatch(m[key]))
        ):
            raise ValueError(f"Missing or invalid artifact {key}.")
    for key in (
        "base_model",
        "template_policy",
        "lora_config",
        "quantization",
        "planned_vs_actual_tokens",
        "timing",
        "performance_summary",
        "environment",
        "training_settings",
    ):
        if not isinstance(m.get(key), dict) or not m[key]:
            raise ValueError(f"Missing or invalid artifact {key}.")
    if (
        type(m.get("fixture_only")) is not bool
        or m.get("data_mode") != ("fixture" if m["fixture_only"] else "release")
        or (m["fixture_only"] and not allow_fixture)
        or m.get("split") != "train"
        or m.get("init_adapter") not in ("new", "continue_stage", "resume")
        or m.get("objective") not in ("causal_text", "assistant_only_messages")
        or m.get("backend") not in ("hf", "unsloth")
        or m.get("compute_dtype") not in ("fp32", "bf16", "fp16")
        or type(m.get("seed")) is not int
        or type(m.get("optimizer_updates")) is not int
        or m["optimizer_updates"] < 1
    ):
        raise ValueError("Artifact fixture/split/stage/backend gate rejected.")
    if (
        (m["stop_reason"] in ("paused", "checkpoint_interval")) != (kind == CHECKPOINT)
        or (kind == CHECKPOINT and m["backend"] != "hf")
        or (
            (m["init_adapter"] == "resume")
            != (m["resume_checkpoint_sha256"] is not None)
        )
        or (
            m["init_adapter"] == "continue_stage" and m["parent_adapter_sha256"] is None
        )
        or (m["init_adapter"] == "new" and m["parent_adapter_sha256"] is not None)
    ):
        raise ValueError("Artifact kind, status or lineage mismatch.")
    mode = m["quantization"].get("mode")
    if mode not in ("none", "4bit", "8bit") or m[
        "quantization"
    ] != quantization_identity({"quantization_mode": mode}):
        raise ValueError("Artifact quantization strategy is incomplete or unsupported.")
    base, template = m["base_model"], m["template_policy"]
    files = base.get("weight_files")
    if (
        not isinstance(files, dict)
        or not files
        or "revision" not in base
        or "weight_index_sha256" not in base
        or any(
            not re.fullmatch(r"[\w.-]+\.safetensors", n)
            or not isinstance(v, str)
            or not _SHA.fullmatch(v)
            for n, v in files.items()
        )
        or base.get("weights_sha256") != digest(files)
        or not _SHA.fullmatch(str(base.get("config_sha256")))
        or not template.get("version")
        or template.get("policy_sha256") != digest(template.get("policy"))
        or "template_hash" not in template
        or (
            template["template_hash"] is not None
            and not _SHA.fullmatch(str(template["template_hash"]))
        )
    ):
        raise ValueError(
            "Artifact base weight fingerprint or template policy is incomplete."
        )
    listed = m.get("artifacts")
    required = {
        WEIGHTS,
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }
    if kind == CHECKPOINT:
        required.add(STATE)
    if (
        not isinstance(listed, dict)
        or not required <= listed.keys()
        or (kind == ADAPTER and STATE in listed)
        or listed != inventory(directory)
    ):
        raise ValueError("Artifact file inventory or SHA-256 mismatch.")
    _verify_peft(directory, m["lora_config"])
    return m, manifest_sha


def _verify_peft(directory, lora):
    cfg = read_json(directory / "adapter_config.json")
    targets = cfg.get("target_modules")
    if (
        cfg.get("peft_type") != "LORA"
        or cfg.get("task_type") != "CAUSAL_LM"
        or cfg.get("bias") != "none"
        or not isinstance(targets, list)
        or not all(isinstance(t, str) and t for t in targets)
        or any(
            cfg.get(k)
            for k in (
                "modules_to_save",
                "use_dora",
                "use_rslora",
                "rank_pattern",
                "alpha_pattern",
                "lora_bias",
                "target_parameters",
                "layer_replication",
                "trainable_token_indices",
            )
        )
        or lora
        != {
            "r": cfg.get("r"),
            "alpha": cfg.get("lora_alpha"),
            "dropout": cfg.get("lora_dropout"),
            "bias": cfg.get("bias"),
            "target_modules": sorted(targets),
        }
        or type(lora["r"]) is not int
        or lora["r"] <= 0
        or type(lora["alpha"]) is not int
        or lora["alpha"] <= 0
        or type(lora["dropout"]) not in (int, float)
        or not 0 <= lora["dropout"] < 1
        or not targets
        or len(targets) != len(set(targets))
    ):
        raise ValueError("Artifact PEFT LoRA structure mismatch.")


def require_compatible(parent, identity):
    for key in (
        "base_model",
        "tokenizer_hash",
        "template_policy",
        "lora_config",
        "quantization",
        "compute_dtype",
        "backend",
    ):
        if parent[key] != identity[key]:
            raise ValueError(f"Incompatible parent/checkpoint {key}.")
