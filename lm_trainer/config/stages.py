"""Strict opt-in stage configuration, independent of the legacy YAML defaults."""

import re
from pathlib import Path

from lm_trainer.config.common import asset_path

STAGE_DEFAULTS = {
    "name": None,
    "init_adapter": None,
    "parent": None,
    "resume_from": None,
    "output_dir": None,
    "base_revision": None,
    "tokenizer_name_or_path": None,
    "packing": False,
    "dev_manifest_path": None,
    "dev_probe": False,
    "dev_every": None,
    "save_probe_text": False,
    "stop_after_updates": None,
}
BOUNDED_DEFAULTS = {
    "epochs": 1,
    "batch_size": 1,
    "shuffle": True,
    "auto_pad_batch": True,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 0,
    "save_steps": 100,
    "seed": 42,
    "num_workers": 0,
    "pin_memory": False,
}
STAGE_LOCAL = {
    "name",
    "objective",
    "init_adapter",
    "parent",
    "resume_from",
    "output_dir",
    "checkpoint_dir",
    "sources",
    "target_loss_token_share",
    "lr",
    "max_steps",
    "epochs",
    "warmup_steps",
    "token_budget",
    "max_sample_repeats",
    "max_effective_epochs",
    "stop_after_updates",
}


def identifier(value):
    return isinstance(value, str) and re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}", value
    )


def reject_placeholders(value):
    if isinstance(value, dict):
        for child in value.values():
            reject_placeholders(child)
    elif isinstance(value, list):
        for child in value:
            reject_placeholders(child)
    elif isinstance(value, str) and re.search(r"\bSET_[A-Z0-9_]+\b", value):
        raise ValueError("Replace all SET_ placeholders before training.")


def configure_stage(runtime, root, path_root):
    if runtime["data_mode"] == "legacy":
        raise ValueError("Adapter lifecycle stages require release or fixture data.")
    if not identifier(runtime["name"]):
        raise ValueError("A stage requires a literal name (letters, digits, _, -, .).")
    mode = runtime["init_adapter"]
    if mode not in ("new", "continue_stage", "resume"):
        raise ValueError("init_adapter must be new, continue_stage or resume.")
    if runtime["backend"] == "unsloth" and mode != "new":
        raise ValueError(
            "Unsloth continue_stage/resume unsupported; run the parent and lifecycle in HF."
        )
    if (runtime["parent"] is not None) != (mode == "continue_stage") or (
        runtime["resume_from"] is not None
    ) != (mode == "resume"):
        raise ValueError(
            "continue_stage requires only parent; resume requires only resume_from."
        )
    if runtime["packing"] is not False:
        raise ValueError("packing must be false for stage training.")
    if runtime["lora_bias"] != "none":
        raise ValueError("Stage LoRA requires bias=none to freeze all base parameters.")
    if runtime["base_revision"] is not None and not identifier(
        runtime["base_revision"]
    ):
        raise ValueError(
            "base_revision must be a literal identifier; actual bytes are also hashed."
        )
    if (
        runtime["max_steps"] > 100_000
        or runtime["epochs"] > 1000
        or (runtime["token_budget"] or 0) > 100_000_000
    ):
        raise ValueError(
            "Unreasonable stage budget: max_steps <= 100000, epochs <= 1000, token_budget <= 100000000."
        )
    if not 0 <= runtime["seed"] < 2**32 or runtime["max_length"] < 2:
        raise ValueError("Stage seed must be uint32 and max_length must be at least 2.")
    if runtime["stop_after_updates"] is not None and (
        type(runtime["stop_after_updates"]) is not int
        or not 0 < runtime["stop_after_updates"] < runtime["max_steps"]
        or runtime["backend"] != "hf"
    ):
        raise ValueError(
            "stop_after_updates requires HF and a positive value below max_steps."
        )
    for key in ("dev_probe", "save_probe_text"):
        if type(runtime[key]) is not bool:
            raise ValueError(f"{key} must be boolean.")
    interval = runtime["dev_every"]
    if interval is not None and (
        type(interval) is not int
        or interval < 1
        or not (runtime["dev_probe"] or runtime["dev_manifest_path"])
    ):
        raise ValueError(
            "dev_every requires a positive optimizer interval and an enabled dev hook."
        )
    if runtime["save_probe_text"] and (
        not runtime["dev_probe"] or runtime["data_mode"] != "fixture"
    ):
        raise ValueError("Probe text requires dev_probe and fixture mode.")
    if runtime["dev_probe"] and runtime["objective"] != "assistant_only_messages":
        raise ValueError("The dialogue probe requires the SFT template policy.")
    for key, folders in {
        "parent": ("checkpoints", "runs"),
        "resume_from": ("checkpoints", "runs"),
        "tokenizer_name_or_path": ("assets",),
        "dev_manifest_path": ("data/eval",),
    }.items():
        value = runtime[key]
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path.")
        if key == "parent" and value.startswith("stage:"):
            continue
        if root is None:
            path = path_root / Path(value).expanduser()
            runtime[key] = str(path.absolute())
        else:
            runtime[key] = str(
                asset_path(
                    root,
                    value,
                    key,
                    *folders,
                    directory=key != "dev_manifest_path",
                    required=key not in ("parent", "resume_from"),
                )
            )
    runtime.pop("output_dir")  # canonical output key is checkpoint_dir
    return runtime


def check_stage_paths(stages):
    """Disjoint outputs; only a named earlier final may be inside another output."""
    previous = {}
    for runtime in stages:
        name = runtime["name"]
        if name in previous:
            raise ValueError("Duplicate stage name.")
        parent = runtime["parent"]
        if parent and parent.startswith("stage:"):
            parent_name = parent.removeprefix("stage:")
            if parent_name not in previous:
                raise ValueError(
                    "stage: parent must name an earlier stage in this plan."
                )
            runtime["parent_stage"] = parent_name
            runtime["parent"] = str(previous[parent_name] / "final")
        output = Path(runtime["checkpoint_dir"]).resolve()
        for other in previous.values():
            if output.is_relative_to(other) or other.is_relative_to(output):
                raise ValueError("Stage output directories must be disjoint.")
        for key in (
            "parent",
            "resume_from",
            "model_name_or_path",
            "tokenizer_name_or_path",
            "manifest_path",
            "dev_manifest_path",
        ):
            if runtime.get(key) is None:
                continue
            source = Path(runtime[key]).resolve()
            # A resume is deliberately a fresh continuation directory, not
            # an in-place append or an opportunity to overwrite an old run.
            if output.is_relative_to(source) or source.is_relative_to(output):
                raise ValueError(f"Stage output must not overlap {key}.")
            if key in ("parent", "resume_from") and output.is_relative_to(
                source.parent
            ):
                raise ValueError(
                    "Output must not be inside the referenced artifact's run."
                )
        previous[name] = output
