"""Compatibility interface for lm_trainer.train.performance."""

from lm_trainer.train.performance import (
    sha256_file,
    summarize_updates,
    write_json,
)
from diary_core.config.common import PROJECT_ROOT
from lm_trainer.train import performance as core


def legacy_lock_files():
    return {name: PROJECT_ROOT / name for name in ("uv.lock", "envs/unsloth/uv.lock")}


def environment_metadata():
    return core.environment_metadata(legacy_lock_files())
