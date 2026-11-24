"""Compatibility interface for lm_trainer.train.bootstrap."""

from lm_trainer.train.bootstrap import (
    initialize_backend,
    UNSLOTH_VERSIONS,
)
import os
from lm_trainer.train import bootstrap as core


def dispatch_unsloth(runtime, argv):
    variable = (
        "BOA_UNSLOTH_PYTHON"
        if os.environ.get("BOA_UNSLOTH_PYTHON")
        else "LM_TRAINER_UNSLOTH_PYTHON"
    )
    return core.dispatch_unsloth(runtime, argv, interpreter_variable=variable)
