"""Select the training backend before importing any model libraries."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

# Tested independent Unsloth environment. The HF runtime has its own requirements.
UNSLOTH_VERSIONS = {
    "unsloth": "2026.9.12",
    "unsloth-zoo": "2026.9.8",
    "torch": "2.12.1+cu130",
    "transformers": "5.5.0",
    "accelerate": "1.15.0",
    "peft": "0.21.1",
    "bitsandbytes": "0.50.2",
    "datasets": "4.3.0",
    "xformers": "0.0.35",
    "torchvision": "0.27.1+cu130",
}


def dispatch_unsloth(
    runtime, argv, *, interpreter_variable="LM_TRAINER_UNSLOTH_PYTHON"
):
    interpreter = os.environ.get(interpreter_variable)
    if runtime["backend"] != "unsloth" or not interpreter:
        return False
    # Start a fresh interpreter even when both venv launchers link to one binary.
    env = os.environ.copy()
    env.pop(interpreter_variable, None)
    env.pop("LM_TRAINER_UNSLOTH_PYTHON", None)
    subprocess.run([interpreter, *argv], env=env, check=True)
    return True


def initialize_backend(runtime):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("本训练器仅支持单进程单卡。")
    if runtime.get("backend", "hf") == "hf":
        if "unsloth" in sys.modules or "unsloth_zoo" in sys.modules:
            raise RuntimeError("HF 需要独立进程，请重新启动 HF 解释器。")
        return
    if runtime["backend"] != "unsloth":
        raise ValueError(f"Unknown backend {runtime['backend']}")
    if "unsloth" not in sys.modules and any(
        p in sys.modules for p in ("transformers", "peft", "trl")
    ):
        raise RuntimeError("Unsloth 必须先导入，请使用独立进程。")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Unsloth 需要独立虚拟环境。")
    for package, expected in UNSLOTH_VERSIONS.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != expected:
            raise RuntimeError(
                f"Unsloth requires {package}=={expected}, found {installed}. Check the selected backend interpreter."
            )
    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
    cache = Path(runtime["checkpoint_dir"]).resolve() / ".cache"
    for key, name in {
        "UNSLOTH_COMPILE_LOCATION": "unsloth",
        "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "HF_HOME": "huggingface",
        "HF_DATASETS_CACHE": "datasets",
        "MPLCONFIGDIR": "matplotlib",
    }.items():
        os.environ.setdefault(key, str(cache / name))
    import unsloth  # noqa: F401  (must precede Transformers and PEFT)
