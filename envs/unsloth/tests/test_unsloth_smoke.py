from __future__ import annotations

import torch
import transformers
import datasets
import peft
import bitsandbytes


def test_core_imports() -> None:
    assert torch.cuda.is_available()
    assert transformers.__version__.startswith("5.5.")
    assert datasets.__version__.startswith("4.3.")
