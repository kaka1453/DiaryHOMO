from __future__ import annotations

import argparse
import json
import os
import random

import torch


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducibility smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_reproducibility(args.seed)
    print(json.dumps({"seed": args.seed, "cuda": torch.cuda.is_available()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
