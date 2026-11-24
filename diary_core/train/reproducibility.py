"""Compatibility interface for lm_trainer.train.reproducibility."""

from lm_trainer.train.reproducibility import (
    set_reproducibility,
)

import argparse
import json
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducibility smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_reproducibility(args.seed)
    print(
        json.dumps(
            {"seed": args.seed, "cuda": torch.cuda.is_available()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
