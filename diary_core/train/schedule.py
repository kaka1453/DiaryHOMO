"""Compatibility interface for lm_trainer.train.schedule."""

from lm_trainer.train.schedule import (
    SizedLoader,
    compute_schedule,
)

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Training schedule smoke test")
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()
    runtime = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
    }
    print(
        json.dumps(
            compute_schedule(runtime, SizedLoader(args.num_batches)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
