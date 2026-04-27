from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


@dataclass
class SizedLoader:
    num_batches: int

    def __len__(self):
        return self.num_batches


def compute_schedule(runtime: dict, train_loader) -> dict[str, int]:
    num_update_steps_per_epoch = max(
        1,
        math.ceil(len(train_loader) / runtime["gradient_accumulation_steps"]),
    )
    total_possible_steps = num_update_steps_per_epoch * runtime["epochs"]
    planned_training_steps = min(runtime["max_steps"], total_possible_steps)
    return {
        "num_update_steps_per_epoch": num_update_steps_per_epoch,
        "total_possible_steps": total_possible_steps,
        "planned_training_steps": planned_training_steps,
    }


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
    print(json.dumps(compute_schedule(runtime, SizedLoader(args.num_batches)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
