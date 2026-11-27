from __future__ import annotations

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
    planned_epochs = getattr(
        getattr(train_loader, "batch_sampler", None),
        "planned_epochs",
        runtime["epochs"],
    )
    total_possible_steps = num_update_steps_per_epoch * planned_epochs
    planned_training_steps = min(runtime["max_steps"], total_possible_steps)
    if runtime.get("init_adapter") and runtime["warmup_steps"] > planned_training_steps:
        raise ValueError("warmup_steps exceeds actual planned optimizer updates.")
    return {
        "planned_epochs": planned_epochs,
        "num_update_steps_per_epoch": num_update_steps_per_epoch,
        "total_possible_steps": total_possible_steps,
        "planned_training_steps": planned_training_steps,
    }
