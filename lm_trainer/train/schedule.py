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
    total_possible_steps = num_update_steps_per_epoch * runtime["epochs"]
    planned_training_steps = min(runtime["max_steps"], total_possible_steps)
    return {
        "num_update_steps_per_epoch": num_update_steps_per_epoch,
        "total_possible_steps": total_possible_steps,
        "planned_training_steps": planned_training_steps,
    }
