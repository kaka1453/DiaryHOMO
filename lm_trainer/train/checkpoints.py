"""HF optimizer-boundary state. Only safe torch tensors and plain containers."""

import random


def capture_rng():
    import numpy as np
    import torch

    engine, keys, position, gaussian, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": (
            engine,
            torch.from_numpy(keys.astype("int64")),
            position,
            gaussian,
            cached,
        ),
    }


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG device count mismatch.")
        torch.cuda.set_rng_state_all(state["cuda"])
    elif state["cuda"]:
        raise ValueError("CUDA checkpoint RNG state cannot be restored on CPU.")
    engine, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state(
        (engine, keys.numpy().astype("uint32"), position, gaussian, cached)
    )


def consumed_cursor(loader, schedule, accumulation, updates):
    """Cursor of *committed* batches, never of dispatched/prefetched batches."""
    per_epoch = schedule["num_update_steps_per_epoch"]
    epoch, offset = divmod(updates, per_epoch)
    return {
        **loader.batch_sampler.state_dict(),
        "epoch": epoch,
        "next_batch": offset * accumulation,
    }


def training_state(
    optimizer,
    scheduler,
    accelerator,
    cursor,
    exposure,
    updates,
    identity,
    manifest,
    dev_history,
):
    return {
        "version": 1,
        "identity": identity,
        "run_id": manifest["run_id"],
        "optimizer_updates": len(updates),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "grad_scaler": accelerator.scaler.state_dict()
        if accelerator.scaler is not None
        else None,
        "rng": capture_rng(),
        "sampler": cursor,
        "exposure": exposure.state_dict(),
        "updates": updates,
        "dev_history": dev_history,
    }


def restore_training(
    directory,
    optimizer,
    scheduler,
    accelerator,
    loader,
    exposure,
    identity,
    manifest,
    schedule,
    accumulation,
):
    import torch

    from lm_trainer.train.artifacts import STATE

    state = torch.load(directory / STATE, weights_only=True, map_location="cpu")
    completed = manifest["optimizer_updates"]
    if (
        state["version"] != 1
        or state["identity"] != identity
        or state["run_id"] != manifest["run_id"]
        or state["optimizer_updates"] != completed
        or len(state["updates"]) != completed
        or state["sampler"]
        != consumed_cursor(loader, schedule, accumulation, completed)
        or not 0 < completed < schedule["planned_training_steps"]
        or state["exposure"]["updates"] != completed
        or (state["grad_scaler"] is None) != (accelerator.scaler is None)
    ):
        raise ValueError(
            "Checkpoint identity, schedule, counts, sampler cursor or AMP state mismatch."
        )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if accelerator.scaler is not None:
        accelerator.scaler.load_state_dict(state["grad_scaler"])
    loader.batch_sampler.load_state_dict(state["sampler"])
    exposure.load_state_dict(state["exposure"])
    restore_rng(state["rng"])
    return state["updates"], state["dev_history"]
