__all__ = ["simulate_pipeline", "train"]


def __getattr__(name):
    if name in __all__:
        from diary_core.train.engine import simulate_pipeline, train

        return {"simulate_pipeline": simulate_pipeline, "train": train}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
