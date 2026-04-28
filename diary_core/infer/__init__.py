"""Inference helpers."""

__all__ = ["DiaryResult", "DiaryRuntime"]


def __getattr__(name):
    if name in __all__:
        from diary_core.infer.diary_runtime import DiaryResult, DiaryRuntime

        return {"DiaryResult": DiaryResult, "DiaryRuntime": DiaryRuntime}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
