"""Task D local Gradio workbench.

Consumes the Task C API only (``ChatRuntime`` or ``MockChatRuntime``) through a
single ``RuntimeProvider`` injection point. The module never imports a model
backend, never renders a chat template and never checks card permissions.
"""

from lm_trainer.webui.app import (
    UIState,
    build_demo,
    create_app,
    default_runtime_provider,
    launch_args,
)

__all__ = [
    "UIState",
    "build_demo",
    "create_app",
    "default_runtime_provider",
    "launch_args",
]
