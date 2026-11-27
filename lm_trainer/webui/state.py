"""Per-browser UI state.

The browser session owns only IDs and UI options. Authoritative history, model
binding and the model lock live inside Task C. Nothing here is a global history
list, so two browser sessions cannot read each other's conversation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class UIState:
    session_id: str | None = None
    request_id: str | None = None
    model_id: str | None = None
    role_id: str = "CS_AGENT"
    busy: bool = False
    last_error: str | None = None
    last_result: dict | None = None
    # Explicit, independent authorizations (default off).
    save_enabled: bool = False
    debug_enabled: bool = False
    save_dir: str = ""
    debug_log: list[str] = field(default_factory=list)
    # Card selection IDs (the runtime decides what actually gets injected).
    selected_cards: tuple[str, ...] = ()
    model_dirty: bool = False
    last_question: str = ""

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "request_id": self.request_id,
            "model_id": self.model_id,
            "role_id": self.role_id,
            "busy": self.busy,
            "last_error": self.last_error,
            "last_result": self.last_result,
            "save_enabled": self.save_enabled,
            "debug_enabled": self.debug_enabled,
            "save_dir": self.save_dir,
            "debug_log": list(self.debug_log),
            "selected_cards": list(self.selected_cards),
            "model_dirty": self.model_dirty,
            "last_question": self.last_question,
        }

    @classmethod
    def from_dict(cls, value: dict | None) -> UIState:
        if not isinstance(value, dict):
            return cls()
        state = cls()
        state.session_id = value.get("session_id")
        state.request_id = value.get("request_id")
        state.model_id = value.get("model_id")
        state.role_id = value.get("role_id") or "CS_AGENT"
        state.busy = bool(value.get("busy"))
        state.last_error = value.get("last_error")
        state.last_result = value.get("last_result")
        state.save_enabled = bool(value.get("save_enabled"))
        state.debug_enabled = bool(value.get("debug_enabled"))
        state.save_dir = value.get("save_dir") or ""
        state.debug_log = list(value.get("debug_log") or [])
        state.selected_cards = tuple(value.get("selected_cards") or ())
        state.model_dirty = bool(value.get("model_dirty"))
        state.last_question = value.get("last_question") or ""
        return state

    def note(self, message: str) -> None:
        """Record a debug line only when debug logging was explicitly enabled."""

        if self.debug_enabled:
            self.debug_log.append(message)
            if len(self.debug_log) > 200:
                del self.debug_log[:100]
