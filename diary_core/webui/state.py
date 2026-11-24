from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

import torch


@dataclass
class Conversation:
    history: list[str] = field(default_factory=list)
    revision: int = 0
    output_run_dir: str | None = None

    def clear(self):
        self.revision += 1
        self.history.clear()
        self.output_run_dir = None


class WebUIState:
    def __init__(self, app_config: dict):
        self.app_config = app_config
        self.generation_params = {
            "max_new_tokens": app_config["max_new_tokens"],
            "temperature": app_config["temperature"],
            "top_p": app_config["top_p"],
            "top_k": app_config["top_k"],
            "repetition_penalty": app_config["repetition_penalty"],
            "num_beams": app_config["num_beams"],
        }
        self.tokenizer = None
        self.model = None
        self.lock = RLock()
        self.model_revision = 0
        self.default_session = Conversation()

    @property
    def conversation_history(self) -> list[str]:
        return self.default_session.history

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def clear_model(self) -> None:
        with self.lock:
            self.model_revision += 1
            self.model = None
            self.tokenizer = None
            self.default_session.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
