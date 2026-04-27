from __future__ import annotations

import torch


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
        self.conversation_history: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def clear_model(self) -> None:
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

