from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import uuid

from diary_core.config.common import resolve_path, str2bool


DEFAULT_PROMPT_DEBUG_CONFIG = {
    "enabled": False,
    "output_dir": "debug/prompt",
    "include_rendered_prompt": True,
    "include_messages": True,
    "include_model_output": True,
}


def normalize_prompt_debug_config(config: dict | None) -> dict:
    normalized = dict(DEFAULT_PROMPT_DEBUG_CONFIG)
    if isinstance(config, dict):
        normalized.update(config)
    normalized["enabled"] = str2bool(normalized.get("enabled", False))
    normalized["include_rendered_prompt"] = str2bool(normalized.get("include_rendered_prompt", True))
    normalized["include_messages"] = str2bool(normalized.get("include_messages", True))
    normalized["include_model_output"] = str2bool(normalized.get("include_model_output", True))
    normalized["output_dir"] = str(resolve_path(normalized.get("output_dir") or "debug/prompt"))
    return normalized


class PromptDebugRun:
    def __init__(self, config: dict | None):
        self.config = normalize_prompt_debug_config(config)
        self.enabled = self.config["enabled"]
        self.run_dir: Path | None = None
        if self.enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = uuid.uuid4().hex[:6]
            self.run_dir = Path(self.config["output_dir"]) / f"{timestamp}_{suffix}"
            self.run_dir.mkdir(parents=True, exist_ok=False)
            self.log("prompt_debug run created")

    @property
    def run_dir_text(self) -> str | None:
        return str(self.run_dir) if self.run_dir else None

    def write_text(self, filename: str, text: str, enabled: bool = True) -> None:
        if not self.enabled or self.run_dir is None:
            return
        path = self.run_dir / filename
        path.write_text(text if enabled else "(disabled by prompt_debug config)\n", encoding="utf-8")
        self.log(f"write {filename}")

    def write_json(self, filename: str, data, enabled: bool = True) -> None:
        if not self.enabled or self.run_dir is None:
            return
        payload = data if enabled else {"disabled": True}
        path = self.run_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.log(f"write {filename}")

    def log(self, message: str) -> None:
        if not self.enabled or self.run_dir is None:
            return
        timestamp = datetime.now().isoformat(timespec="seconds")
        with (self.run_dir / "flow.log").open("a", encoding="utf-8") as fh:
            fh.write(f"[{timestamp}] {message}\n")
