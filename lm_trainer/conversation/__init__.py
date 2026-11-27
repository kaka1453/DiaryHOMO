"""Public tokenizer-independent contract for multi-participant conversations."""

from .contract import (
    RENDERER_VERSION,
    SOURCE_SCHEMA,
    WINDOW_SCHEMA,
    ConversationError,
    adapt_window,
    author_blocks,
    inspect_candidate,
    observe,
    parse_catalog,
    validate_route,
)
from .renderer import render_observation

__all__ = [
    "RENDERER_VERSION",
    "SOURCE_SCHEMA",
    "WINDOW_SCHEMA",
    "ConversationError",
    "adapt_window",
    "author_blocks",
    "inspect_candidate",
    "observe",
    "parse_catalog",
    "render_observation",
    "validate_route",
]
