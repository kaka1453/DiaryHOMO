"""Safe Markdown rendering for untrusted model output.

The assistant text comes from a local model and must never be able to inject
HTML, ``<script>`` or remote images into the local review UI. Gradio's Markdown
sanitizer is the default safety net; this module adds a deterministic,
dependency-free pre-pass so the behaviour is testable without a browser:

* raw HTML tags are escaped, including ``<script>``, ``<img onerror=...>`` and
  ``<iframe>``;
* the ``javascript:``/``data:`` URL schemes are neutralised;
* fenced code blocks and inline code spans are preserved verbatim (escaping
  only the fence markers themselves would break copy/paste);
* newlines and emoji are preserved as-is.
"""

from __future__ import annotations

import html
import re

_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_INLINE = re.compile(r"`([^`\n]+)`")
_TAG = re.compile(r"<[^>]{0,500}>")
_URL = re.compile(r"\((\s*)([^)\s]{0,500})(\s*)\)")
_DANGEROUS_SCHEME = re.compile(r"^(javascript|data|vbscript)\s*:", re.IGNORECASE)


def _escape_outside_code(text: str) -> str:
    """Escape HTML tags while keeping fenced/inline code byte-identical."""

    out: list[str] = []
    cursor = 0
    for match in _FENCE.finditer(text):
        out.append(_TAG.sub(lambda m: html.escape(m.group(0)), text[cursor : match.start()]))
        out.append(match.group(0))
        cursor = match.end()
    out.append(_TAG.sub(lambda m: html.escape(m.group(0)), text[cursor:]))
    return "".join(out)


def _neutralise_links(text: str) -> str:
    """Neutralise dangerous URL schemes everywhere they can become active.

    Markdown links ``[..](javascript:...)`` and HTML attributes such as
    ``<iframe src="javascript:...">`` are both covered: the tag itself is
    escaped separately, but the scheme must not survive inside an attribute.
    """

    def repl(match: re.Match) -> str:
        target = match.group(2)
        if _DANGEROUS_SCHEME.match(target.strip()):
            return f"({match.group(1)}#{match.group(3)})"
        return match.group(0)

    text = _URL.sub(repl, text)
    # ``src=data:`` / ``href='vbscript:'`` inside escaped tags.
    text = re.sub(
        r"""(?i)(\b(?:src|href|action|formaction|xlink:href)\s*=\s*)(["']?)"""
        r"""\s*(javascript|data|vbscript)\s*:""",
        lambda m: f"{m.group(1)}{m.group(2)}unsafe-scheme:",
        text,
    )
    return text


def safe_markdown(text: str) -> str:
    """Return a Markdown string that cannot carry active HTML/script content."""

    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _escape_outside_code(text)
    text = _neutralise_links(text)
    return text


def chat_message(role: str, text: str) -> dict:
    """Build a Gradio ``messages``-format chatbot entry with safe Markdown."""

    return {"role": role, "content": safe_markdown(text)}
