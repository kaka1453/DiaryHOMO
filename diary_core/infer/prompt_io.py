from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_prompts(input_file: str) -> list[str]:
    with open(input_file, "r", encoding="utf-8") as fh:
        prompts = [line.strip() for line in fh if line.strip()]
    if not prompts:
        raise ValueError(f"输入文件中未读取到有效 prompt: {input_file}")
    return prompts


def format_markdown_block(
    index: int,
    prompt: str,
    text: str,
    guard: dict | None = None,
    debug_dir: str | None = None,
    audit_config: dict | None = None,
) -> str:
    audit = format_guard_audit(guard, debug_dir=debug_dir, audit_config=audit_config)
    return f"## 第{index}篇\n\n引言: {prompt}\n\n{text}{audit}\n\n---\n"


def format_guard_audit(
    guard: dict | None,
    debug_dir: str | None = None,
    audit_config: dict | None = None,
) -> str:
    audit_config = audit_config or {}
    if not guard or not audit_config.get("include_guard_table", False):
        return ""
    selected = _selected_guard(guard)
    warnings = selected.get("warnings") or selected.get("quality_warnings") or []
    if not audit_config.get("include_warnings", True):
        warnings = []
    retry_count = guard.get("retry_count", 0)
    lines = [
        "",
        "<!-- GUARD_AUDIT",
        "| final | topic | drift | format | language | quality | decision | retry |",
        "|---:|---:|---:|---:|---:|---:|---|---:|",
        (
            f"| {_fmt_score(selected.get('final_score'))} | {_fmt_score(selected.get('topic_score'))} | "
            f"{_fmt_score(selected.get('drift_score'))} | {_fmt_score(selected.get('format_score'))} | "
            f"{_fmt_score(selected.get('language_score'))} | {_fmt_score(selected.get('quality_score'))} | "
            f"{selected.get('decision', guard.get('decision', ''))} | {retry_count} |"
        ),
        f"warnings: {', '.join(warnings) if warnings else 'none'}",
    ]
    if debug_dir:
        lines.append(f"debug_dir: {debug_dir}")
    lines.append("-->")
    return "\n".join(lines)


def _selected_guard(guard: dict) -> dict:
    selected_attempt = guard.get("selected_attempt")
    for attempt in guard.get("attempts") or []:
        if attempt.get("attempt") == selected_attempt:
            return attempt.get("guard") or {}
    if guard.get("attempts"):
        return (guard["attempts"][0] or {}).get("guard") or {}
    return guard


def _fmt_score(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def write_results(results: list[str], output_file: str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.writelines(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt IO smoke test")
    parser.add_argument("--input-file", required=True)
    args = parser.parse_args()
    prompts = load_prompts(args.input_file)
    print(json.dumps({"count": len(prompts), "first": prompts[0]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
