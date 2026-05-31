from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from diary_core.infer.diary_runtime import DiaryResult


AUDIT_JSONL_FILENAME = "audit.jsonl"
AUDIT_SUMMARY_FILENAME = "audit_summary.json"
AUDIT_TABLE_FILENAME = "audit_table.md"


DEFAULT_AUDIT_CONFIG = {
    "enabled": True,
    "write_jsonl": True,
    "write_summary": True,
    "write_table": True,
    "include_contract": True,
    "include_guard": True,
    "include_debug_dir": True,
    "include_generation_request": True,
    "include_postprocess": True,
    "include_final_text": True,
    "include_guard_table": True,
    "include_warnings": True,
}


def normalize_audit_config(config: dict | None) -> dict:
    normalized = dict(DEFAULT_AUDIT_CONFIG)
    if isinstance(config, dict):
        normalized.update(config)
    for key, default_value in DEFAULT_AUDIT_CONFIG.items():
        if isinstance(default_value, bool):
            normalized[key] = _as_bool(normalized.get(key, default_value))
    return normalized


def build_audit_record(
    *,
    index: int,
    result: DiaryResult,
    sample: dict | None = None,
    audit_config: dict | None = None,
) -> dict:
    audit_config = normalize_audit_config(audit_config)
    sample = sample or {}
    selected_guard = selected_guard_from_summary(result.guard)
    record = {
        "index": index,
        "id": sample.get("id"),
        "category": sample.get("category"),
        "raw_prompt": result.raw_prompt,
        "retry_count": result.guard.get("retry_count", 0) if isinstance(result.guard, dict) else 0,
        "selected_attempt": result.guard.get("selected_attempt", 1) if isinstance(result.guard, dict) else 1,
        "scores": extract_scores(selected_guard),
        "decision": selected_guard.get("decision", result.guard.get("decision") if isinstance(result.guard, dict) else None),
        "warnings": selected_guard.get("warnings") or selected_guard.get("quality_warnings") or [],
        "forbidden_hits": selected_guard.get("forbidden_hits") or [],
        "format_hits": selected_guard.get("format_hits") or [],
        "language_noise_hits": selected_guard.get("language_noise_hits") or [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if audit_config["include_final_text"]:
        record["final_text"] = result.final_text
    if audit_config["include_contract"]:
        record["contract"] = result.contract.to_dict()
    if audit_config["include_generation_request"]:
        record["generation_request"] = result.generation_request
    if audit_config["include_postprocess"]:
        record["postprocess"] = result.postprocess
    if audit_config["include_guard"]:
        record["guard"] = compact_guard_summary(result.guard)
    if audit_config["include_debug_dir"]:
        record["debug_dir"] = result.debug_dir
    return record


def append_audit_record(output_run_dir: str | Path, record: dict, audit_config: dict | None = None) -> Path | None:
    audit_config = normalize_audit_config(audit_config)
    if not audit_config["enabled"] or not audit_config["write_jsonl"]:
        return None
    path = Path(output_run_dir) / AUDIT_JSONL_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_audit_artifacts(
    output_run_dir: str | Path,
    records: list[dict],
    audit_config: dict | None = None,
) -> dict:
    audit_config = normalize_audit_config(audit_config)
    if not audit_config["enabled"]:
        return {}
    run_dir = Path(output_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_audit_records(records)
    if audit_config["write_jsonl"]:
        write_jsonl(run_dir / AUDIT_JSONL_FILENAME, records)
    if audit_config["write_summary"]:
        (run_dir / AUDIT_SUMMARY_FILENAME).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if audit_config["write_table"]:
        (run_dir / AUDIT_TABLE_FILENAME).write_text(format_audit_table(records, summary), encoding="utf-8")
    return summary


def refresh_audit_artifacts_from_jsonl(output_run_dir: str | Path, audit_config: dict | None = None) -> dict:
    path = Path(output_run_dir) / AUDIT_JSONL_FILENAME
    if not path.exists():
        return {}
    records = read_jsonl(path)
    return write_audit_artifacts(output_run_dir, records, audit_config)


def summarize_audit_records(records: list[dict]) -> dict:
    count = len(records)
    decisions = [str(record.get("decision") or "") for record in records]
    warnings = [record.get("warnings") or [] for record in records]
    categories = sorted({record.get("category") for record in records if record.get("category")})
    summary = {
        "count": count,
        "overall_pass_rate": _rate(decision in {"pass", "pass_with_warnings"} for decision in decisions),
        "warning_rate": _rate(bool(item) for item in warnings),
        "retry_rate": _rate(int(record.get("retry_count") or 0) > 0 for record in records),
        "revise_rate": _rate(decision == "revise" for decision in decisions),
        "fail_rate": _rate(decision == "fail" for decision in decisions),
        "avg_final_score": _avg_score(records, "final"),
        "avg_topic_score": _avg_score(records, "topic"),
        "avg_drift_score": _avg_score(records, "drift"),
        "avg_format_score": _avg_score(records, "format"),
        "avg_language_score": _avg_score(records, "language"),
        "avg_quality_score": _avg_score(records, "quality"),
        "category": {},
    }
    for category in categories:
        items = [record for record in records if record.get("category") == category]
        item_decisions = [str(record.get("decision") or "") for record in items]
        summary["category"][category] = {
            "count": len(items),
            "pass_rate": _rate(decision in {"pass", "pass_with_warnings"} for decision in item_decisions),
            "warning_rate": _rate(bool(record.get("warnings") or []) for record in items),
            "retry_rate": _rate(int(record.get("retry_count") or 0) > 0 for record in items),
            "avg_final_score": _avg_score(items, "final"),
            "avg_quality_score": _avg_score(items, "quality"),
        }
    return summary


def format_audit_table(records: list[dict], summary: dict | None = None) -> str:
    lines = [
        "# Audit Table",
        "",
        "| # | id | category | decision | final | topic | drift | quality | retry | warnings | prompt |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for record in records:
        scores = record.get("scores") or {}
        warnings = ", ".join(record.get("warnings") or [])
        lines.append(
            "| {index} | {id} | {category} | {decision} | {final} | {topic} | {drift} | {quality} | {retry} | {warnings} | {prompt} |".format(
                index=record.get("index", ""),
                id=_cell(record.get("id")),
                category=_cell(record.get("category")),
                decision=_cell(record.get("decision")),
                final=_score(scores.get("final")),
                topic=_score(scores.get("topic")),
                drift=_score(scores.get("drift")),
                quality=_score(scores.get("quality")),
                retry=record.get("retry_count", 0),
                warnings=_cell(warnings or "none"),
                prompt=_cell(_truncate(record.get("raw_prompt") or "", 48)),
            )
        )
    if summary:
        lines.extend(
            [
                "",
                "## Summary",
                "",
                f"- count: {summary.get('count', 0)}",
                f"- overall_pass_rate: {_score(summary.get('overall_pass_rate'))}",
                f"- avg_final_score: {_score(summary.get('avg_final_score'))}",
                f"- avg_quality_score: {_score(summary.get('avg_quality_score'))}",
                f"- warning_rate: {_score(summary.get('warning_rate'))}",
                f"- retry_rate: {_score(summary.get('retry_rate'))}",
            ]
        )
    return "\n".join(lines) + "\n"


def selected_guard_from_summary(guard: dict | None) -> dict:
    if not isinstance(guard, dict):
        return {}
    selected_attempt = guard.get("selected_attempt")
    for attempt in guard.get("attempts") or []:
        if attempt.get("attempt") == selected_attempt:
            return attempt.get("guard") or {}
    if guard.get("attempts"):
        return (guard["attempts"][0] or {}).get("guard") or {}
    return guard


def compact_guard_summary(guard: dict | None) -> dict:
    if not isinstance(guard, dict):
        return {}
    compact = {key: value for key, value in guard.items() if key != "config"}
    return compact


def extract_scores(selected_guard: dict) -> dict:
    return {
        "final": selected_guard.get("final_score"),
        "topic": selected_guard.get("topic_score"),
        "drift": selected_guard.get("drift_score"),
        "format": selected_guard.get("format_score"),
        "language": selected_guard.get("language_score"),
        "quality": selected_guard.get("quality_score"),
    }


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def _avg_score(records: list[dict], key: str) -> float:
    values = []
    for record in records:
        value = (record.get("scores") or {}).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(sum(values) / len(values), 4) if values else 0.0


def _rate(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(1 for item in items if item) / len(items), 4)


def _score(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.4g}"
    return str(value)


def _truncate(text: str, max_len: int) -> str:
    text = str(text).replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _cell(value) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
