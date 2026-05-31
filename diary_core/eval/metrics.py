from __future__ import annotations

import json
from pathlib import Path
import re

from diary_core.eval.validation import ValidationSample
from diary_core.infer.audit import compact_guard_summary, selected_guard_from_summary, write_jsonl


PASS_DECISIONS = {"pass", "pass_with_warnings"}


def build_eval_result(sample: ValidationSample, output: str, guard: dict) -> dict:
    selected_guard = selected_guard_from_summary(guard)
    scores = {
        "final": selected_guard.get("final_score"),
        "topic": selected_guard.get("topic_score"),
        "drift": selected_guard.get("drift_score"),
        "format": selected_guard.get("format_score"),
        "language": selected_guard.get("language_score"),
        "quality": selected_guard.get("quality_score"),
    }
    must_not_hits = check_must_not_include(output, sample.must_not_include)
    max_chars_ok = True if sample.max_chars is None else len(output) <= sample.max_chars
    min_guard_score_ok = float(scores.get("final") or 0) >= sample.min_guard_score
    topic_hit_rate = topic_terms_hit_rate(output, sample.topic_terms)
    decision = selected_guard.get("decision", guard.get("decision"))
    passed = bool(
        decision in PASS_DECISIONS
        and min_guard_score_ok
        and not must_not_hits
        and max_chars_ok
    )
    return {
        "id": sample.id,
        "category": sample.category,
        "prompt": sample.prompt,
        "output": output,
        "guard": compact_guard_summary(guard),
        "checks": {
            "must_not_hits": must_not_hits,
            "max_chars_ok": max_chars_ok,
            "max_chars": sample.max_chars,
            "output_chars": len(output),
            "min_guard_score_ok": min_guard_score_ok,
            "min_guard_score": sample.min_guard_score,
            "topic_terms_hit_rate": topic_hit_rate,
        },
        "scores": scores,
        "decision": decision,
        "warnings": selected_guard.get("warnings") or selected_guard.get("quality_warnings") or [],
        "pass": passed,
    }


def check_must_not_include(output: str, must_not_include: list[str]) -> list[str]:
    hits = []
    for item in must_not_include:
        term = str(item).strip()
        if not term:
            continue
        if is_special_must_not_hit(output, term) or compact(term) in compact(output):
            hits.append(term)
    return hits


def is_special_must_not_hit(output: str, term: str) -> bool:
    normalized = compact(term).lower()
    if normalized in {"markdown表格", "markdown_table"}:
        return bool(re.search(r"(?m)\|.+\|.*\n\|?\s*:?-{3,}:?\s*\|", output))
    if normalized in {"markdown标题", "markdownheading"}:
        return bool(re.search(r"(?m)^#{1,6}\s+|\n#{1,6}\s+", output))
    if normalized in {"外部链接", "链接", "url"}:
        return bool(re.search(r"https?://|\[[^\]]+\]\([^)]+\)", output))
    if normalized in {"代码块", "codeblock"}:
        return "```" in output
    if normalized in {"多篇分隔符"}:
        return bool(re.search(r"\n---+|\n##\s*第", output))
    return False


def topic_terms_hit_rate(output: str, topic_terms: list[str]) -> float:
    terms = [term for term in topic_terms if str(term).strip()]
    if not terms:
        return 1.0
    hits = sum(1 for term in terms if compact(term) in compact(output))
    return round(hits / len(terms), 4)


def summarize_eval_results(results: list[dict]) -> dict:
    count = len(results)
    categories = sorted({result.get("category") for result in results if result.get("category")})
    summary = {
        "count": count,
        "overall_pass_rate": rate(result.get("pass") for result in results),
        "category_pass_rate": {},
        "avg_final_score": avg_score(results, "final"),
        "avg_topic_score": avg_score(results, "topic"),
        "avg_drift_score": avg_score(results, "drift"),
        "avg_format_score": avg_score(results, "format"),
        "avg_language_score": avg_score(results, "language"),
        "avg_quality_score": avg_score(results, "quality"),
        "warning_rate": rate(bool(result.get("warnings")) for result in results),
        "retry_rate": rate((result.get("guard") or {}).get("retry_count", 0) > 0 for result in results),
        "revise_rate": rate(result.get("decision") == "revise" for result in results),
        "fail_rate": rate(result.get("decision") == "fail" for result in results),
        "must_not_hit_rate": rate(bool((result.get("checks") or {}).get("must_not_hits")) for result in results),
        "max_chars_violation_rate": rate(not (result.get("checks") or {}).get("max_chars_ok", True) for result in results),
        "topic_terms_avg_hit_rate": avg_check(results, "topic_terms_hit_rate"),
        "category": {},
    }
    for category in categories:
        items = [result for result in results if result.get("category") == category]
        category_pass_rate = rate(result.get("pass") for result in items)
        summary["category_pass_rate"][category] = category_pass_rate
        summary["category"][category] = {
            "count": len(items),
            "pass_rate": category_pass_rate,
            "avg_final_score": avg_score(items, "final"),
            "avg_quality_score": avg_score(items, "quality"),
            "warning_rate": rate(bool(result.get("warnings")) for result in items),
            "retry_rate": rate((result.get("guard") or {}).get("retry_count", 0) > 0 for result in items),
            "must_not_hit_rate": rate(bool((result.get("checks") or {}).get("must_not_hits")) for result in items),
            "topic_terms_avg_hit_rate": avg_check(items, "topic_terms_hit_rate"),
        }
    return summary


def write_eval_artifacts(output_run_dir: str | Path, results: list[dict], summary: dict) -> None:
    run_dir = Path(output_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(run_dir / "eval_results.jsonl", results)
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "eval_summary.md").write_text(format_eval_summary_md(summary), encoding="utf-8")


def format_eval_summary_md(summary: dict) -> str:
    lines = [
        "# Eval Summary",
        "",
        f"- count: {summary.get('count', 0)}",
        f"- overall_pass_rate: {summary.get('overall_pass_rate', 0)}",
        f"- avg_final_score: {summary.get('avg_final_score', 0)}",
        f"- avg_quality_score: {summary.get('avg_quality_score', 0)}",
        f"- warning_rate: {summary.get('warning_rate', 0)}",
        f"- retry_rate: {summary.get('retry_rate', 0)}",
        f"- must_not_hit_rate: {summary.get('must_not_hit_rate', 0)}",
        f"- topic_terms_avg_hit_rate: {summary.get('topic_terms_avg_hit_rate', 0)}",
        "",
        "## Category",
        "",
        "| category | count | pass_rate | avg_quality | warning_rate | retry_rate | must_not_hit_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, data in (summary.get("category") or {}).items():
        lines.append(
            f"| {category} | {data.get('count', 0)} | {data.get('pass_rate', 0)} | "
            f"{data.get('avg_quality_score', 0)} | {data.get('warning_rate', 0)} | "
            f"{data.get('retry_rate', 0)} | {data.get('must_not_hit_rate', 0)} |"
        )
    return "\n".join(lines) + "\n"


def avg_score(results: list[dict], key: str) -> float:
    values = []
    for result in results:
        value = (result.get("scores") or {}).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(sum(values) / len(values), 4) if values else 0.0


def avg_check(results: list[dict], key: str) -> float:
    values = []
    for result in results:
        value = (result.get("checks") or {}).get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(sum(values) / len(values), 4) if values else 0.0


def rate(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(1 for item in items if item) / len(items), 4)


def compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()
