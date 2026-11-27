"""Report rendering and Task B aggregate reading (never invents numbers)."""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lm_trainer.eval.schema import EvalError, canonical_json, sha256_text

REPORT_SCHEMA = "lm_trainer.eval-report.v1"


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarize(result_dict: dict) -> dict:
    """Aggregate a runner result into per-model, per-kind, per-condition stats."""

    records = result_dict.get("records", [])
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (
            record["model_id"],
            record["kind"],
            record.get("memory_condition", "M0"),
        )
        groups[key].append(record)
    per_model = []
    for key in sorted(groups):
        rows = groups[key]
        metrics = {
            "units": len(rows),
            "committed": sum(1 for r in rows if r.get("committed")),
            "errors": sum(1 for r in rows if r.get("error_code")),
            "finish_reasons": _count(
                r.get("finish_reason") for r in rows
            ),
            "input_tokens": _sum(r.get("input_tokens") for r in rows),
            "output_tokens": _sum(r.get("output_tokens") for r in rows),
            "ttft_ms": _mean([r.get("ttft_ms") for r in rows]),
            "tokens_per_second": _mean([r.get("tokens_per_second") for r in rows]),
            "latency_ms": _mean([r.get("latency_ms") for r in rows]),
            "peak_vram_mb": _max([r.get("peak_vram_mb") for r in rows]),
        }
        per_model.append(
            {
                "model_id": key[0],
                "kind": key[1],
                "memory_condition": key[2],
                "metrics_source": rows[0].get("metrics_source", "mock"),
                **metrics,
            }
        )
    total_cases = len({r["case_id"] for r in records})
    total_events = len({r.get("event_group", "") for r in records})
    return {
        "schema": REPORT_SCHEMA,
        "fixture_only": True,
        "run_id": result_dict.get("run_id"),
        "config_identity": result_dict.get("config_identity"),
        "mode": result_dict.get("mode", "mock"),
        "total_records": len(records),
        "distinct_cases": total_cases,
        "distinct_events": total_events,
        "skipped": result_dict.get("skipped", []),
        "per_model": per_model,
        "metrics_source": result_dict.get("mode", "mock"),
    }


def _count(values) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for value in values:
        out[str(value)] += 1
    return dict(sorted(out.items()))


def _sum(values) -> int | None:
    values = [v for v in values if isinstance(v, int)]
    return sum(values) if values else None


def _max(values) -> float | None:
    values = [v for v in values if isinstance(v, (int, float))]
    return max(values) if values else None


def to_csv(result_dict: dict) -> str:
    """Export the flat per-unit table (no rubric, no reference)."""

    columns = [
        "case_id", "model_id", "kind", "memory_condition", "turn_index",
        "finish_reason", "committed", "error_code", "input_tokens",
        "output_tokens", "ttft_ms", "tokens_per_second", "latency_ms",
        "peak_vram_mb", "prompt_sha256", "card_audit", "output",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for record in result_dict.get("records", []):
        row = dict(record)
        row["card_audit"] = canonical_json(record.get("card_audit") or [])
        writer.writerow({k: row.get(k, "") for k in columns})
    return buffer.getvalue()


def to_jsonl(result_dict: dict) -> str:
    lines = []
    for record in result_dict.get("records", []):
        lines.append(canonical_json(record))
    return "\n".join(lines) + ("\n" if lines else "")


def render_markdown(
    summary: dict,
    *,
    reveal: dict | None = None,
    title: str = "工业客服评测报告",
) -> str:
    """Render a shareable report. No card plaintext, no rubric, no reference."""

    mode = summary.get("mode", "mock")
    label = "Mock（无真实推理）" if mode == "mock" else "真实本地推理"
    lines = [
        f"# {title}",
        "",
        f"- 模式：`{mode}`（{label}）",
        "- fixture_only：`true`（全部为虚构机器视觉客服数据）",
        f"- 运行 ID：`{summary.get('run_id')}`",
        f"- 配置指纹：`{summary.get('config_identity')}`",
        f"- 独立 case 数：{summary.get('distinct_cases', 0)}；独立 event 组数：{summary.get('distinct_events', 0)}",
        f"- 记录条数：{summary.get('total_records', 0)}",
        "",
    ]
    if summary.get("skipped"):
        lines += ["## 缺失模型（SKIP）", ""]
        for item in summary["skipped"]:
            lines.append(
                f"- `{item['model_id']}`：`{item.get('status', 'SKIP')}`，原因 `{item.get('reason', 'MODEL_UNAVAILABLE')}`"
            )
        lines.append("")
    lines += [
        "## 分组实测指标",
        "",
        "| 模型 | 类型 | 条件 | 单元 | 提交 | 错误 | 输入 tokens | 输出 tokens | TTFT ms | tok/s | 峰值显存 MiB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("per_model", []):
        lines.append(
            "| `{model_id}` | {kind} | {condition} | {units} | {committed} | {errors} | {input_tokens} | {output_tokens} | {ttft} | {tps} | {vram} |".format(
                model_id=row["model_id"],
                kind=row["kind"],
                condition=row["memory_condition"],
                units=row["units"],
                committed=row["committed"],
                errors=row["errors"],
                input_tokens=row["input_tokens"] if row["input_tokens"] is not None else "N/A",
                output_tokens=row["output_tokens"] if row["output_tokens"] is not None else "N/A",
                ttft=row["ttft_ms"] if row["ttft_ms"] is not None else "N/A",
                tps=row["tokens_per_second"] if row["tokens_per_second"] is not None else "N/A",
                vram=row["peak_vram_mb"] if row["peak_vram_mb"] is not None else "N/A",
            )
        )
    lines.append("")
    if reveal:
        lines += [
            "## 匿名盲评揭盲汇总",
            "",
            f"- 独立 case：{reveal.get('independent_cases', 0)}；独立 event：{reveal.get('independent_events', 0)}",
            f"- 已评审：{reveal.get('reviewed_count', 0)} / {reveal.get('item_count', 0)}；未评审：{reveal.get('unreviewed_count', 0)}",
            f"- 胜场：{reveal.get('wins', {})}",
            f"- 选择分布：{reveal.get('counts', {})}",
            "",
            "| 维度 | 样本数 | 均值 | 缺失 | 分布 1–5 |",
            "|---|---:|---:|---:|---|",
        ]
        for name, stats in (reveal.get("dimension_stats") or {}).items():
            lines.append(
                f"| `{name}` | {stats['count']} | {stats['mean'] if stats['mean'] is not None else 'N/A'} | {stats['missing']} | {stats['distribution']} |"
            )
        lines.append("")
        if reveal.get("duplicate_seed_note"):
            lines += [f"> {reveal['duplicate_seed_note']}", ""]
    lines += [
        "## 卡片审计（仅 ID/版本/原因）",
        "",
        "共享报告不输出卡片明文。注入与拒载明细见每单元记录的 `card_audit`。",
        "",
        "> 本报告仅用于虚构 fixture 的工程验证，不代表真实客户场景效果。",
    ]
    return "\n".join(lines) + "\n"


def _escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_html(summary: dict, *, reveal: dict | None = None) -> str:
    """Self-contained HTML report built from escaped text only."""

    md = render_markdown(summary, reveal=reveal)
    title = _escape_html("工业客服评测报告")
    body = _escape_html(md).replace("\n", "<br>\n")
    return (
        "<!doctype html><html lang=\"zh\"><head><meta charset=\"utf-8\">"
        f"<title>{title}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;"
        "padding:0 1rem;line-height:1.6}code{background:#f4f4f4;padding:2px 4px;}"
        "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:4px 8px;}"
        "</style></head><body><pre style=\"white-space:pre-wrap;font-family:inherit\">"
        f"{body}</pre></body></html>"
    )


def write_reports(
    result_dict: dict,
    output_dir: Path | str,
    *,
    reveal: dict | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(result_dict)
    paths = {
        "json": output_dir / "report.json",
        "jsonl": output_dir / "records.jsonl",
        "csv": output_dir / "records.csv",
        "markdown": output_dir / "report.md",
        "html": output_dir / "report.html",
    }
    paths["json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["jsonl"].write_text(to_jsonl(result_dict), encoding="utf-8")
    paths["csv"].write_text(to_csv(result_dict), encoding="utf-8")
    paths["markdown"].write_text(
        render_markdown(summary, reveal=reveal), encoding="utf-8"
    )
    paths["html"].write_text(
        render_html(summary, reveal=reveal), encoding="utf-8"
    )
    return paths


@dataclass(frozen=True, slots=True)
class TaskBAggregate:
    schema: str
    status: str
    levels: tuple[dict, ...]
    acceptance: tuple[dict, ...]
    aggregate: tuple[dict, ...]
    fixture_only: bool
    notes: tuple[str, ...]
    source: str

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "status": self.status,
            "levels": list(self.levels),
            "acceptance": list(self.acceptance),
            "aggregate": list(self.aggregate),
            "fixture_only": self.fixture_only,
            "notes": list(self.notes),
            "source": self.source,
        }


def load_benchmark_aggregate(path: Path | str) -> TaskBAggregate:
    """Read a Task B ``lm_trainer.benchmark.v1`` aggregate; never fabricate values."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvalError("BENCHMARK_INVALID", f"cannot read {path.name}") from exc
    if payload.get("schema") != "lm_trainer.benchmark.v1":
        raise EvalError("BENCHMARK_INVALID", "schema")
    if payload.get("fixture_only") is not True:
        raise EvalError("FIXTURE_ONLY_REQUIRED", "benchmark")
    for key in ("levels", "acceptance", "aggregate"):
        if not isinstance(payload.get(key), list):
            raise EvalError("BENCHMARK_INVALID", key)
    return TaskBAggregate(
        payload["schema"],
        payload.get("status", "NOT RUN"),
        tuple(payload["levels"]),
        tuple(payload["acceptance"]),
        tuple(payload["aggregate"]),
        True,
        tuple(payload.get("notes", [])),
        path.name,
    )


def benchmark_rows(aggregate: TaskBAggregate) -> list[dict]:
    """Flatten the Task B aggregate for display; missing values become NOT RUN."""

    rows = []
    for level in aggregate.levels:
        rows.append(
            {
                "scope": f"level:{level.get('level', '?')}",
                "status": level.get("status", "NOT RUN"),
                "metric": level.get("scope", ""),
                "value": level.get("tests_passed")
                if level.get("tests_passed") is not None
                else "NOT RUN",
            }
        )
    for acceptance in aggregate.acceptance:
        rows.append(
            {
                "scope": f"acceptance:{acceptance.get('id', '?')}",
                "status": acceptance.get("status", "NOT RUN"),
                "metric": acceptance.get("evidence", ""),
                "value": acceptance.get("status", "NOT RUN"),
            }
        )
    for entry in aggregate.aggregate:
        rows.append(
            {
                "scope": f"aggregate:{entry.get('case', '?')}/{entry.get('stage_name', '?')}",
                "status": entry.get("status", "NOT RUN"),
                "metric": "input_tokens_per_second",
                "value": entry.get("input_tokens_per_second", "NOT RUN"),
            }
        )
    return rows


def write_summary_json(summary: dict, path: Path | str) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def sha256_of_payload(payload: Any) -> str:
    return sha256_text(canonical_json(payload))
