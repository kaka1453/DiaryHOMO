from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


"""
验证集用途说明：

data/val/diary_validation_prompts.jsonl 只用于生成式验证和回归测试，
用于评估 Prompt Contract、DiaryGuard、模型 checkpoint 是否守题。
不要把它混入训练集。

data/dataset/json/boa_*.jsonl 是当前日记 LM 训练数据。
如果后续做指令化训练，请另建 data/dataset/instruction/*.jsonl
和 data/val/instruction_validation.jsonl。
"""


@dataclass
class ValidationSample:
    id: str
    category: str
    prompt: str
    topic_terms: list[str]
    style_hints: list[str]
    must_not_include: list[str]
    max_chars: int | None
    min_guard_score: float
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], line_no: int) -> "ValidationSample":
        prompt = str(data.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"验证集第 {line_no} 行缺少 prompt。")
        return cls(
            id=str(data.get("id") or f"line_{line_no:04d}"),
            category=str(data.get("category") or "uncategorized"),
            prompt=prompt,
            topic_terms=[str(item) for item in data.get("topic_terms") or [] if str(item)],
            style_hints=[str(item) for item in data.get("style_hints") or [] if str(item)],
            must_not_include=[str(item) for item in data.get("must_not_include") or [] if str(item)],
            max_chars=int(data["max_chars"]) if data.get("max_chars") is not None else None,
            min_guard_score=float(data.get("min_guard_score", 75)),
            notes=str(data.get("notes") or ""),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "topic_terms": self.topic_terms,
            "style_hints": self.style_hints,
            "must_not_include": self.must_not_include,
            "max_chars": self.max_chars,
            "min_guard_score": self.min_guard_score,
            "notes": self.notes,
        }


def load_validation_samples(path: str | Path, limit: int | None = None) -> list[ValidationSample]:
    samples = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            samples.append(ValidationSample.from_dict(json.loads(line), line_no))
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"验证集中未读取到样本: {path}")
    return samples


def summarize_samples(samples: list[ValidationSample]) -> dict:
    categories: dict[str, int] = {}
    for sample in samples:
        categories[sample.category] = categories.get(sample.category, 0) + 1
    return {
        "count": len(samples),
        "categories": categories,
        "first": samples[0].to_dict() if samples else None,
    }
