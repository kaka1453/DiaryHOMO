"""Fictional context-card conditions M0–M4 for the Task D card experiment.

Only the card snapshot handed to Task C changes between conditions. Model,
role, template and generation settings stay fixed. The runtime decides what is
actually injected; ``card_condition`` only prepares the candidate snapshot and
never fabricates an ``injected`` audit result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lm_trainer.chat.memory import CardPermissions, ContextCard
from lm_trainer.eval.schema import MEMORY_CONDITIONS, EvalError

M0_NO_CARD = "M0"
M1_RELEVANT = "M1"
M2_IRRELEVANT = "M2"
M3_CONFLICT = "M3"
M4_AMBIGUOUS = "M4"

CONDITIONS = MEMORY_CONDITIONS

FAKE_NAMESPACE = "fixture-line-iris"


@dataclass(frozen=True, slots=True)
class CardCondition:
    condition: str
    label: str
    candidates: tuple[ContextCard, ...]


CARD_FIELDS = (
    "card_id",
    "namespace",
    "kind",
    "facts",
    "event_time",
    "recorded_at",
    "version",
    "status",
    "supersedes",
    "source_ref",
    "uncertainty",
    "permissions",
    "scope",
    "session_id",
)


def _card(payload: dict, *, namespace: str | None = None) -> ContextCard:
    # ``condition`` is a Task D grouping label, not a Task C card field.
    body = {key: payload[key] for key in CARD_FIELDS if key in payload}
    unknown = set(payload) - set(CARD_FIELDS) - {"condition"}
    if unknown:
        raise EvalError("SCHEMA_INVALID", f"unknown card fields {sorted(unknown)}")
    if namespace is not None:
        body["namespace"] = namespace
    try:
        body["permissions"] = CardPermissions(**body["permissions"])
        return ContextCard(**body)
    except (KeyError, TypeError, ValueError) as exc:
        raise EvalError("SCHEMA_INVALID", "context card") from exc


def load_card_fixture(path: Path | str) -> tuple[dict[str, Any], list[ContextCard]]:
    """Load a fictional card fixture file and return (raw, cards)."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvalError("SCHEMA_INVALID", f"cannot read {path.name}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "lm_trainer.eval-cards.v1":
        raise EvalError("SCHEMA_INVALID", "card fixture schema")
    if payload.get("fixture_only") is not True:
        raise EvalError("FIXTURE_ONLY_REQUIRED", "card fixture")
    cards = payload.get("cards")
    if not isinstance(cards, list) or not cards:
        raise EvalError("SCHEMA_INVALID", "card fixture cards")
    parsed = [_card(entry) for entry in cards]
    return payload, parsed


def build_conditions(fixture: dict[str, Any]) -> dict[str, CardCondition]:
    """Build the five memory conditions from a loaded card fixture.

    The fixture groups cards by ``condition``. ``M0`` is always empty. Every
    other condition must be present exactly once, which keeps the experiment
    honest: no silent fallback to a different card set.
    """

    groups: dict[str, list[ContextCard]] = {M0_NO_CARD: []}
    for raw in fixture.get("cards", []):
        condition = raw.get("condition")
        if condition not in CONDITIONS:
            raise EvalError("SCHEMA_INVALID", f"card condition {condition!r}")
        groups.setdefault(condition, []).append(_card(raw))
    if set(groups) != set(CONDITIONS):
        raise EvalError(
            "SCHEMA_INVALID",
            f"card fixture must define all of {list(CONDITIONS)}",
        )
    labels = {
        M0_NO_CARD: "无卡对照",
        M1_RELEVANT: "正确相关卡",
        M2_IRRELEVANT: "无关产线卡",
        M3_CONFLICT: "更新/冲突卡",
        M4_AMBIGUOUS: "指代不清卡",
    }
    return {
        condition: CardCondition(condition, labels[condition], tuple(cards))
        for condition, cards in groups.items()
    }


def card_condition(condition: str, fixture: dict[str, Any]) -> CardCondition:
    """Return one memory condition, validating its name first."""

    if condition not in CONDITIONS:
        raise EvalError("SCHEMA_INVALID", f"unknown memory condition {condition!r}")
    return build_conditions(fixture)[condition]
