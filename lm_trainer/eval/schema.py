"""Versioned, fictional-only evaluation data contract.

The contract is deliberately separate from the training release schema:

* evaluation cases live under ``lm_trainer.eval.v1``;
* a manifest pins the case file SHA-256 digests and the allowed ``split``;
* ``rubric`` and ``reference_only`` are review-only fields. They are validated
  and carried in the artifact, but never reach the runtime prompt. The
  isolation is structural: :mod:`lm_trainer.eval.runner` builds prompts from
  ``messages`` only and asserts the review-only fields are absent from what is
  handed to Task C.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "lm_trainer.eval.v1"
MANIFEST_SCHEMA = "lm_trainer.eval-manifest.v1"
ALLOWED_SPLITS = ("dev", "test")
REVIEW_ONLY_FIELDS = ("rubric", "reference_only")
MEMORY_CONDITIONS = ("M0", "M1", "M2", "M3", "M4")
KINDS = ("fixed_history", "multi_turn_rollout")
MAX_MESSAGES = 32
MAX_CONTENT_CHARS = 20_000


class EvalError(Exception):
    """Fixed, content-free error codes for evaluation gating."""

    def __init__(self, code: str, detail: str | None = None):
        self.code = code
        self.detail = detail
        super().__init__(detail or code)


def canonical_json(value: Any) -> str:
    """Stable JSON encoding used for hashing and artifact identity."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require(cond: bool, code: str, detail: str | None = None):
    if not cond:
        raise EvalError(code, detail)


def _string(value: Any, code: str) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_case(
    value: Any,
    *,
    source: str = "inline",
    allowed_split: str | None = None,
) -> dict:
    """Validate one evaluation case and return a normalized shallow copy.

    ``allowed_split`` enforces the manifest gate: the runner passes the locked
    split so ``test`` is rejected without the explicit lock.
    """

    _require(isinstance(value, dict), "SCHEMA_INVALID", f"{source}: not an object")
    _require(
        value.get("schema") == CONTRACT_SCHEMA,
        "SCHEMA_INVALID",
        f"{source}: wrong schema",
    )
    allowed = {k for k in value}
    expected = {
        "schema",
        "case_id",
        "target_person",
        "event_group",
        "split",
        "fixture_only",
        "kind",
        "messages",
        "rubric",
        "reference_only",
        "memory_condition",
    }
    _require(
        allowed == expected,
        "SCHEMA_INVALID",
        f"{source}: fields {sorted(allowed ^ expected)}",
    )
    case_id = value["case_id"]
    _require(
        isinstance(case_id, str)
        and bool(case_id)
        and all(c.isalnum() or c in "-_." for c in case_id),
        "SCHEMA_INVALID",
        f"{source}: case_id",
    )
    _require(
        isinstance(value["kind"], str) and value["kind"] in KINDS,
        "SCHEMA_INVALID",
        f"{source}: kind",
    )
    _require(
        isinstance(value["split"], str) and value["split"] in ALLOWED_SPLITS,
        "SCHEMA_INVALID",
        f"{source}: split",
    )
    if allowed_split is not None:
        _require(
            value["split"] == allowed_split,
            "SPLIT_NOT_AUTHORIZED",
            f"{source}: split {value['split']} requires --allow-test",
        )
    _require(value["fixture_only"] is True, "FIXTURE_ONLY_REQUIRED", source)
    _require(
        isinstance(value["target_person"], str) and bool(value["target_person"]),
        "SCHEMA_INVALID",
        f"{source}: target_person",
    )
    _require(
        isinstance(value["event_group"], str) and bool(value["event_group"]),
        "SCHEMA_INVALID",
        f"{source}: event_group",
    )
    messages = value["messages"]
    _require(
        isinstance(messages, list) and len(messages) >= 1,
        "SCHEMA_INVALID",
        f"{source}: messages",
    )
    _require(len(messages) <= MAX_MESSAGES, "SCHEMA_INVALID", f"{source}: too long")
    fixed = value["kind"] == "fixed_history"
    if fixed:
        _require(len(messages) % 2 == 1, "SCHEMA_INVALID", f"{source}: odd length")
    else:
        _require(len(messages) >= 2, "SCHEMA_INVALID", f"{source}: rollout needs turns")
    for index, message in enumerate(messages):
        _require(
            isinstance(message, dict) and set(message) == {"role", "content"},
            "SCHEMA_INVALID",
            f"{source}: message {index}",
        )
        role, content = message["role"], message["content"]
        if fixed:
            _require(
                role == ("assistant" if index % 2 else "user"),
                "SCHEMA_INVALID",
                f"{source}: message {index} role",
            )
        else:
            _require(
                index % 2 == 0 or role == "assistant",
                "SCHEMA_INVALID",
                f"{source}: message {index} role",
            )
            _require(
                index % 2 == 1 or role == "user",
                "SCHEMA_INVALID",
                f"{source}: message {index} role",
            )
        _require(
            isinstance(content, str)
            and bool(content.strip())
            and len(content) <= MAX_CONTENT_CHARS,
            "SCHEMA_INVALID",
            f"{source}: message {index} content",
        )
        _require(
            "<reference_data>" not in content and "</reference_data>" not in content,
            "RESERVED_MARKER",
            f"{source}: message {index}",
        )
    rubric = value["rubric"]
    _require(isinstance(rubric, list), "SCHEMA_INVALID", f"{source}: rubric")
    for index, item in enumerate(rubric):
        _require(
            isinstance(item, str) and bool(item.strip()),
            "SCHEMA_INVALID",
            f"{source}: rubric {index}",
        )
    reference = value["reference_only"]
    _require(isinstance(reference, list), "SCHEMA_INVALID", f"{source}: reference")
    for index, item in enumerate(reference):
        _require(
            isinstance(item, str) and bool(item.strip()),
            "SCHEMA_INVALID",
            f"{source}: reference {index}",
        )
    _require(
        value["memory_condition"] in MEMORY_CONDITIONS,
        "SCHEMA_INVALID",
        f"{source}: memory_condition",
    )
    return dict(value)


def build_manifest(cases: list[dict], *, split: str = "dev") -> dict:
    """Build a manifest for already validated cases."""

    if not cases:
        raise EvalError("SCHEMA_INVALID", "empty case set")
    for case in cases:
        validate_case(case)
    digests = []
    for case in cases:
        digests.append(
            {
                "case_id": case["case_id"],
                "split": case["split"],
                "sha256": sha256_text(canonical_json(case)),
            }
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "contract": CONTRACT_SCHEMA,
        "split": split,
        "fixture_only": True,
        "cases": digests,
        "case_count": len(cases),
        "purpose": "fictional vision-support evaluation only",
    }


def validate_manifest(
    manifest: Any,
    cases: list[dict],
    *,
    split: str | None = None,
    require_case_count: bool = True,
) -> dict:
    """Verify the manifest, the per-case digest and the split authorization."""

    _require(
        isinstance(manifest, dict)
        and manifest.get("schema") == MANIFEST_SCHEMA
        and manifest.get("contract") == CONTRACT_SCHEMA,
        "MANIFEST_INVALID",
        "schema",
    )
    _require(
        isinstance(manifest.get("cases"), list) and manifest["cases"],
        "MANIFEST_INVALID",
        "cases",
    )
    _require(
        manifest.get("fixture_only") is True,
        "FIXTURE_ONLY_REQUIRED",
        "manifest",
    )
    locked = split or manifest.get("split")
    _require(
        isinstance(locked, str) and locked in ALLOWED_SPLITS,
        "MANIFEST_INVALID",
        "split",
    )
    _require(
        manifest.get("split") == locked,
        "MANIFEST_INVALID",
        "manifest split does not match the authorized split",
    )
    entries = manifest["cases"]
    by_id = {entry.get("case_id"): entry for entry in entries if isinstance(entry, dict)}
    _require(
        len(by_id) == len(entries) == len(cases) and set(by_id) == {c["case_id"] for c in cases},
        "MANIFEST_INVALID",
        "case set does not match manifest",
    )
    for case in cases:
        validate_case(case, allowed_split=locked)
        entry = by_id[case["case_id"]]
        _require(
            entry.get("sha256") == sha256_text(canonical_json(case)),
            "HASH_MISMATCH",
            case["case_id"],
        )
        _require(
            entry.get("split") == case["split"],
            "MANIFEST_INVALID",
            case["case_id"],
        )
    if require_case_count:
        _require(
            manifest.get("case_count") == len(cases),
            "MANIFEST_INVALID",
            "case_count",
        )
    return manifest


def validate_manifest_for_run(
    manifest: Any,
    cases: list[dict],
    *,
    allow_test: bool = False,
) -> dict:
    """Run gate: ``test`` is locked unless the operator explicitly unlocks it."""

    validate_manifest(manifest, cases, split=manifest.get("split") if isinstance(manifest, dict) else None)
    split = manifest["split"]
    if split == "test" and not allow_test:
        raise EvalError("TEST_SPLIT_LOCKED", "test requires --allow-test")
    return manifest


def validate_cases_file(
    payload: Any,
    *,
    allow_test: bool = False,
    source: str = "cases",
) -> tuple[list[dict], dict]:
    """Validate the ``cases.json`` bundle and its embedded manifest.

    Accepted layout::

        {"schema": "lm_trainer.eval-cases.v1", "manifest": {...},
         "cases": [...]}

    The manifest digest is recomputed over every case, so a modified case,
    a swapped-in ``test`` case, or an unauthorized purpose is rejected.
    """

    _require(
        isinstance(payload, dict)
        and payload.get("schema") == "lm_trainer.eval-cases.v1",
        "SCHEMA_INVALID",
        source,
    )
    cases = payload.get("cases")
    _require(isinstance(cases, list) and cases, "SCHEMA_INVALID", f"{source}: cases")
    split = payload.get("split")
    _require(
        isinstance(split, str) and split in ALLOWED_SPLITS,
        "SCHEMA_INVALID",
        f"{source}: split",
    )
    for case in cases:
        validate_case(case, source=f"{source}:{case.get('case_id') if isinstance(case, dict) else '?'}", allowed_split=split)
    validate_manifest_for_run(payload.get("manifest"), cases, allow_test=allow_test)
    _require(payload.get("fixture_only") is True, "FIXTURE_ONLY_REQUIRED", source)
    return cases, payload["manifest"]


def load_cases(
    path: Path | str, *, allow_test: bool = False
) -> tuple[list[dict], dict]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvalError("SCHEMA_INVALID", f"cannot read {path.name}") from exc
    return validate_cases_file(payload, allow_test=allow_test, source=path.name)


