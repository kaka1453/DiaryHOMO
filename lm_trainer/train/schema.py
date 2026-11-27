"""Lightweight v1 corpus schema and fail-closed data preflight (no tokenizer)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from lm_trainer.config.common import asset_path
from lm_trainer.conversation import (
    RENDERER_VERSION,
    WINDOW_SCHEMA,
    ConversationError,
    adapt_window,
    parse_catalog,
    render_observation,
    validate_route,
)
from lm_trainer.conversation.contract import HOLD_VIEWS, Window
from lm_trainer.conversation.renderer import target_turn
from lm_trainer.template_policy import validate_policy

CORPUS_SCHEMA = "lm_trainer.corpus.v1"
RELEASE_SCHEMA = "lm_trainer.release.v1"
CONVERSATION_RELEASE_SCHEMA = "lm_trainer.release.v2"
GROUP_SCHEMA = "lm_trainer.source-groups.v1"
INDEX_SCHEMA = "lm_trainer.source-index.v1"
OBJECTIVES = {"causal_text": "cpt", "assistant_only_messages": "sft"}
SPLITS = ("train", "dev", "test")
DATA_DEFAULTS = {
    "data_mode": "legacy",
    "objective": None,
    "manifest_path": None,
    "sources": None,
    "target_loss_token_share": "natural",
    "token_budget": None,
    "max_effective_epochs": None,
    "max_sample_repeats": None,
    "corpus_schema_allowlist": None,
    "template_policy": None,
    "add_bos": False,
    "add_eos": True,
    "allow_eos_only": False,
    "supervise_target_eos": False,
    "overlength_policy": "reject",
}
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_SAFE_ID = re.compile(r"[\w.:-]{1,160}")
_LEGACY_ENVELOPE = frozenset(
    (
        "schema",
        "kind",
        "target_person",
        "event_group",
        "source_ids",
        "source_index_ref",
        "permissions",
        "review_status",
        "candidate_status",
        "split",
        "messages",
        "target_mask",
        "training_authorized",
        "fixture_only",
        "owner_review_only",
        "scope",
    )
)


class DataContractError(ValueError):
    def __init__(self, reason, *, path=None, sample_id=None, line=None):
        self.diagnostic = {"reason": reason}
        if path is not None:
            self.diagnostic["file"] = path
        if isinstance(sample_id, str) and _SAFE_ID.fullmatch(sample_id):
            self.diagnostic["sample_id"] = sample_id
        if line is not None:
            self.diagnostic["line"] = line
        super().__init__(json.dumps(self.diagnostic, ensure_ascii=False))


def nonblank(value):
    return isinstance(value, str) and bool(value.strip())


def configure_data(runtime):
    """Validate options without opening any data; safe for --dry-run."""
    options = {key: runtime.get(key, value) for key, value in DATA_DEFAULTS.items()}
    mode, objective = options["data_mode"], options["objective"]
    if mode not in ("legacy", "fixture", "release"):
        raise ValueError("data_mode must be legacy, fixture or release")
    legacy = mode == "legacy"
    if legacy:
        if options["manifest_path"] is not None or objective not in (None, "legacy"):
            raise ValueError(
                "legacy requires data_file_path, not a manifest or structured objective"
            )
        if any(
            options[key] != DATA_DEFAULTS[key]
            for key in DATA_DEFAULTS
            if key not in ("data_mode", "objective", "manifest_path")
        ):
            raise ValueError("legacy does not accept structured data options")
        options["objective"] = "legacy"
    elif (
        not isinstance(objective, str)
        or objective not in OBJECTIVES
        or not nonblank(options["manifest_path"])
        or runtime.get("data_file_path") is not None
    ):
        raise ValueError(
            "fixture/release require manifest_path and an explicit objective, not data_file_path"
        )
    schemas = options["corpus_schema_allowlist"]
    if schemas is None:
        schemas = [CORPUS_SCHEMA]
    if (
        not isinstance(schemas, list)
        or not schemas
        or any(
            not isinstance(name, str)
            or (
                name != WINDOW_SCHEMA
                and (
                    name.startswith("conversation.stage3.structured.")
                    or not re.fullmatch(r"[\w.-]+\.v1", name)
                )
            )
            for name in schemas
        )
        or len(set(schemas)) != len(schemas)
    ):
        raise ValueError(
            "corpus_schema_allowlist must contain unique compatible v1 aliases or the registered v3 window"
        )
    if WINDOW_SCHEMA in schemas and (
        schemas != [WINDOW_SCHEMA] or objective != "assistant_only_messages"
    ):
        raise ValueError(
            "v3 windows require an explicit, standalone structured SFT allowlist"
        )
    if WINDOW_SCHEMA in schemas and runtime.get("backend", "hf") != "hf":
        raise ValueError(
            "v3 conversation backend requires HF; Unsloth parity has not been verified"
        )
    if not legacy:
        options["corpus_schema_allowlist"] = schemas
    sources = options["sources"]
    if sources is not None and (
        not isinstance(sources, list)
        or not sources
        or not all(isinstance(s, str) and _SAFE_ID.fullmatch(s) for s in sources)
        or len(set(sources)) != len(sources)
    ):
        raise ValueError("sources must be a nonempty list of unique source names")
    share = options["target_loss_token_share"]
    if share != "natural":
        if (
            not isinstance(share, dict)
            or not share
            or not all(
                isinstance(name, str)
                and _SAFE_ID.fullmatch(name)
                and type(v) in (int, float)
                and 0 < v <= 1
                for name, v in share.items()
            )
            or not math.isclose(math.fsum(share.values()), 1.0, rel_tol=0, abs_tol=1e-9)
        ):
            raise ValueError(
                "target_loss_token_share must be natural or positive shares summing to 1"
            )
        if sources is not None and set(share) != set(sources):
            raise ValueError("target_loss_token_share keys must match sources")
    for key in ("token_budget", "max_sample_repeats"):
        value = options[key]
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{key} must be a positive integer")
    cap = options["max_effective_epochs"]
    if cap is not None and (type(cap) not in (int, float) or not 0 < cap < math.inf):
        raise ValueError("max_effective_epochs must be finite and positive")
    for key in ("add_bos", "add_eos", "allow_eos_only", "supervise_target_eos"):
        if type(options[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if options["overlength_policy"] != "reject":
        raise ValueError("only overlength_policy=reject is supported")
    if objective == "assistant_only_messages":
        if any(
            options[key] != DATA_DEFAULTS[key]
            for key in ("add_bos", "add_eos", "allow_eos_only")
        ):
            raise ValueError("CPT boundary settings are not SFT template settings")
        options["template_policy"] = validate_policy(options["template_policy"])
    elif (
        objective == "causal_text"
        and runtime.get("init_adapter")
        and options["template_policy"] is not None
        and not options["supervise_target_eos"]
    ):
        # CPT does not render the template; bind the downstream policy to its adapter.
        options["template_policy"] = validate_policy(options["template_policy"])
    elif options["template_policy"] is not None or options["supervise_target_eos"]:
        raise ValueError("template_policy and supervise_target_eos are SFT-only")
    if objective == "assistant_only_messages" and share != "natural":
        raise ValueError("explicit token mixture is supported only for CPT")
    return options


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON key")
        result[name] = value
    return result


def _invalid_constant(_):
    raise ValueError("non-finite JSON number")


def _json(value):
    return json.loads(
        value, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
    )


def _decode(payload, path):
    try:
        return payload.decode("utf-8-sig")
    except UnicodeError:
        raise DataContractError("invalid UTF-8", path=path) from None


def _read_json(value, path, line=None):
    try:
        return _json(value)
    except (ValueError, RecursionError):
        raise DataContractError(
            "invalid JSON or duplicate keys", path=path, line=line
        ) from None


def _rows(payload, path):
    rows = []
    # splitlines() would incorrectly split unescaped U+2028/U+2029 inside JSON strings.
    for i, line in enumerate(_decode(payload, path).split("\n"), 1):
        if not line.strip():
            continue
        row = _read_json(line, path, i)
        if not isinstance(row, dict):
            raise DataContractError("record must be a JSON object", path=path, line=i)
        rows.append((i, row))
    if not rows:
        raise DataContractError("empty data file", path=path)
    return rows


def _literal_relative(value):
    return (
        nonblank(value)
        and not any(c in value for c in ("*", "?", "[", "]", "::", "\\", "\x00"))
        and all(part not in ("", ".", "..") for part in value.split("/"))
        and not Path(value).is_absolute()
    )


def _verified_file(root, spec, seen):
    if not isinstance(spec, dict) or not _literal_relative(spec.get("path")):
        raise DataContractError(
            "manifest reference must be a literal relative path without traversal"
        )
    name = spec["path"]
    if name in seen:
        raise DataContractError("duplicate referenced path", path=name)
    digest = spec.get("sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise DataContractError("invalid SHA-256 declaration", path=name)
    seen.add(name)
    try:
        path = asset_path(root, name, "manifest reference", ".")
        # Pin each directory component while descending. A rename/symlink swap
        # between lstat and open must not redirect the bytes being authenticated.
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in path.relative_to(root).parts[:-1]:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                os.close(fd)
                fd = child
            file_fd = os.open(
                path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=fd
            )
            with os.fdopen(file_fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("not a file")
                payload = stream.read()
        finally:
            os.close(fd)
    except (OSError, ValueError):
        raise DataContractError(
            "missing or unsafe manifest path (symlinks forbidden)", path=name
        ) from None
    if hashlib.sha256(payload).hexdigest() != digest.lower():
        raise DataContractError("SHA-256 mismatch", path=name)
    return payload


def _ids(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(nonblank(s) for s in value)
        and len(value) == len(set(value))
    )


def _train_permissions(perms):
    return (
        isinstance(perms, dict)
        and all(
            type(perms.get(k)) is bool for k in ("local_review", "train", "external")
        )
        and perms["train"] is True
    )


def _approval_flags(row):
    return (
        row.get("review_status", "approved") == "approved"
        and row.get("candidate_status", "approved") == "approved"
        and row.get("owner_review_only", False) is False
        and row.get("training_authorized", True) is True
        and ("permissions" not in row or _train_permissions(row["permissions"]))
    )


def _permitted(row, error, split):
    if not _train_permissions(row.get("permissions")):
        error("permissions: expected independent booleans and train=true")
    if row.get("review_status") != "approved" or not _approval_flags(row):
        error("record is not approved for training")
    if row.get("split") != split:
        error(f"only split={split} is allowed in this input")


def _record(row, *, kind, target, schemas, error, split="train"):
    _permitted(row, error, split)
    if (
        row.get("schema") not in schemas
        or row.get("kind") != kind
        or row.get("target_person") != target
    ):
        error("incompatible schema, kind or target_person")
    if not all(nonblank(row.get(k)) for k in ("id", "event_group", "source_index_ref")):
        error("missing id, event_group or source_index_ref")
    if kind == "cpt":
        if not nonblank(row.get("text")) or any(
            k in row for k in ("messages", "target_mask")
        ):
            error("CPT requires nonempty text, without messages or target_mask")
        if not _ids(row.get("source_ids")):
            error("CPT source_ids must be nonempty and unique")
        return set(row["source_ids"]), set(), len(row["text"])
    messages, mask = row.get("messages"), row.get("target_mask")
    if (
        "text" in row
        or not isinstance(messages, list)
        or not isinstance(mask, list)
        or len(messages) != len(mask)
        or not all(type(v) is bool for v in mask)
        or not any(mask)
    ):
        error(
            "SFT requires messages, no text, and an aligned boolean target_mask with a target"
        )
    source_ids, targets, speakers = set(), set(), set()
    for i, (message, selected) in enumerate(zip(messages, mask)):
        if (
            not isinstance(message, dict)
            or not nonblank(message.get("speaker"))
            or message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str)
            or not _ids(message.get("source_ids"))
            or any(
                key in message
                for key in (
                    "tool_calls",
                    "tool_call_id",
                    "reasoning",
                    "reasoning_content",
                )
            )
        ):
            error("invalid message, source_ids or unsupported reasoning/tool protocol")
        role, speaker = message["role"], message["speaker"]
        if role == "system":
            if i:
                error(
                    "only an initial, explicitly configured system instruction is supported"
                )
        elif speaker != target:
            speakers.add(speaker)
        ids = set(message["source_ids"])
        source_ids.update(ids)
        if selected:
            if (
                role != "assistant"
                or speaker != target
                or not nonblank(message["content"])
            ):
                error(
                    "selected turn must be nonempty assistant content by target_person"
                )
            if targets & ids:
                error("duplicate SFT target source_id within a sample")
            targets.update(ids)
    if len(speakers) > 1:
        error("multiple non-target speakers require an audited serialization strategy")
    return source_ids, targets, sum(len(m["content"]) for m in messages)


@dataclass(frozen=True)
class Sample:
    row: dict
    source: str
    path: str
    line: int
    characters: int
    conversation: Window | None = None

    def fail(self, reason):
        raise DataContractError(
            reason, path=self.path, line=self.line, sample_id=self.row.get("id")
        )


@dataclass(frozen=True)
class ValidatedData:
    plan: dict
    samples: tuple[Sample, ...]
    events: frozenset = frozenset()
    origin_ids: frozenset = frozenset()


def _has_envelope(row):
    pending = [row]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if _LEGACY_ENVELOPE.intersection(value):
                return True
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return False


def _legacy(runtime, options):
    path = Path(runtime["data_file_path"])
    label = "data_file_path"
    try:
        if not path.is_file():
            raise OSError
        payload = path.read_bytes()
    except OSError:
        raise DataContractError(
            "missing or unreadable legacy data", path=label
        ) from None
    text = _decode(payload, label)
    if text.lstrip().startswith("["):
        rows = list(enumerate(_read_json(text, label), 1))
    elif path.suffix == ".json":
        try:
            row = _json(text)
        except ValueError:
            rows = _rows(payload, label)
        else:
            rows = [(1, row)]
    else:
        rows = _rows(payload, label)
    if not rows:
        raise DataContractError("empty data file", path=label)
    samples = []
    for line, row in rows:
        if not isinstance(row, dict) or _has_envelope(row):
            raise DataContractError(
                "legacy accepts plain text records, not structured corpus envelopes",
                path=label,
                line=line,
            )
        if not nonblank(row.get("text")):
            raise DataContractError(
                "legacy requires nonempty text", path=label, line=line
            )
        samples.append(Sample(row, "legacy", label, line, len(row["text"])))
    return ValidatedData(
        {
            **options,
            "plan_schema": "lm_trainer.stage-data.v1",
            "sources": ["legacy"],
            "seed": runtime.get("seed", 42),
            "max_length": runtime.get("max_length"),
            "fixture_only": False,
            "data_sha256": hashlib.sha256(payload).hexdigest(),
            "statistics": {
                "legacy": {
                    "samples": len(samples),
                    "characters": sum(s.characters for s in samples),
                }
            },
            "cross_stage_overlap_source_ids": 0,
        },
        tuple(samples),
    )


def _group_ledger(payload, path):
    groups, event_splits, source_splits = {}, {}, {}
    for line, row in _rows(payload, path):

        def fail(reason, line=line, row=row):
            raise DataContractError(
                reason, path=path, line=line, sample_id=row.get("id")
            )

        if (
            row.get("schema") != GROUP_SCHEMA
            or not all(
                nonblank(row.get(k)) for k in ("id", "event_group", "target_person")
            )
            or row.get("split") not in SPLITS
            or row.get("kind") not in ("cpt", "sft")
            or not _ids(row.get("source_ids"))
        ):
            fail("invalid source-group entry")
        if row["id"] in groups:
            fail("duplicate sample id in source-group ledger")
        split = row["split"]
        if event_splits.setdefault(row["event_group"], split) != split:
            fail("event_group crosses splits")
        if any(
            source_splits.setdefault(sid, split) != split for sid in row["source_ids"]
        ):
            fail("source_id crosses splits")
        groups[row["id"]] = row
    return groups


def _source_index(payload, path):
    index = {}
    for line, row in _rows(payload, path):
        if (
            row.get("schema") != INDEX_SCHEMA
            or not nonblank(row.get("source_id"))
            or row.get("disposition") not in ("C", "S", "F", "X")
            or row.get("review_status")
            not in ("approved", "pending_owner", "owner_review_only")
            or row["source_id"] in index
        ):
            raise DataContractError(
                "invalid or conflicting source-index entry", path=path, line=line
            )
        index[row["source_id"]] = row
    return index


def _conversation_bundle(data, root, seen, ledger, index, fixture, split):
    """Strict additions for release v2; all auxiliary bytes pass the v1 hash gate."""

    def read(key, *, jsonl=False):
        spec = data.get(key)
        raw = _verified_file(root, spec, seen)
        if jsonl:
            return [row for _, row in _rows(raw, spec["path"])]
        return _read_json(_decode(raw, spec["path"]), spec["path"])

    raw = read("conversation_source")
    if not isinstance(raw, dict) or raw.get("fixture_only") is not fixture:
        raise DataContractError(
            "conversation source fixture scope disagrees with release v2"
        )
    if raw.get("split") != split:
        raise DataContractError(
            "conversation source catalog must be for the requested split"
        )
    try:
        catalog = parse_catalog(raw)
    except ConversationError as exc:
        raise DataContractError(
            str(exc), path=data["conversation_source"]["path"]
        ) from None
    routes = read("routing", jsonl=True)
    splits = data.get("partition_splits")
    declared = {entry["split"] for entry in ledger.values()}
    if (
        not _ids(splits)
        or set(splits) != declared
        or split not in declared
        or not fixture
        and ("train" not in declared or not declared.intersection(("dev", "test")))
    ):
        raise DataContractError(
            "incomplete or conflicting source-group partition allocation"
        )
    allocations, group_splits = {}, {}
    ledger_keys = {
        "schema",
        "id",
        "kind",
        "event_group",
        "target_person",
        "source_ids",
        "split",
    }
    for entry in ledger.values():
        if (
            set(entry) != ledger_keys
            or entry["kind"] != "sft"
            or entry["target_person"] != data["target_person"]
        ):
            raise DataContractError(
                "structured source-group ledger must contain only partition metadata"
            )
        group, assigned = entry["event_group"], entry["split"]
        group_splits[group] = assigned
        for sid in entry["source_ids"]:
            allocations[sid] = (group, assigned)
            # Foreign split payloads are deliberately not opened by preflight.
            if assigned == split:
                conv = catalog.by_source.get(sid)
                if not conv or conv.event_group != group:
                    raise DataContractError(
                        "source-group allocation disagrees with source catalog"
                    )
    for conv in catalog.conversations.values():
        allocated = group_splits.get(conv.event_group)
        if allocated != split or any(
            sid in allocations and allocations[sid] != (conv.event_group, split)
            for sid in conv.events
        ):
            raise DataContractError(
                "conversation catalog contains foreign or unassigned source groups"
            )
    if not fixture and index is None:
        raise DataContractError("release v2 requires an independent source_index")
    if index is not None:
        index_keys = {
            "schema",
            "source_id",
            "disposition",
            "event_group",
            "split",
            "review_status",
            "candidate_status",
            "owner_review_only",
            "training_authorized",
            "permissions",
            "approval_ref",
        }
        if any(
            set(item) - index_keys
            or (
                isinstance(item.get("permissions"), dict)
                and set(item["permissions"]) != {"local_review", "train", "external"}
            )
            for item in index.values()
        ):
            raise DataContractError(
                "structured source_index must contain only allocation and approval metadata"
            )
    if not fixture:
        if index.keys() != allocations.keys():
            raise DataContractError(
                "formal source_index and partition ledger must have exact source coverage"
            )
        for sid, (group, assigned) in allocations.items():
            item = index[sid]
            if (
                item.get("disposition") not in ("S", "F")
                or item.get("split") != assigned
                or item.get("event_group") != group
                or item.get("review_status") != "approved"
                or not _approval_flags(item)
                or item.get("training_authorized") is not True
                or not _train_permissions(item.get("permissions"))
            ):
                raise DataContractError(
                    "missing independent approval or split for a source"
                )
    return catalog, routes


def _conversation_record(sample, catalog, data, kind, split, fixture):
    row, error = sample.row, sample.fail
    if (
        row.get("schema") != WINDOW_SCHEMA
        or kind != "sft"
        or row.get("kind") != "sft"
        or row.get("target_person") != data["target_person"]
        or row.get("fixture_only") is not fixture
        or row.get("scope") != data["scope"]
        or row.get("training_authorized") is not (not fixture)
        or row.get("disposition") != ("adapter_candidate" if fixture else "approved")
        or not fixture
        and not nonblank(row.get("approval_ref"))
        or row.get("candidate_status") != "approved"
        or "text" in row
    ):
        error("structured record has incompatible schema, scope or training approval")
    # Explicit fixture-only authorization for the test path, never for a release.
    _permitted({**row, "training_authorized": True}, error, split)
    if row.get("view") in HOLD_VIEWS:
        error("hold: opening or continuation is not standard SFT")
    canonical_ref = data.get("source_index", data["conversation_source"])["path"]
    if row.get("source_index_ref") != canonical_ref:
        error("structured record must name its verified source index")
    try:
        window = adapt_window(row, catalog)
        for block in window.opportunities:
            target_turn(block)
            render_observation(
                window.observation(block[0].source_id), window.target_person, None
            )
    except ConversationError as exc:
        error(str(exc))
    if row.get("event_group") != window.conversation.event_group:
        error("window event_group disagrees with its source catalog")
    if "source_ids" in row and (
        not _ids(row["source_ids"]) or set(row["source_ids"]) != set(window.timeline)
    ):
        error("record source_ids do not close the structured timeline")
    return window


def preflight_data(runtime, *, split="train"):
    """Authenticate exact parsed bytes once; return a plan and validated in-memory samples."""
    if split not in ("train", "dev"):
        raise ValueError("Only train or dev may be opened.")
    options = configure_data(runtime)
    if options["data_mode"] == "legacy":
        if split != "train":
            raise ValueError("Dev requires a structured manifest.")
        return _legacy(runtime, options)

    manifest = Path(options.pop("manifest_path"))
    fixture = options["data_mode"] == "fixture"
    v3 = options["corpus_schema_allowlist"] == [WINDOW_SCHEMA]
    try:
        if manifest.is_symlink() or not manifest.is_file():
            raise OSError
        payload = manifest.read_bytes()
    except OSError:
        raise DataContractError(
            "missing or unsafe manifest", path="manifest_path"
        ) from None
    data = _read_json(_decode(payload, "manifest_path"), "manifest_path")
    if (
        not isinstance(data, dict)
        or data.get("schema") != (CONVERSATION_RELEASE_SCHEMA if v3 else RELEASE_SCHEMA)
        or data.get("scope") != options["data_mode"]
        or data.get("fixture_only") is not fixture
        or data.get("training_authorized") is not (not fixture)
        or data.get("review_status") != "approved"
        or not _approval_flags({**data, "training_authorized": True})
        or not fixture
        and not nonblank(data.get("approval_ref"))
        or not all(
            isinstance(data.get(k), str) and _SAFE_ID.fullmatch(data[k])
            for k in ("release_id", "target_person", "parent_version")
        )
        or not isinstance(data.get("files"), list)
        or not data["files"]
    ):
        raise DataContractError(
            "invalid schema, scope, authorization, approval or empty manifest",
            path="manifest_path",
        )
    # Check every declaration before opening referenced payloads. A dev manifest
    # is a separate bundle; a test file is never opened by this entry point.
    if any(
        not isinstance(spec, dict) or spec.get("split") != split
        for spec in data["files"]
    ):
        raise DataContractError(
            f"invalid file split ({split} required)", path="manifest_path"
        )
    root, seen = manifest.parent.resolve(), {manifest.name}
    group_spec = data.get("source_group_manifest")
    ledger = _group_ledger(_verified_file(root, group_spec, seen), group_spec["path"])
    index = None
    if "source_index" in data:
        index_spec = data["source_index"]
        index = _source_index(
            _verified_file(root, index_spec, seen), index_spec["path"]
        )

    catalog, routes = (
        _conversation_bundle(data, root, seen, ledger, index, fixture, split)
        if v3
        else (None, None)
    )
    windows = []
    all_ids, cpt_ids, target_ids = set(), set(), set()
    by_kind = defaultdict(lambda: defaultdict(list))
    files = []
    ids_by_source = defaultdict(set)
    for spec in data["files"]:
        raw = _verified_file(root, spec, seen)
        name, kind, source = spec["path"], spec.get("kind"), spec.get("source")
        if (
            kind not in ("cpt", "sft")
            or not isinstance(source, str)
            or not _SAFE_ID.fullmatch(source)
            or type(spec.get("records")) is not int
            or spec["records"] < 1
        ):
            raise DataContractError("invalid file kind, source or count", path=name)
        if (
            not _approval_flags(spec)
            or spec.get("fixture_only", fixture) is not fixture
        ):
            raise DataContractError(
                "file is not approved for the selected scope", path=name
            )
        rows = _rows(raw, name)
        if len(rows) != spec["records"]:
            raise DataContractError("record count mismatch", path=name)
        for line, row in rows:
            sample = Sample(row, source, name, line, 0)
            window = None
            if v3:
                window = _conversation_record(
                    sample, catalog, data, kind, split, fixture
                )
                windows.append(window)
                source_ids, selected = set(window.timeline), set(window.targets)
                chars = sum(len(m["content"]) for m in row["messages"])
            else:
                source_ids, selected, chars = _record(
                    row,
                    kind=kind,
                    target=data["target_person"],
                    schemas=options["corpus_schema_allowlist"],
                    error=sample.fail,
                    split=split,
                )
            if row.get("fixture_only", fixture) is not fixture:
                sample.fail("record fixture_only does not agree with manifest scope")
            if row["id"] in all_ids:
                sample.fail("duplicate record id")
            all_ids.add(row["id"])
            entry = ledger.get(row["id"])
            if (
                entry is None
                or any(
                    row[key] != entry[key]
                    for key in ("event_group", "split", "target_person", "kind")
                )
                or set(entry["source_ids"]) != source_ids
            ):
                sample.fail("record does not match its source-group ledger entry")
            if kind == "sft":
                if selected & target_ids:
                    sample.fail("duplicate SFT target source_id across samples")
                if index is not None and any(
                    sid not in index
                    or index[sid]["disposition"] not in ("S", "F")
                    or not _approval_flags(index[sid])
                    for sid in selected
                ):
                    sample.fail("source-index ruling does not authorize an SFT target")
                target_ids.update(selected)
                policy = options["template_policy"]
                if (
                    not v3
                    and policy is not None
                    and row["messages"][0]["role"] == "system"
                    and (row["messages"][0]["content"] != policy["system_prompt"])
                ):
                    sample.fail(
                        "system instruction must match template_policy.system_prompt"
                    )
            else:
                cpt_ids.update(source_ids)
            ids_by_source[kind, source].update(
                selected if kind == "sft" else source_ids
            )
            by_kind[kind][source].append(Sample(row, source, name, line, chars, window))
        files.append(
            {
                k: spec[k]
                for k in ("path", "sha256", "kind", "source", "records", "split")
            }
        )

    if v3:
        if all_ids != {key for key, row in ledger.items() if row["split"] == split}:
            raise DataContractError(
                "release windows do not cover their complete split allocation"
            )
        try:
            if set(validate_route(windows, routes)) != all_ids:
                raise ConversationError(
                    "release files must contain only default routing windows"
                )
        except ConversationError as exc:
            raise DataContractError(str(exc), path=data["routing"]["path"]) from None

    kind = OBJECTIVES[options["objective"]]
    available = by_kind[kind]
    sources = options["sources"] or sorted(available)
    if not sources or any(name not in available for name in sources):
        raise DataContractError(
            "objective or requested source is missing from the manifest"
        )
    if options["target_loss_token_share"] != "natural" and set(sources) != set(
        options["target_loss_token_share"]
    ):
        raise DataContractError(
            "target_loss_token_share keys must match selected sources"
        )
    # The complete manifest was validated even when this stage selects a subset.
    return ValidatedData(
        {
            **options,
            "plan_schema": "lm_trainer.stage-data.v1",
            **(
                {
                    "release_schema": CONVERSATION_RELEASE_SCHEMA,
                    "renderer_version": RENDERER_VERSION,
                    "partition_splits": data["partition_splits"],
                    "partition_sha256": data["source_group_manifest"]["sha256"].lower(),
                    "source_index_sha256": data["source_index"]["sha256"].lower()
                    if "source_index" in data
                    else None,
                    "conversation_source_sha256": data["conversation_source"][
                        "sha256"
                    ].lower(),
                }
                if v3
                else {}
            ),
            "sources": sources,
            "seed": runtime.get("seed", 42),
            "max_length": runtime.get("max_length"),
            "epochs": runtime.get("epochs", 1),
            "max_steps": runtime.get("max_steps"),
            "batch_size": runtime.get("batch_size"),
            "gradient_accumulation_steps": runtime.get("gradient_accumulation_steps"),
            "fixture_only": fixture,
            "release_id": data["release_id"],
            "split": split,
            "target_person": data["target_person"],
            "parent_version": data["parent_version"],
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "files": [f for f in files if f["kind"] == kind and f["source"] in sources],
            "statistics": {
                s: {
                    "samples": len(available[s]),
                    "events": len(
                        {sample.row["event_group"] for sample in available[s]}
                    ),
                    "characters": sum(sample.characters for sample in available[s]),
                    "cross_stage_overlap_source_ids": len(
                        ids_by_source[kind, s]
                        & (cpt_ids if kind == "sft" else target_ids)
                    ),
                }
                for s in sources
            },
            "cross_stage_overlap_source_ids": len(cpt_ids & target_ids),
        },
        tuple(sample for source in sources for sample in available[source]),
        frozenset(
            row["event_group"] for row in ledger.values() if row["split"] == split
        ),
        frozenset(
            sid
            for row in ledger.values()
            if row["split"] == split
            for sid in row["source_ids"]
        ),
    )
