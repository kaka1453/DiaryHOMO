"""Versioned, tokenizer-independent source facts and candidate views."""

from __future__ import annotations

import itertools
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType

SOURCE_SCHEMA = "lm_trainer.conversation-source.v1"
WINDOW_SCHEMA = "conversation.stage3.structured.v3"
RENDERER_VERSION = "lm_trainer.conversation-renderer.v1"
KINDS = ("text", "image", "file", "audio", "video", "unknown")
_ID = re.compile(r"[\w.:-]{1,160}")


class ConversationError(ValueError):
    """Privacy-safe, static diagnostics for contract or rendering failures."""


def require(condition, reason):
    if not condition:
        raise ConversationError(reason)


def identifier(value):
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def id_list(value):
    return (
        isinstance(value, list)
        and all(map(identifier, value))
        and len(set(value)) == len(value)
    )


def timestamp(value):
    try:
        result = datetime.fromisoformat(value)
        if result.utcoffset() is not None:
            return result
    except (ValueError, TypeError, OverflowError):
        pass
    raise ConversationError("source timestamps must be timezone-aware ISO 8601")


@dataclass(frozen=True, slots=True)
class Event:
    source_id: str
    position: int
    speaker: str
    person: str
    time: datetime
    kind: str
    body_available: bool
    readable: bool
    selection: str
    content: str | None
    available_for_training: bool
    witnesses: tuple[str, ...]
    visible_from: int


@dataclass(frozen=True, slots=True)
class Relation:
    id: str
    owner_id: str
    kind: str
    resolution: str
    anchor_id: str | None
    speaker: str | None


@dataclass(frozen=True, slots=True)
class Conversation:
    id: str
    event_group: str
    mode: str
    events: Mapping[str, Event]
    relations: Mapping[str, tuple[Relation, ...]]
    aliases: Mapping[str, str]
    authors: Mapping[str, str]
    first_observed: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class Catalog:
    conversations: Mapping[str, Conversation]
    by_source: Mapping[str, Conversation]


def _event(raw):
    require(
        isinstance(raw, dict)
        and all(identifier(raw.get(k)) for k in ("source_id", "speaker", "person")),
        "invalid source identity",
    )
    position = raw.get("position")
    require(type(position) is int and position >= 0, "invalid source position")
    kind, selection = raw.get("kind"), raw.get("selection")
    body, readable = raw.get("body_available"), raw.get("readable")
    content = raw.get("content")
    training = raw.get("available_for_training", False)
    require(
        kind in KINDS
        and selection in ("text", "structure", "hidden")
        and all(type(v) is bool for v in (body, readable, training)),
        "invalid source availability or selection",
    )
    require(
        (body and readable and isinstance(content, str) and bool(content.strip()))
        if selection == "text"
        else "content" not in raw,
        "source selected text is empty/protected or unselected body was supplied",
    )
    require(
        not training or selection == "text", "non-text source cannot authorize a target"
    )
    basis = raw.get("selection_basis")
    witnesses = raw.get("witness_source_ids", [])
    visible_from = raw.get("visible_from_position", position)
    require(
        (basis == "online" and selection in ("text", "structure"))
        or (basis == "witness" and selection == "text")
        or (basis == "hidden" and selection == "hidden"),
        "source requires independent selection_basis (online, witness or hidden)",
    )
    has_witness = "witness_source_ids" in raw or "visible_from_position" in raw
    require(
        id_list(witnesses)
        and type(visible_from) is int
        and (
            bool(witnesses) and visible_from > position
            if basis == "witness"
            else not has_witness
        ),
        "invalid or missing selection witness",
    )
    return Event(
        raw["source_id"],
        position,
        raw["speaker"],
        raw["person"],
        timestamp(raw.get("timestamp")),
        kind,
        body,
        readable,
        selection,
        content,
        training,
        tuple(witnesses),
        visible_from,
    )


def _relations(raw, events, authors):
    require(isinstance(raw, list), "relations must be a list")
    by_owner, seen = {}, set()
    base = {"id", "owner_id", "kind", "resolution"}
    for val in raw:
        require(
            isinstance(val, dict)
            and identifier(val.get("id"))
            and identifier(val.get("owner_id"))
            and val["id"] not in seen
            and val["owner_id"] in events,
            "invalid relation id or owner",
        )
        eid, owner = val["id"], val["owner_id"]
        kind, resolution = val.get("kind"), val.get("resolution")
        require(
            kind in ("reply", "quote", "mention")
            and resolution
            in ("resolved", "unknown", "protected", "display_handle_only"),
            "invalid relation kind or resolution",
        )
        require(
            resolution != "display_handle_only" or kind == "mention",
            "display_handle_only is only a mention",
        )
        anchor, speaker = None, None
        allowed = base
        if resolution == "resolved" and kind != "mention":
            allowed = base | {"anchor_id"}
            anchor = val.get("anchor_id")
            require(
                identifier(anchor)
                and anchor in events
                and events[anchor].position < events[owner].position,
                "resolved quote/reply must anchor to an earlier source",
            )
        elif resolution == "resolved":
            allowed = base | {"speaker"}
            speaker = val.get("speaker")
            require(
                identifier(speaker) and speaker in authors,
                "resolved mention must name a known account",
            )
        require(
            set(val) == allowed,
            "unresolved or protected relation cannot supply anchor, identity or body",
        )
        seen.add(eid)
        by_owner.setdefault(owner, []).append(
            Relation(eid, owner, kind, resolution, anchor, speaker)
        )
    return {owner: tuple(items) for owner, items in by_owner.items()}


def _conversation(raw):
    require(
        isinstance(raw, dict)
        and identifier(raw.get("id"))
        and identifier(raw.get("event_group"))
        and raw.get("mode") in ("direct", "group")
        and isinstance(raw.get("events"), list)
        and raw["events"],
        "invalid source conversation envelope",
    )
    events, authors = {}, {}
    prev = None
    for item in raw["events"]:
        event = _event(item)
        require(
            event.source_id not in events
            and (
                prev is None
                or (event.position > prev.position and event.time >= prev.time)
            ),
            "source events must be unique and ordered by position and time",
        )
        require(
            authors.setdefault(event.speaker, event.person) == event.person,
            "account changes authorship within a conversation",
        )
        events[event.source_id] = prev = event
    require(
        raw["mode"] != "direct"
        or len({e.speaker for e in events.values() if e.selection == "text"}) <= 2,
        "direct conversation has multiple other speakers",
    )
    relations = _relations(raw.get("relations", []), events, authors)
    for event in events.values():
        for wid in event.witnesses:
            witness = events.get(wid)
            require(
                witness is not None
                and witness.selection == "text"
                and not witness.witnesses
                and event.position < witness.position < event.visible_from
                and any(
                    edge.anchor_id == event.source_id for edge in relations.get(wid, ())
                ),
                "selection witness must be a later, online quote/reply owner",
            )
        if event.witnesses:
            require(
                event.visible_from
                == max(events[wid].position for wid in event.witnesses) + 1,
                "selection activation must follow the last witness",
            )
    # Derived only from trusted chronological availability, never from a window roster.
    first = {}
    for event in sorted(events.values(), key=lambda e: (e.visible_from, e.position)):
        if event.selection != "hidden":
            first.setdefault(event.speaker, event.visible_from)
    aliases = {speaker: f"P{i}" for i, speaker in enumerate(first, 1)}
    return Conversation(
        raw["id"],
        raw["event_group"],
        raw["mode"],
        MappingProxyType(events),
        MappingProxyType(relations),
        MappingProxyType(aliases),
        MappingProxyType(authors),
        MappingProxyType(first),
    )


def parse_catalog(raw):
    """Validate independent provenance; no I/O, authorization or platform SDK."""
    require(
        isinstance(raw, dict)
        and raw.get("schema") == SOURCE_SCHEMA
        and isinstance(raw.get("conversations"), list)
        and raw["conversations"],
        "unknown conversation source schema or empty catalog",
    )
    conversations, sources = {}, {}
    for val in raw["conversations"]:
        conv = _conversation(val)
        require(
            conv.id not in conversations and not sources.keys() & conv.events.keys(),
            "duplicate conversation or source_id in source catalog",
        )
        conversations[conv.id] = conv
        sources.update(dict.fromkeys(conv.events, conv))
    return Catalog(MappingProxyType(conversations), MappingProxyType(sources))


@dataclass(frozen=True, slots=True)
class Observation:
    conversation: Conversation
    events: tuple[Event, ...]


def observe(catalog, conversation_id, observed_ids):
    """Build the causal projection of what the caller actually observed (no target)."""
    require(
        identifier(conversation_id) and conversation_id in catalog.conversations,
        "unknown observation conversation",
    )
    return _observe(catalog.conversations[conversation_id], tuple(observed_ids))


def _observe(conv, ids):
    require(
        id_list(list(ids)) and all(sid in conv.events for sid in ids),
        "unknown or duplicate observation source",
    )
    events = tuple(conv.events[sid] for sid in ids)
    require(
        all(a.position < b.position for a, b in itertools.pairwise(events)),
        "observation sources must be in source order",
    )
    seen = set(ids)
    return Observation(
        conv,
        tuple(
            e
            for e in events
            if e.selection != "hidden" and seen.issuperset(e.witnesses)
        ),
    )


def _contiguous(conv, previous, current):
    return (
        previous is not None
        and previous.speaker == current.speaker
        and current.position == previous.position + 1
        and (current.time - previous.time).total_seconds() <= 20
        and not conv.relations.get(current.source_id)
        and previous.kind == current.kind == "text"
    )


def author_blocks(catalog, conversation_id, person, observed_ids):
    """Available SFT-independent CPT candidates, without joining interrupted runs."""
    obs = observe(catalog, conversation_id, observed_ids)
    blocks, previous = [], None
    for event in obs.events:
        if event.selection != "text" or event.person != person:
            previous = None
            continue
        if _contiguous(obs.conversation, previous, event):
            blocks[-1].append(event)
        else:
            blocks.append([event])
        previous = event
    return tuple(tuple(block) for block in blocks)


@dataclass(frozen=True, slots=True)
class Window:
    id: str
    conversation: Conversation
    target_person: str
    timeline: tuple[str, ...]
    targets: tuple[str, ...]
    opportunities: tuple[tuple[Event, ...], ...]
    boundaries: Mapping[str, int]

    def observation(self, focus_id):
        require(
            identifier(focus_id) and focus_id in self.boundaries,
            "unknown generation focus",
        )
        n = self.boundaries[focus_id]
        require(self.timeline[n] == focus_id, "invalid exclusive causal boundary")
        return _observe(self.conversation, self.timeline[:n])


def _rows(row, key):
    val = row.get(key)
    require(
        isinstance(val, list) and all(isinstance(m, dict) for m in val),
        "invalid messages, timeline or generation blocks",
    )
    return val


def _check_refs(value, expected):
    require(
        isinstance(value, list)
        and all(
            isinstance(ref, list) and len(ref) == 2 and all(map(identifier, ref))
            for ref in value
        ),
        "invalid history or audit relation references",
    )
    actual = [tuple(pair) for pair in value]
    require(
        len(actual) == len(expected) and set(actual) == expected,
        "history or audit relation reference does not match its canonical owner",
    )


def adapt_window(row, catalog):
    """Validate all redundant v3 fields against immutable canonical source facts."""
    require(
        isinstance(row, dict) and row.get("schema") == WINDOW_SCHEMA,
        "unknown structured conversation schema",
    )
    require(
        all(identifier(row.get(k)) for k in ("id", "conversation", "target_person"))
        and row["conversation"] in catalog.conversations,
        "invalid window identity",
    )
    conv = catalog.conversations[row["conversation"]]
    target = row["target_person"]
    require(
        sum(person == target for person in conv.authors.values()) == 1,
        "target_person must identify one account in the conversation",
    )
    timeline = _rows(row, "timeline")
    ids = [slot.get("id") for slot in timeline]
    require(
        ids and id_list(ids) and all(sid in conv.events for sid in ids),
        "invalid timeline source closure",
    )
    events = [conv.events[sid] for sid in ids]
    require(
        all(a.position < b.position for a, b in itertools.pairwise(events)),
        "timeline must be in source position order",
    )
    selected_slots = set(ids)
    online = [e for e in events if not e.witnesses]
    require(
        online
        and all(
            e.source_id in selected_slots
            for e in conv.events.values()
            if online[0].position <= e.position <= events[-1].position
            and e.selection != "hidden"
            and not e.witnesses
        ),
        "timeline omits an online source within the selected window",
    )
    mask = row.get("target_mask")
    messages = _rows(row, "messages")
    require(
        isinstance(mask, list)
        and len(mask) == len(messages)
        and all(type(v) is bool for v in mask)
        and any(mask),
        "invalid target_mask",
    )
    text_events, slot_targets, boundaries = [], [], {}
    for i, (slot, event) in enumerate(zip(timeline, events)):
        text, structure = event.selection == "text", event.selection == "structure"
        require(
            (text or structure)
            and slot.get("text_slot") is text
            and slot.get("body_available") is event.body_available
            and slot.get("structure_selected") is structure
            and slot.get("event") is (structure or event.kind != "text")
            and type(slot.get("target")) is bool
            and (text or not slot["target"]),
            "timeline visibility, body_available or structure disagrees with source",
        )
        if text:
            text_events.append(event)
            slot_targets.append(slot["target"])
        if slot["target"]:
            boundaries[event.source_id] = i
    require(
        len(text_events) == len(messages) and mask == slot_targets,
        "target_mask or text slots disagree with timeline",
    )
    blocks, opportunities = [], []
    previous = None
    for event, message, selected in zip(text_events, messages, mask):
        require(
            message.get("source_id") == event.source_id
            and type(message.get("position")) is int
            and message["position"] == event.position
            and message.get("speaker") == event.speaker
            and message.get("person") == event.person
            and timestamp(message.get("timestamp")) == event.time
            and message.get("content") == event.content
            and message.get("supervised") is selected
            and not {
                "role",
                "tool_calls",
                "reasoning_content",
                "reasoning",
                "tool_call_id",
            }
            & message.keys(),
            "message body, identity or supervision disagrees with source",
        )
        require(
            not selected
            or (
                event.person == target
                and event.kind == "text"
                and event.available_for_training
                and not event.witnesses
            ),
            "target source is not eligible visible text by target_person",
        )
        if _contiguous(conv, previous, event) and blocks[-1]["target"] is selected:
            blocks[-1]["ids"].append(event.source_id)
            if selected:
                opportunities[-1].append(event)
        else:
            blocks.append(
                {
                    "speaker": event.speaker,
                    "person": event.person,
                    "target": selected,
                    "ids": [event.source_id],
                }
            )
            if selected:
                opportunities.append([event])
        previous = event
    supplied = _rows(row, "generation_blocks")
    require(
        all(type(b.get("target")) is bool for b in supplied) and supplied == blocks,
        "generation_blocks do not exactly match continuous source bubbles",
    )
    supplied_boundaries = row.get("target_causal_prefix_lengths")
    require(
        row.get("target_ids") == list(boundaries)
        and isinstance(supplied_boundaries, dict)
        and all(type(n) is int for n in supplied_boundaries.values())
        and supplied_boundaries == boundaries,
        "target_ids or exclusive causal boundaries disagree",
    )
    history_refs = {
        (event.source_id, edge.id)
        for event in text_events
        for edge in conv.relations.get(event.source_id, ())
    }
    _check_refs(row.get("history_relation_refs"), history_refs)
    _check_refs(
        row.get("target_relation_audit_only"),
        {p for p in history_refs if p[0] in boundaries},
    )
    return Window(
        row["id"],
        conv,
        target,
        tuple(ids),
        tuple(boundaries),
        tuple(tuple(b) for b in opportunities),
        MappingProxyType(boundaries),
    )


def inspect_candidate(row, catalog):
    require(
        isinstance(row, dict)
        and row.get("training_authorized") is False
        and row.get("split", "unassigned") == "unassigned"
        and row.get("disposition")
        in ("adapter_candidate", "review_only", "retained_no_exposure"),
        "expected a read-only, unassigned candidate, not training approval",
    )
    return adapt_window(row, catalog)


HOLD_VIEWS = ("independent_opening", "author_continuation", "opening", "continuation")


def validate_route(windows, routes):
    """Return default window IDs; alternate routes never confer training permission."""
    windows = tuple(windows)
    known = {w.id: w for w in windows}
    require(len(known) == len(windows), "duplicate window id")
    seen, used, selected = set(), set(), []
    for route in routes:
        require(
            isinstance(route, dict)
            and identifier(route.get("window_id"))
            and route["window_id"] in known
            and route["window_id"] not in seen
            and type(route.get("default")) is bool,
            "invalid or duplicate routing entry",
        )
        wid = route["window_id"]
        seen.add(wid)
        if route["default"]:
            targets = set(known[wid].targets)
            require(
                not used & targets,
                "duplicate SFT target source_id across default routes",
            )
            used.update(targets)
            selected.append(wid)
    require(seen == known.keys(), "routing does not cover the input windows")
    return tuple(selected)
