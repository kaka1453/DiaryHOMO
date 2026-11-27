"""Render observed events to native message roles; no release or platform logic."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from lm_trainer.template_policy import PROTOCOL_MARKERS, chat_template_kwargs

from .contract import require

MEDIA = {
    "image": "[图片]",
    "file": "[附件]",
    "audio": "[音频]",
    "video": "[视频]",
    "unknown": "[未识别媒体]",
    "text": "[正文未选]",
}
# Never allow a message to manufacture a native role header or our own attribution.
_ROLE_HEADER = re.compile(
    r"(?im)^[ \t#>*`\[]*"
    r"(?:system|assistant|developer|user|tool|系统|开发者|用户|助手|客服)"
    r"\s*(?:[:：]|\]\s*$|$)|"
    r"\[\s*(?:p[0-9]+\s*\]|客服\s*\]|(?:回复|引用|@)\s*[:：]|间隔\s*[0-9]|"
    r"(?:图片|附件|音频|视频|未识别媒体|正文未选)\s*\])",
    re.IGNORECASE,
)
_PROTOCOL = (
    *PROTOCOL_MARKERS,
    "<|",
    "[INST",
    "[/INST",
    "<<SYS",
    "<system",
    "</system",
    "<assistant",
    "</assistant",
    "<developer",
    "<user",
    "<tool",
    "</tool",
    "<function",
    "</function",
    "<analysis",
    "<reasoning",
)


def check_visible_text(text, controls=(), *, application=False):
    normalized = unicodedata.normalize("NFKC", text)
    folded = normalized.casefold()
    require(
        not any(token.casefold() in folded for token in (*_PROTOCOL, *controls))
        and (application or not _ROLE_HEADER.search(normalized))
        and not any(
            (
                unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp", "Cs")
                and c not in "\n\t\u200d\u200c"
            )
            for c in text
        ),
        "reserved control, fake role header or unsafe formatting in message",
    )


@dataclass(frozen=True, slots=True)
class Piece:
    text: str
    kind: str
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class Turn:
    role: str
    pieces: tuple[Piece, ...]

    def message(self):
        return {
            "role": self.role,
            "content": "".join(piece.text for piece in self.pieces),
        }


def render_native(
    tokenizer, policy, messages, *, for_training=False, tokenize=False, **kwargs
):
    """Always use the caller's native chat template and shared policy flags."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        **chat_template_kwargs(policy, for_training=for_training),
        **kwargs,
    )


def target_turn(events):
    pieces = []
    for event in events:
        check_visible_text(event.content)
        if pieces:
            pieces.append(Piece("\n", "target_separator"))
        pieces.append(Piece(event.content, "target_body", event.source_id))
    return Turn("assistant", tuple(pieces))


def render_observation(observation, target_person, system_prompt):
    """Project only observations; selected/audit/target fields never enter this API."""
    conv = observation.conversation
    require(
        sum(person == target_person for person in conv.authors.values()) == 1,
        "target_person must identify one account in the conversation",
    )
    events = observation.events
    visible = {event.source_id: event for event in events}
    group = conv.mode == "group"

    def alias(speaker):
        return (
            "客服" if conv.authors[speaker] == target_person else conv.aliases[speaker]
        )

    turns = []
    if system_prompt is not None:
        check_visible_text(system_prompt, application=True)
        turns.append(("system", [Piece(system_prompt, "system")]))
    previous = None
    user_observed = False
    for event in events:
        sid = event.source_id
        role = (
            "assistant"
            if event.person == target_person
            and event.selection == "text"
            and event.kind == "text"
            else "user"
        )
        if turns and turns[-1][0] == role:
            pieces = turns[-1][1]
            pieces.append(Piece("\n", "history_separator"))
        else:
            pieces = []
            turns.append((role, pieces))
        if role == "user":
            user_observed = True
            if group:
                if previous is not None:
                    minutes = int((event.time - previous.time).total_seconds()) // 60
                    if minutes >= 5:
                        pieces.append(Piece(f"[间隔{minutes}m]\n", "gap"))
                pieces.append(Piece(f"[{alias(event.speaker)}]\n", "alias", sid))
        if event.kind != "text" or event.selection == "structure":
            pieces.append(Piece(MEDIA[event.kind], "media", sid))
            if event.selection == "text":
                pieces.append(Piece("\n", "history_separator"))
        if event.selection == "text":
            check_visible_text(event.content)
            pieces.append(Piece(event.content, "history_body", sid))
            # Relation metadata never affects the same target generation. It is
            # rendered here only for observed owners, with no copied quote text.
            for edge in conv.relations.get(sid, ()) if group else ():
                if edge.resolution != "resolved":
                    continue
                if edge.kind == "mention":
                    if conv.first_observed.get(
                        edge.speaker, event.position + 1
                    ) <= event.position and any(
                        e.speaker == edge.speaker and e.position <= event.position
                        for e in events
                    ):
                        pieces.append(
                            Piece(f"\n[@:{alias(edge.speaker)}]", "mention", sid)
                        )
                else:
                    anchor = visible.get(edge.anchor_id)
                    if (
                        anchor is not None
                        and anchor.readable
                        and anchor.selection == "text"
                    ):
                        label = "回复" if edge.kind == "reply" else "引用"
                        pieces.append(
                            Piece(
                                f"\n[{label}:{alias(anchor.speaker)}]", edge.kind, sid
                            )
                        )
        previous = event
    require(user_observed, "hold: opening or continuation has no observed user input")
    require(
        turns[-1][0] == "user",
        "hold: pure assistant continuation has no new observation",
    )
    return tuple(Turn(role, tuple(pieces)) for role, pieces in turns)
