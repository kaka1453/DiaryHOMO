"""Audit the encoder's output; never assemble prompts or infer spans from text."""

from lm_trainer.train.encoding import IGNORE, EncodingError


def audit_encoding(result, plan, *, policy=None):
    ids, labels = result.input_ids, result.labels
    if len(ids) != len(labels) or len(ids) != len(result.roles) or len(ids) < 2:
        raise EncodingError("unaligned or empty encoded sequence")
    if plan["objective"] != "legacy" and labels[0] != IGNORE:
        raise EncodingError("first token is not a predictable label")
    effective = sum(v != IGNORE for v in labels[1:])
    if not effective or effective != result.body_targets + result.eos_targets:
        raise EncodingError(
            "declared targets disagree with shifted labels or are empty"
        )
    if any(v != IGNORE and v != token for token, v in zip(ids, labels)):
        raise EncodingError("label differs from its input token")
    if policy is None:
        return effective

    # Structured consumers compare EVERY position against Task A's explicit
    # spans. There is no search for body text, speaker, or matching token IDs.
    cursor = 0
    body_ids = []
    bodies = stops = native_stops = 0
    control = set(policy["control_token_ids"])
    for span in result.spans:
        a, b = span.start, span.end
        if a != cursor or not a < b <= len(ids):
            raise EncodingError(
                "renderer spans do not partition the native token stream"
            )
        selected = False
        if span.kind == "target_body":
            if not a or not span.source_id or control.intersection(ids[a:b]):
                raise EncodingError("invalid renderer target body span")
            body_ids.append(span.source_id)
            bodies += b - a
            selected = True
        if span.kind == "target_stop":
            native_stops += 1
            if b != a + 1 or ids[a] not in policy["stop_token_ids"]:
                raise EncodingError(
                    "renderer stop does not match the native stop policy"
                )
            selected = plan["supervise_target_eos"]
            stops += int(selected)
        if any(
            v != (token if selected else IGNORE)
            for v, token in zip(labels[a:b], ids[a:b])
        ):
            raise EncodingError("labels disagree with exact renderer supervision spans")
        cursor = b
    if (
        cursor != len(ids)
        or not bodies
        or bodies != result.body_targets
        or stops != result.eos_targets
        or native_stops != 1
        or tuple(body_ids) != result.focus_ids
        or len(set(body_ids)) != len(body_ids)
        or ids[: len(result.prefix_ids)] != list(result.prefix_ids)
        or next(s.start for s in result.spans if s.kind == "target_body")
        != len(result.prefix_ids)
    ):
        raise EncodingError(
            "renderer target, stop, or generation-prefix evidence is inconsistent"
        )
    return effective
