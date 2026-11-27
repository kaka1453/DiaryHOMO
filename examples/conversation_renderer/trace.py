"""Generate or check an auditable, fictional native-tokenizer conversation trace."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from lm_trainer.chat.context import ContextBuilder
from lm_trainer.conversation import (
    inspect_candidate,
    observe,
    parse_catalog,
    validate_route,
)
from lm_trainer.template_policy import resolve_policy
from lm_trainer.train.encoding import encode_conversation, verify_training_prefix

from .fixture import scenario_data

DEFAULT_TOKENIZER = (
    Path(__file__).resolve().parents[1] / "vision_support/task_b_fixture_adapters/sft"
)
STOP = "<|im_end|>"


def make_trace(tokenizer, *, local_native=False):
    raw, rows, routes = scenario_data()
    catalog = parse_catalog(raw)
    windows = [inspect_candidate(row, catalog) for row in rows]
    defaults = set(validate_route(windows, routes))
    template, policy = resolve_policy(
        tokenizer,
        {"stop_tokens": [STOP], **({"enable_thinking": False} if local_native else {})},
    )
    samples, totals = [], {}
    for window in windows:
        if window.id not in defaults:
            continue
        summary = totals[window.id] = {
            k: 0
            for k in (
                "opportunities",
                "unique_targets",
                "input_tokens",
                "prefix_tokens",
                "body_tokens",
                "eos_tokens",
                "target_separator_tokens",
                "context_annotation_tokens",
            )
        }
        for result in encode_conversation(
            tokenizer,
            window,
            template,
            policy,
            {"max_length": 1024, "supervise_target_eos": True},
        ):
            # Build an independent inference request from observed source IDs, not label flags.
            focus = result.focus_ids[0]
            prefix_events = window.timeline[: window.boundaries[focus]]
            observation = observe(catalog, window.conversation.id, prefix_events)
            direct = ContextBuilder.build_conversation(
                tokenizer,
                policy,
                observation,
                target_person=window.target_person,
                context_limit=1024,
                max_new_tokens=128,
            )
            verify_training_prefix(
                result, direct.input_ids, identity=direct.conversation_renderer
            )
            samples.append(
                {
                    "window_id": window.id,
                    "format": window.conversation.mode,
                    "focus_ids": result.focus_ids,
                    "input_ids": result.input_ids,
                    "labels": result.labels,
                    "spans": [asdict(s) for s in result.spans],
                    "direct_prefix_ids": direct.input_ids,
                    "prefix_token_equal": True,
                    "renderer_version": result.renderer_version,
                    "template_hash": result.template_hash,
                    "stop_hash": result.stop_hash,
                }
            )
            summary["opportunities"] += 1
            summary["unique_targets"] += len(result.focus_ids)
            summary["input_tokens"] += len(result.input_ids)
            summary["prefix_tokens"] += len(direct.input_ids)
            summary["body_tokens"] += result.body_targets
            summary["eos_tokens"] += result.eos_targets
            summary["target_separator_tokens"] += result.roles.count("target_separator")
            summary["context_annotation_tokens"] += sum(
                result.roles.count(k)
                for k in (
                    "alias",
                    "gap",
                    "media",
                    "reply",
                    "quote",
                    "mention",
                )
            )
    return {
        "schema": "lm_trainer.conversation-token-trace.v1",
        "fixture_only": True,
        "training_authorized": False,
        "tokenizer": "Qwen3.8 native/local"
        if local_native
        else "bundled synthetic HF BPE",
        "stop_tokens": list(
            zip(policy["stop_tokens"], policy["stop_token_ids"], strict=True)
        ),
        "aggregate_by_window": totals,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--local-native",
        action="store_true",
        help="Use the native Qwen non-thinking flags",
    )
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--output", type=Path, help="Create a new output file")
    targets.add_argument("--check", type=Path, help="Compare against a stored trace")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    if not args.tokenizer.is_dir():
        parser.error("local tokenizer directory required")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    data = make_trace(tok, local_native=args.local_native)
    # Compact the individual native streams while retaining a regular JSON artifact.
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
    if args.check:
        if json.loads(args.check.read_text(encoding="utf-8")) != json.loads(encoded):
            parser.error("frozen conversation trace is not reproducible")
        print("conversation trace matches")
    elif args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        print(args.output)
    else:
        print(json.dumps(data["aggregate_by_window"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
