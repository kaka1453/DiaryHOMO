"""Offline CLI for metadata and fictional fixture smoke tests."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from lm_trainer.train.artifacts import read_json

from . import (
    CardPermissions,
    ChatError,
    ChatRuntime,
    ContextCard,
    GenerationSettings,
    ManualCardProvider,
    load_registry,
    load_roles,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--roles", type=Path)
    parser.add_argument(
        "--base", type=Path, help="override all base paths for local smoke tests"
    )
    parser.add_argument("--device", help="cpu or cuda:N")
    parser.add_argument("--allow-fixture", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--model", default="base")
    parser.add_argument("--role")
    parser.add_argument(
        "--question",
        action="append",
        help="repeat for multi-turn smoke; fictional text only",
    )
    parser.add_argument(
        "--cards", type=Path, help="optional explicit JSON context-card fixture"
    )
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    try:
        registry = load_registry(
            args.models,
            allow_fixture=args.allow_fixture,
            base_override=args.base,
            device_override=args.device,
        )
        if args.list_models:
            print(
                json.dumps(
                    list(map(asdict, registry.list_models())), ensure_ascii=False
                )
            )
            return 0
        if not args.roles or not args.role or not args.question:
            raise ChatError("INVALID_CONFIG")
        cards = None
        if args.cards:
            try:
                cards = [
                    ContextCard(
                        **{**c, "permissions": CardPermissions(**c["permissions"])}
                    )
                    for c in read_json(args.cards)
                ]
            except (OSError, KeyError, ValueError, TypeError):
                raise ChatError("INVALID_CONFIG") from None
        runtime = ChatRuntime(
            registry,
            load_roles(args.roles),
            memory_provider=ManualCardProvider() if cards is not None else None,
        )
        status = 0
        try:
            s = runtime.create_session(
                args.model,
                args.role,
                GenerationSettings(
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    do_sample=args.seed is not None,
                ),
            )
            for q in args.question:
                if args.stream:
                    with runtime.stream(s.session_id, q, cards) as events:
                        for event in events:
                            if event.result is not None:
                                result = event.result
                else:
                    result = runtime.send(s.session_id, q, cards)
                report = asdict(result)
                del report["text"]
                print(json.dumps(report, ensure_ascii=False), flush=True)
                if result.finish_reason in ("error", "cancel"):
                    status = 1
                    break
        finally:
            runtime.unload()
        return status
    except ChatError as exc:
        print(json.dumps({"error_code": exc.code}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
