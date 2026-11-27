"""Offline operator CLI. No request-facing registration or database restore command."""

import argparse
import collections
import json
import sqlite3
import sys

from .schema import BUNDLE_SCHEMA, EventRecord, MemoryErrorV1, TrustedScope, require
from .store import MemoryStore


def load_bundle(payload):
    require(
        isinstance(payload, dict)
        and payload.keys() == {"schema", "scopes", "events", "revoked_event_ids"}
        and payload["schema"] == BUNDLE_SCHEMA
        and all(
            isinstance(payload[k], list)
            for k in ("scopes", "events", "revoked_event_ids")
        )
    )
    try:
        return (
            [TrustedScope(**r) for r in payload["scopes"]],
            [EventRecord.from_dict(r) for r in payload["events"]],
            payload["revoked_event_ids"],
        )
    except TypeError:
        raise MemoryErrorV1("MEMORY_SCHEMA_INVALID") from None


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m lm_trainer.memory")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="owner-only directory outside every Git workspace",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "update"):
        commands.add_parser(name).add_argument(
            "--bundle", required=True, help="trusted import JSON, or - for stdin"
        )
    commands.add_parser("rebuild")
    commands.add_parser("audit").add_argument(
        "--scope", help="report only the scope's eligible count"
    )
    for name in ("revoke", "delete"):
        keys = commands.add_parser(name).add_mutually_exclusive_group(required=True)
        keys.add_argument("--event")
        keys.add_argument("--source")
    args = parser.parse_args(argv)
    try:
        if args.command in ("build", "update"):
            if args.bundle == "-":
                payload = json.load(sys.stdin)
            else:
                with open(args.bundle, encoding="utf-8") as stream:
                    payload = json.load(stream)
            scopes, events, revoked = load_bundle(payload)
            store = MemoryStore(args.data_dir, create=args.command == "build")
            result = {
                "outcomes": dict(
                    collections.Counter(
                        store.update(
                            events, scopes=scopes, revoked_event_ids=revoked
                        ).values()
                    )
                )
            }
        else:
            store = MemoryStore(args.data_dir)
            if args.command == "rebuild":
                result = {"indexed_events": store.rebuild()}
            elif args.command == "audit":
                result = store.audit(scope_id=args.scope)
            else:
                result = {
                    "denied_events": store.deny(
                        event_id=args.event,
                        source_id=args.source,
                        delete=args.command == "delete",
                    )
                }
        print(json.dumps(result, sort_keys=True))
    except MemoryErrorV1 as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError, sqlite3.Error):
        print("MEMORY_OPERATION_FAILED", file=sys.stderr)
        return 2
    return int(bool(result.get("outcomes", {}).get("duplicate_version")))


if __name__ == "__main__":
    raise SystemExit(main())
