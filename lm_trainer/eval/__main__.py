"""``python -m lm_trainer.eval`` — headless evaluation CLI.

Subcommands
-----------
``validate``   validate a cases file (schema, hash, split authorization)
``run``        run fixed-history / rollout evaluation for one or more models
``cards``      run the M0–M4 context-card experiment
``blind``      build an anonymous blind-review pack from a finished run
``review``     submit a private anonymous review with dimension reasons
``reveal``     join reviews with the private map and print a reveal summary
``report``     export JSON/JSONL/CSV/Markdown/HTML reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lm_trainer.chat import ChatError, ModelInfo, RoleConfig, load_roles
from lm_trainer.eval.blind import (
    load_blind_pack,
    load_reviews,
    make_blind_pack,
    reveal,
    save_blind_pack,
    save_reveal_summary,
    submit_review,
)
from lm_trainer.eval.cards import build_conditions, load_card_fixture
from lm_trainer.eval.provider import (
    MockRuntimeProvider,
    RegistryRuntimeProvider,
    manual_cards_provider,
)
from lm_trainer.eval.report import write_reports
from lm_trainer.eval.runner import (
    EvalCase,
    EvalResult,
    GenerationConfig,
    ModelPlan,
    run_evaluation,
)
from lm_trainer.eval.schema import (
    EvalError,
    canonical_json,
    sha256_text,
    validate_cases_file,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a cases file")
    validate.add_argument("--cases", required=True, type=Path)
    validate.add_argument("--allow-test", action="store_true")

    run = sub.add_parser("run", help="run a headless evaluation")
    run.add_argument("--cases", required=True, type=Path)
    run.add_argument("--models", type=Path, help="chat-models.v1 JSON (real inference)")
    run.add_argument("--roles", type=Path, help="chat-roles.v1 JSON")
    run.add_argument("--role", default="CS_AGENT")
    run.add_argument("--model", action="append", help="model_id, repeatable")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--base", type=Path)
    run.add_argument("--device", help="cpu or cuda:N")
    run.add_argument("--allow-fixture", action="store_true")
    run.add_argument("--allow-test", action="store_true")
    run.add_argument("--mock", action="store_true", help="use MockChatRuntime instead of real inference")
    run.add_argument("--mock-responder-file", type=Path, help="JSON with mock responder text")
    run.add_argument("--max-new-tokens", type=int, default=128)
    run.add_argument("--do-sample", action="store_true")
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument("--top-p", type=float, default=0.9)
    run.add_argument("--top-k", type=int, default=50)
    run.add_argument("--repetition-penalty", type=float, default=1.0)
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--memory-conditions", default="M0")
    run.add_argument("--cards", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--fresh", action="store_true")
    run.add_argument("--run-id")

    cards_p = sub.add_parser("cards", help="run the M0–M4 card experiment")
    cards_p.add_argument("--cases", required=True, type=Path)
    cards_p.add_argument("--cards", required=True, type=Path)
    cards_p.add_argument("--models", type=Path)
    cards_p.add_argument("--roles", type=Path)
    cards_p.add_argument("--role", default="CS_AGENT")
    cards_p.add_argument("--model", action="append")
    cards_p.add_argument("--output", required=True, type=Path)
    cards_p.add_argument("--base", type=Path)
    cards_p.add_argument("--device")
    cards_p.add_argument("--allow-fixture", action="store_true")
    cards_p.add_argument("--allow-test", action="store_true")
    cards_p.add_argument("--mock", action="store_true")
    cards_p.add_argument("--max-new-tokens", type=int, default=128)
    cards_p.add_argument("--do-sample", action="store_true")
    cards_p.add_argument("--temperature", type=float, default=0.7)
    cards_p.add_argument("--top-p", type=float, default=0.9)
    cards_p.add_argument("--top-k", type=int, default=50)
    cards_p.add_argument("--repetition-penalty", type=float, default=1.0)
    cards_p.add_argument("--seed", type=int, default=1234)

    blind = sub.add_parser("blind", help="build an anonymous blind pack")
    blind.add_argument("--run", required=True, type=Path, help="run.json")
    blind.add_argument("--output", required=True, type=Path)
    blind.add_argument("--seed", type=int, default=2026)
    blind.add_argument("--pair", nargs=2, metavar=("LEFT_CANDIDATE", "RIGHT_CANDIDATE"),
                       help="Select exactly two models; actual left/right remains blinded.")

    review = sub.add_parser("review", help="submit scores and per-dimension reasons from JSON")
    review.add_argument("--pack", required=True, type=Path)
    review.add_argument("--reviews", required=True, type=Path)
    review.add_argument("--entry", required=True, type=Path)

    reveal_p = sub.add_parser("reveal", help="reveal a blind pack")
    reveal_p.add_argument("--pack", required=True, type=Path)
    reveal_p.add_argument("--map", required=True, type=Path)
    reveal_p.add_argument("--reviews", required=True, type=Path)
    reveal_p.add_argument("--output", required=True, type=Path)

    report = sub.add_parser("report", help="export reports")
    report.add_argument("--run", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    report.add_argument("--reveal", type=Path)
    report.add_argument("--benchmark", type=Path)

    return parser


MOCK_ROLE = {
    "role_id": "CS_AGENT",
    "namespace": "fixture-line-iris",
    "version": "1",
    "system_prompt": (
        "You are a fictional machine-vision support agent. "
        "Separate observations from hypotheses and ask for missing facts."
    ),
}


def _roles(args):
    if getattr(args, "roles", None):
        return tuple(load_roles(args.roles))
    return (RoleConfig(**MOCK_ROLE),)


def _provider(args, models_required=True):
    if args.mock:
        raw = {}
        if getattr(args, "mock_responder_file", None):
            payload = json.loads(args.mock_responder_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                raw = payload
        models = [
            ModelInfo(m, "adapter" if m != "base" else "base", True, True, None, None)
            for m in (args.model or ["base", "sft_final"])
        ]
        return MockRuntimeProvider(tuple(models), _roles(args), responder=_responder(raw))
    if not getattr(args, "models", None):
        raise EvalError(
            "SCHEMA_INVALID", "real inference requires --models and --roles"
        )
    return RegistryRuntimeProvider(
        args.models,
        args.roles,
        allow_fixture=args.allow_fixture,
        base_override=args.base,
        device_override=args.device,
        memory_provider=manual_cards_provider(),
    )


def _responder(prompts: dict):
    def respond(model_id, role, history, text):
        return prompts.get(model_id, prompts.get("*", "Mock 回复：请先核对触发计数与存图数量。"))

    return respond


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args)
        if args.command == "run":
            return _run(args)
        if args.command == "cards":
            return _cards(args)
        if args.command == "blind":
            return _blind(args)
        if args.command == "review":
            return _review(args)
        if args.command == "reveal":
            return _reveal(args)
        if args.command == "report":
            return _report(args)
    except EvalError as exc:
        print(json.dumps({"error_code": exc.code, "detail": exc.detail}), file=sys.stderr)
        return 2
    except ChatError as exc:
        print(json.dumps({"error_code": exc.code}), file=sys.stderr)
        return 2
    return 2


def _validate(args) -> int:
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases, manifest = validate_cases_file(
        payload, allow_test=args.allow_test, source=args.cases.name
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "case_count": len(cases),
                "split": manifest["split"],
                "manifest_case_count": manifest.get("case_count"),
                "case_ids": [c["case_id"] for c in cases],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _generation(args) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )


def _cases(args) -> tuple[list[EvalCase], dict]:
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    raw, manifest = validate_cases_file(
        payload, allow_test=args.allow_test, source=args.cases.name
    )
    return [EvalCase.from_dict(c) for c in raw], manifest


def _run(args) -> int:
    cases, _manifest = _cases(args)
    provider = _provider(args)
    conditions = tuple(x.strip() for x in args.memory_conditions.split(",") if x.strip())
    cards = None
    if args.cards:
        fixture, _ = load_card_fixture(args.cards)
        cards = build_conditions(fixture)
    models = [ModelPlan(m.model_id, m.available, m.reason, m.fixture_only) for m in provider.list_models()]
    if args.model:
        wanted = set(args.model)
        models = [m for m in models if m.model_id in wanted]
        missing = wanted - {m.model_id for m in models}
        for model_id in sorted(missing):
            models.append(ModelPlan(model_id, False, "UNKNOWN_MODEL", None))
    result = run_evaluation(
        provider=provider,
        cases=cases,
        models=models,
        generation=_generation(args),
        role_id=args.role,
        memory_conditions=conditions,
        cards=cards,
        output_dir=args.output,
        run_id=args.run_id,
        allow_test=args.allow_test,
        resume=args.resume,
        case_set_sha256=sha256_text(canonical_json([c.to_dict() for c in cases])),
    )
    summary = _summary(result)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _summary(result: EvalResult) -> dict:
    from lm_trainer.eval.report import summarize

    return summarize(result.to_dict())


def _cards(args) -> int:
    cases, _manifest = _cases(args)
    fixture, _ = load_card_fixture(args.cards)
    conditions = build_conditions(fixture)
    provider = _provider(args)
    models = [ModelPlan(m.model_id, m.available, m.reason, m.fixture_only) for m in provider.list_models()]
    if args.model:
        wanted = set(args.model)
        models = [m for m in models if m.model_id in wanted]
    result = run_evaluation(
        provider=provider,
        cases=cases,
        models=models,
        generation=_generation(args),
        role_id=args.role,
        memory_conditions=("M0", "M1", "M2", "M3", "M4"),
        cards=conditions,
        output_dir=args.output,
        run_id=getattr(args, "run_id", None),
        allow_test=args.allow_test,
        resume=False,
        case_set_sha256=sha256_text(canonical_json([c.to_dict() for c in cases])),
    )
    print(json.dumps(_summary(result), ensure_ascii=False, indent=2))
    return 0


def _blind(args) -> int:
    payload = json.loads(args.run.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if args.pair:
        if len(set(args.pair)) != 2:
            raise EvalError("BLIND_INVALID", "--pair requires two different candidates")
        records = [record for record in records if record["model_id"] in args.pair]
    pack, private = make_blind_pack(
        records,
        seed=args.seed,
        mode=payload.get("mode", "mock"),
    )
    pack_path, map_path = save_blind_pack(pack, private, args.output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "pack": str(pack_path),
                "map": str(map_path),
                "item_count": len(pack.items),
                "seed": args.seed,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _review(args) -> int:
    try:
        entry = json.loads(args.entry.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvalError("REVIEW_INVALID", "cannot read review JSON") from exc
    required = {"item_id", "choice"}
    if (not isinstance(entry, dict) or not required <= entry.keys()
        or entry.keys() - required - {"scores", "dimension_reasons", "reason", "reviewer"}
        or any(not isinstance(entry[k], str) for k in required | ({"reason", "reviewer"} & entry.keys()))):
        raise EvalError("REVIEW_INVALID", "invalid review entry fields")
    record = submit_review(load_blind_pack(args.pack), args.reviews, **entry)
    print(json.dumps({"status": "PASS", "item_id": record.item_id}))
    return 0


def _reveal(args) -> int:
    pack = load_blind_pack(args.pack)
    mapping = json.loads(args.map.read_text(encoding="utf-8"))
    reviews = load_reviews(pack, args.reviews)
    summary = reveal(pack, mapping, reviews)
    save_reveal_summary(summary, args.output)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _report(args) -> int:
    payload = json.loads(args.run.read_text(encoding="utf-8"))
    reveal_payload = None
    if args.reveal:
        reveal_payload = json.loads(args.reveal.read_text(encoding="utf-8"))
    paths = write_reports(payload, args.output, reveal=reveal_payload)
    print(json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
