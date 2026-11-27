"""Headless evaluation runner (Task D v1).

Correctness notes that shape the implementation:

* **Single active model.** Models run one at a time: the runtime for a model is
  built, every (case, condition, turn) unit for that model runs, and the runtime
  is unloaded before the next model is built. Two large models are never
  resident at the same time.
* **Only the Task C API.** The runner calls ``create_session``/``send``. It never
  imports a model backend, renders a chat template, or checks card permissions.
* **Review-only fields never reach the prompt.** ``EvalCase.prompt_messages`` is
  built exclusively from ``messages``. ``rubric`` and ``reference_only`` are
  carried for offline human review only and are structurally unable to enter a
  prompt (see ``test_d04``).
* ``fixed_history``: Task C only commits model-generated assistant turns, so a
  literal stored transcript cannot be injected as session history. The runner
  therefore replays the fixed transcript inside the current user turn and
  records ``prompt_sha256``; every compared model receives byte-identical text.
* ``multi_turn_rollout``: the predefined user follow-up sequence is sent turn by
  turn in **one session per case**, so branch differences come from each model's
  own earlier answers and stay visible in the records.
* **Resume.** ``fixed_history`` units are independent, so each turn resumes on
  its own. A ``multi_turn_rollout`` case is resumed at case granularity: a
  partially recorded case is restarted from turn 0 in a fresh session, and the
  newer ledger entries supersede the partial ones (last entry per key wins).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lm_trainer.chat import ChatError, ChatResult, GenerationSettings
from lm_trainer.chat.memory import ContextCard
from lm_trainer.eval.cards import M0_NO_CARD, CardCondition
from lm_trainer.eval.provider import MOCK, RuntimeProvider
from lm_trainer.eval.schema import (
    EvalError,
    canonical_json,
    sha256_text,
    validate_case,
)

RESULT_SCHEMA = "lm_trainer.eval-result.v1"
RUN_SCHEMA = "lm_trainer.eval-run.v1"
CONDITIONS = ("M0", "M1", "M2", "M3", "M4")

TRANSCRIPT_OPEN = "【历史对话记录】"
TRANSCRIPT_CLOSE = "【历史对话记录结束】"
QUESTION_OPEN = "【当前问题】"


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    seed: int | None = 1234

    def settings(self) -> GenerationSettings:
        return GenerationSettings(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
        )

    def validate(self) -> GenerationConfig:
        try:
            self.settings()
        except ChatError as exc:
            raise EvalError("GENERATION_CONFIG_INVALID", exc.code) from None
        return self


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_id: str
    target_person: str
    event_group: str
    split: str
    kind: str
    messages: tuple[dict, ...]
    rubric: tuple[str, ...] = ()
    reference_only: tuple[str, ...] = ()
    memory_condition: str = M0_NO_CARD
    scenario: str | None = None

    @classmethod
    def from_dict(cls, value: dict) -> EvalCase:
        case = validate_case(value)
        return cls(
            case["case_id"],
            case["target_person"],
            case["event_group"],
            case["split"],
            case["kind"],
            tuple(dict(m) for m in case["messages"]),
            tuple(case["rubric"]),
            tuple(case["reference_only"]),
            case["memory_condition"],
            case.get("scenario"),
        )

    def to_dict(self) -> dict:
        return {
            "schema": "lm_trainer.eval.v1",
            "case_id": self.case_id,
            "target_person": self.target_person,
            "event_group": self.event_group,
            "split": self.split,
            "fixture_only": True,
            "kind": self.kind,
            "messages": [dict(m) for m in self.messages],
            "rubric": list(self.rubric),
            "reference_only": list(self.reference_only),
            "memory_condition": self.memory_condition,
            **({"scenario": self.scenario} if self.scenario is not None else {}),
        }

    def prompt_messages(self) -> tuple[dict, ...]:
        """The only source of model input. ``rubric``/``reference_only`` excluded."""

        return tuple({"role": m["role"], "content": m["content"]} for m in self.messages)

    @property
    def event_key(self) -> str:
        return self.event_group


@dataclass(frozen=True, slots=True)
class ModelPlan:
    model_id: str
    available: bool
    reason: str | None = None
    fixture_only: bool | None = None
    supports_streaming: bool | None = None


@dataclass(frozen=True, slots=True)
class EvalCaseRecord:
    case_id: str
    model_id: str
    kind: str
    memory_condition: str
    event_group: str
    split: str
    target_person: str
    run_seed: int | None
    output: str = field(repr=False)
    finish_reason: str
    committed: bool
    error_code: str | None
    input_tokens: int | None
    output_tokens: int | None
    ttft_ms: float | None
    tokens_per_second: float | None
    latency_ms: float | None
    peak_vram_mb: float | None
    card_audit: tuple[dict, ...]
    dropped_turns: int | None
    turn_index: int = 0
    total_turns: int = 1
    prompt_sha256: str = ""
    prompt: str = field(default="", repr=False)
    history_mode: str = "sequential"
    metrics_source: str = MOCK
    scenario: str | None = None


@dataclass(slots=True)
class RunConfig:
    case_set_sha256: str
    case_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    generation: GenerationConfig
    role_id: str
    memory_conditions: tuple[str, ...]
    split: str
    mode: str
    allow_test: bool = False
    card_fixture_sha256: str | None = None

    def identity(self) -> str:
        payload = {
            "schema": RUN_SCHEMA,
            "case_set_sha256": self.case_set_sha256,
            "case_ids": list(self.case_ids),
            "model_ids": list(self.model_ids),
            "generation": asdict(self.generation),
            "role_id": self.role_id,
            "memory_conditions": list(self.memory_conditions),
            "split": self.split,
            "mode": self.mode,
            "card_fixture_sha256": self.card_fixture_sha256,
        }
        return sha256_text(canonical_json(payload))


@dataclass(slots=True)
class EvalResult:
    run_id: str
    config: RunConfig
    records: list[EvalCaseRecord]
    skipped: list[dict]
    status: str = "PASS"
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "schema": RESULT_SCHEMA,
            "fixture_only": True,
            "mode": self.config.mode,
            "run_id": self.run_id,
            "config_identity": self.config.identity(),
            "config": asdict(self.config),
            "status": self.status,
            "error": self.error,
            "skipped": self.skipped,
            "record_count": len(self.records),
            "records": [asdict(record) for record in self.records],
        }


@dataclass(slots=True)
class RunRecord:
    run_id: str
    config_identity: str
    case_id: str
    model_id: str
    memory_condition: str
    turn_index: int
    record: EvalCaseRecord

    def key(self) -> tuple:
        return (self.case_id, self.model_id, self.memory_condition, self.turn_index)


def rollout_turns(case: EvalCase) -> list[tuple[int, str]]:
    """User turns of a rollout: the predefined follow-up sequence."""

    return [
        (index // 2, message["content"])
        for index, message in enumerate(case.messages)
        if index % 2 == 0
    ]


def fixed_prompt(case: EvalCase) -> str:
    """Deterministic prompt for fixed history, identical for every model."""

    history = case.messages[:-1]
    final = case.messages[-1]["content"]
    if not history:
        return final
    lines = [TRANSCRIPT_OPEN]
    for message in history:
        speaker = "用户" if message["role"] == "user" else "客服"
        lines.append(f"{speaker}：{message['content']}")
    lines.append(TRANSCRIPT_CLOSE)
    lines.append(QUESTION_OPEN)
    lines.append(final)
    return "\n".join(lines)


def _prompt_sha(case: EvalCase, turn: int, text: str) -> str:
    return sha256_text(
        canonical_json({"case_id": case.case_id, "kind": case.kind, "turn": turn, "text": text})
    )


def _record(
    *,
    case: EvalCase,
    model_id: str,
    result: ChatResult | None,
    prompt_sha256: str,
    condition: str,
    mode: str,
    turn_index: int,
    total_turns: int,
    run_seed: int | None,
    history_mode: str,
    error_code: str | None = None,
    prompt: str = "",
) -> EvalCaseRecord:
    return EvalCaseRecord(
        case_id=case.case_id,
        model_id=model_id,
        kind=case.kind,
        memory_condition=condition,
        event_group=case.event_group,
        split=case.split,
        target_person=case.target_person,
        run_seed=run_seed,
        output=result.text if result is not None else "",
        finish_reason=result.finish_reason if result is not None else "error",
        committed=result.committed if result is not None else False,
        error_code=error_code or (result.error_code if result is not None else None),
        input_tokens=result.input_tokens if result is not None else None,
        output_tokens=result.output_tokens if result is not None else None,
        ttft_ms=result.ttft_ms if result is not None else None,
        tokens_per_second=result.tokens_per_second if result is not None else None,
        latency_ms=result.latency_ms if result is not None else None,
        peak_vram_mb=result.peak_vram_mb if result is not None else None,
        card_audit=tuple(asdict(a) for a in result.card_audit) if result else (),
        dropped_turns=result.dropped_turns if result is not None else None,
        turn_index=turn_index,
        total_turns=total_turns,
        prompt_sha256=prompt_sha256,
        prompt=prompt,
        history_mode=history_mode,
        metrics_source=mode,
        scenario=case.scenario,
    )


def _append(path: Path, entry: RunRecord) -> None:
    payload = {
        "run_id": entry.run_id,
        "config_identity": entry.config_identity,
        "key": list(entry.key()),
        "case_id": entry.case_id,
        "model_id": entry.model_id,
        "memory_condition": entry.memory_condition,
        "turn_index": entry.turn_index,
        "record": asdict(entry.record),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_ledger(path: Path, identity: str) -> dict[tuple, RunRecord]:
    """Load a resume ledger. A different config identity is a hard error.

    Duplicate keys may legitimately exist after a rollout case restart; the last
    entry wins and the caller reports the superseded count.
    """

    done: dict[tuple, RunRecord] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise EvalError("RESUME_MISMATCH", "unreadable ledger line") from exc
        if entry.get("config_identity") != identity:
            raise EvalError(
                "RESUME_MISMATCH",
                "ledger belongs to a different case/model/generation config",
            )
        done[tuple(entry["key"])] = RunRecord(
            entry["run_id"],
            entry["config_identity"],
            entry["case_id"],
            entry["model_id"],
            entry["memory_condition"],
            entry["turn_index"],
            EvalCaseRecord(**entry["record"]),
        )
    return done


def _turn_plan(case: EvalCase) -> list[tuple[int, str]]:
    if case.kind == "fixed_history":
        return [(0, fixed_prompt(case))]
    return rollout_turns(case)


def run_evaluation(
    *,
    provider: RuntimeProvider,
    cases: list[EvalCase],
    models: list[ModelPlan],
    generation: GenerationConfig,
    role_id: str,
    memory_conditions: tuple[str, ...] = (M0_NO_CARD,),
    cards: dict[str, CardCondition] | None = None,
    output_dir: Path | str,
    run_id: str | None = None,
    allow_test: bool = False,
    resume: bool = True,
    case_set_sha256: str | None = None,
) -> EvalResult:
    """Run every (model, case, condition, turn) unit with a resumable ledger."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cases:
        raise EvalError("SCHEMA_INVALID", "no evaluation cases")
    for case in cases:
        validate_case(case.to_dict())
        if case.split == "test" and not allow_test:
            raise EvalError("TEST_SPLIT_LOCKED", f"{case.case_id}: test requires --allow-test")
    generation.validate()
    conditions = tuple(memory_conditions) or (M0_NO_CARD,)
    for condition in conditions:
        if condition not in CONDITIONS:
            raise EvalError("SCHEMA_INVALID", f"unknown memory condition {condition!r}")
        if condition != M0_NO_CARD and (cards is None or condition not in cards):
            raise EvalError("SCHEMA_INVALID", f"missing card fixture for {condition}")
    if not models:
        raise EvalError("SCHEMA_INVALID", "no models planned")
    case_ids = [c.case_id for c in cases]
    if len(set(case_ids)) != len(case_ids):
        raise EvalError("SCHEMA_INVALID", "duplicate case_id")
    if case_set_sha256 is None:
        case_set_sha256 = sha256_text(canonical_json([c.to_dict() for c in cases]))
    config = RunConfig(
        case_set_sha256=case_set_sha256,
        case_ids=tuple(case_ids),
        model_ids=tuple(m.model_id for m in models),
        generation=generation,
        role_id=role_id,
        memory_conditions=conditions,
        split=cases[0].split,
        mode=provider.mode,
        allow_test=allow_test,
        card_fixture_sha256=(
            sha256_text(
                canonical_json(
                    [
                        {"condition": name, "cards": [asdict(c) for c in cond.candidates]}
                        for name, cond in cards.items()
                    ]
                )
            )
            if cards
            else None
        ),
    )
    identity = config.identity()
    run_id = run_id or os.urandom(8).hex()
    ledger_path = output_dir / "ledger.jsonl"
    done = load_ledger(ledger_path, identity) if resume else {}
    records: list[EvalCaseRecord] = []
    skipped: list[dict] = []
    for model in models:
        if not model.available:
            skipped.append(
                {
                    "model_id": model.model_id,
                    "status": "SKIP",
                    "reason": model.reason or "MODEL_UNAVAILABLE",
                }
            )
            continue
        runtime = provider.build(model.model_id)
        try:
            for case in cases:
                if case.split == "test" and not allow_test:
                    raise EvalError("TEST_SPLIT_LOCKED", case.case_id)
                for condition in conditions:
                    selected = cards[condition].candidates if cards else ()
                    plan = _turn_plan(case)
                    total = len(plan)
                    keys = [(case.case_id, model.model_id, condition, t) for t, _ in plan]
                    if case.kind == "fixed_history":
                        for (turn, text), key in zip(plan, keys):
                            if key in done:
                                records.append(done[key].record)
                                continue
                            record = _send_unit(
                                runtime, case, model, generation, role_id, text,
                                turn, total, condition, selected, provider.mode,
                            )
                            entry = RunRecord(
                                run_id, identity, case.case_id, model.model_id,
                                condition, turn, record,
                            )
                            _append(ledger_path, entry)
                            done[key] = entry
                            records.append(record)
                        continue
                    # multi-turn rollout: all-or-nothing per case/condition
                    if all(key in done for key in keys):
                        records.extend(done[key].record for key in keys)
                        continue
                    session = runtime.create_session(
                        model.model_id, role_id, generation.settings()
                    )
                    for (turn, text), key in zip(plan, keys):
                        record = _send_unit(
                            runtime, case, model, generation, role_id, text,
                            turn, total, condition, selected, provider.mode,
                            session_id=session.session_id,
                        )
                        entry = RunRecord(
                            run_id, identity, case.case_id, model.model_id,
                            condition, turn, record,
                        )
                        _append(ledger_path, entry)
                        done[key] = entry
                        records.append(record)
        finally:
            unload = getattr(runtime, "unload", None)
            if callable(unload):
                unload()
    result = EvalResult(run_id, config, records, skipped)
    _write_json(output_dir / "run.json", result.to_dict())
    return result


def _send_unit(
    runtime,
    case: EvalCase,
    model: ModelPlan,
    generation: GenerationConfig,
    role_id: str,
    text: str,
    turn: int,
    total: int,
    condition: str,
    cards: tuple[ContextCard, ...],
    mode: str,
    session_id: str | None = None,
) -> EvalCaseRecord:
    prompt_sha256 = _prompt_sha(case, turn, text)
    history_mode = "fixed_transcript" if case.kind == "fixed_history" else "sequential"
    run_seed = generation.seed if generation.do_sample else None
    # Store the exact text sent to Task C for offline review. It is built from
    # ``messages`` only; rubric/reference_only never enter it.
    shown_prompt = text

    def failed(code: str | None) -> EvalCaseRecord:
        return _record(
            case=case, model_id=model.model_id, result=None,
            prompt_sha256=prompt_sha256, condition=condition, mode=mode,
            turn_index=turn, total_turns=total, run_seed=run_seed,
            history_mode=history_mode, error_code=code, prompt=shown_prompt,
        )

    if session_id is None:
        try:
            session = runtime.create_session(
                model.model_id, role_id, generation.settings()
            )
        except ChatError as exc:
            return failed(exc.code)
        session_id = session.session_id
    try:
        result = runtime.send(session_id, text, list(cards) or None)
    except ChatError as exc:
        return failed(exc.code)
    return _record(
        case=case, model_id=model.model_id, result=result,
        prompt_sha256=prompt_sha256, condition=condition, mode=mode,
        turn_index=turn, total_turns=total, run_seed=run_seed,
        history_mode=history_mode,
        prompt=shown_prompt,
    )
