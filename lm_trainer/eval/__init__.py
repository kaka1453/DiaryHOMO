"""Task D v1: headless evaluation, anonymous blind review and card experiments.

Every public entry point consumes the Task C ``ChatRuntime``/``MockChatRuntime``
API only. Importing this package never imports torch, transformers, peft or
gradio, and never loads model weights.
"""

from lm_trainer.eval.blind import (
    BlindPack,
    RevealSummary,
    load_blind_pack,
    load_reveal_summary,
    make_blind_pack,
    submit_review,
)
from lm_trainer.eval.cards import (
    M0_NO_CARD,
    M1_RELEVANT,
    M2_IRRELEVANT,
    M3_CONFLICT,
    M4_AMBIGUOUS,
    MEMORY_CONDITIONS,
    card_condition,
    load_card_fixture,
)
from lm_trainer.eval.report import (
    TaskBAggregate,
    load_benchmark_aggregate,
    render_markdown,
    summarize,
)
from lm_trainer.eval.runner import (
    EvalCase,
    EvalCaseRecord,
    EvalResult,
    GenerationConfig,
    ModelPlan,
    RunConfig,
    RunRecord,
    run_evaluation,
)
from lm_trainer.eval.schema import (
    CONTRACT_SCHEMA,
    EvalError,
    build_manifest,
    canonical_json,
    validate_manifest,
    validate_manifest_for_run,
)

__all__ = [
    "CONTRACT_SCHEMA",
    "M0_NO_CARD",
    "M1_RELEVANT",
    "M2_IRRELEVANT",
    "M3_CONFLICT",
    "M4_AMBIGUOUS",
    "MEMORY_CONDITIONS",
    "BlindPack",
    "EvalCase",
    "EvalCaseRecord",
    "EvalError",
    "EvalResult",
    "GenerationConfig",
    "ModelPlan",
    "RevealSummary",
    "RunConfig",
    "RunRecord",
    "TaskBAggregate",
    "build_manifest",
    "canonical_json",
    "card_condition",
    "load_benchmark_aggregate",
    "load_blind_pack",
    "load_card_fixture",
    "load_reveal_summary",
    "make_blind_pack",
    "render_markdown",
    "run_evaluation",
    "submit_review",
    "summarize",
    "validate_manifest",
    "validate_manifest_for_run",
]
