"""Offline RTX 3090 smoke/measurement using only the pinned Task B tiny fixtures."""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from lm_trainer.chat import (
    CardPermissions,
    ChatRuntime,
    ContextCard,
    GenerationSettings,
    ManualCardProvider,
    load_registry,
    load_roles,
)
from lm_trainer.train.artifacts import read_json

FIXTURES = Path(__file__).resolve().parent


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(args.device)
    registry = load_registry(
        FIXTURES / "task_c_models.json",
        allow_fixture=True,
        base_override=args.base,
        device_override=args.device,
    )
    cards = [
        ContextCard(**{**c, "permissions": CardPermissions(**c["permissions"])})
        for c in read_json(FIXTURES / "task_c_cards.json")
    ]
    runtime = ChatRuntime(
        registry,
        load_roles(FIXTURES / "task_c_roles.json"),
        memory_provider=ManualCardProvider(),
    )
    settings = GenerationSettings(max_new_tokens=16)

    def cuda_memory():
        torch.cuda.synchronize(args.device)
        return {
            "allocated_mib": torch.cuda.memory_allocated(args.device) / 1024**2,
            "reserved_mib": torch.cuda.memory_reserved(args.device) / 1024**2,
        }

    def measured(session, question, memory=None):
        with runtime.stream(session.session_id, question, memory) as stream:
            pieces = 0
            for event in stream:
                if event.kind == "delta":
                    pieces += 1
                else:
                    result = event.result
        return {
            **{
                k: getattr(result, k)
                for k in (
                    "input_tokens",
                    "output_tokens",
                    "ttft_ms",
                    "tokens_per_second",
                    "latency_ms",
                    "peak_vram_mb",
                    "finish_reason",
                    "committed",
                    "dropped_turns",
                    "error_code",
                )
            },
            "delta_events": pieces,
            "card_audit": list(map(asdict, result.card_audit)),
        }

    report = {
        "schema": "lm_trainer.chat-benchmark.v1",
        "fixture_only": True,
        "timestamp": datetime.now(UTC).astimezone().isoformat(),
        "gpu": props.name,
        "gpu_total_mib": props.total_memory / 1024**2,
        "packages": {
            p: version(p)
            for p in ("torch", "transformers", "accelerate", "peft", "tokenizers")
        },
        "settings": asdict(settings),
        "initial_process_memory": cuda_memory(),
        "models": [],
    }
    session = runtime.create_session("base", "CS_AGENT", settings)
    question = "Fictional IRIS camera: which exposure setting should we check?"
    for model_id in ("base", "cpt_stage1", "sft_final", "base"):
        session = runtime.switch_model(session.session_id, model_id)
        entry = {
            "activation": asdict(runtime.active_model()),
            "after_switch": cuda_memory(),
            "cases": [],
        }
        report["models"].append(entry)
        for label, turns, memory in (
            ("short", 0, None),
            ("typical", 2, cards),
            ("long", 14, None),
        ):
            case = runtime.create_session(model_id, "CS_AGENT", settings)
            setup_failures = []
            for i in range(turns):
                result = runtime.send(
                    case.session_id,
                    f"Fictional IRIS report {i + 1}: "
                    "The test trigger count and the number of images disagree. "
                    "Check exposure and compare the next sample.",
                )
                if not result.committed:
                    setup_failures.append(result.error_code or result.finish_reason)
            entry["cases"].append(
                {
                    "history": label,
                    "committed_prior_turns": len(
                        runtime.get_session(case.session_id).messages
                    )
                    // 2,
                    "setup_failures": setup_failures,
                    **measured(case, question, memory),
                    "after_request": cuda_memory(),
                }
            )
    runtime.unload()
    report["after_runtime_unload"] = cuda_memory()
    # This standalone process owns all CUDA work. Diagnostic only: production
    # runtime does NOT clear global PyTorch cuBLAS workspaces used by other code.
    with torch.cuda.device(args.device):
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
    report["after_process_cublas_cleanup"] = cuda_memory()
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write("\n")
    return int(
        report["after_process_cublas_cleanup"] != report["initial_process_memory"]
        or any(
            not case["committed"] or case["setup_failures"]
            for m in report["models"]
            for case in m["cases"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
