"""Run the offline Task B matrix in fresh interpreters; emit a sanitized JSON report."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from examples.vision_support.task_b_workload import (
    POLICY,
    stage_configs,
    tiny_base,
    workload,
)
from lm_trainer.train.artifacts import verify_artifact
from lm_trainer.train.performance import summarize_updates, write_json

REPO = Path(__file__).resolve().parents[2]


def parameter_elements(base):
    """Inspect safetensors headers only, before allowing a benchmark size tier."""
    from safetensors import safe_open

    single = base / "model.safetensors"
    if single.is_file():
        paths = [single]
    else:
        index = json.loads((base / "model.safetensors.index.json").read_text())
        paths = [base / path for path in sorted(set(index["weight_map"].values()))]
    count = 0
    for path in paths:
        if path.parent != base:
            raise ValueError("Only literal local weights are supported.")
        with safe_open(path, framework="pt", device="cpu") as tensors:
            count += sum(
                math.prod(tensors.get_slice(k).get_shape())
                for k in tuple(tensors.keys())
            )
    return count


def environment(directory, *, cpu=False):
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_NO_NETWORK": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "UNSLOTH_DISABLE_STATISTICS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
    }
    env.pop("LM_TRAINER_UNSLOTH_PYTHON", None)
    for key, name in (
        ("HF_HOME", "hf"),
        ("HF_DATASETS_CACHE", "datasets"),
        ("TRITON_CACHE_DIR", "triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("UNSLOTH_COMPILE_LOCATION", "compiled"),
        ("MPLCONFIGDIR", "mpl"),
        ("XDG_CACHE_HOME", "xdg"),
    ):
        env[key] = str(directory / name)
    if cpu:
        env.update(CUDA_VISIBLE_DEVICES="", ACCELERATE_USE_CPU="true")
    else:
        env.pop("ACCELERATE_USE_CPU", None)
    return env


def run_case(
    directory,
    config,
    model,
    *,
    interpreter,
    cwd,
    warmup,
    steps,
    label,
    length,
    repeat,
    cpu=False,
):
    directory.mkdir(parents=True, exist_ok=False)
    env = environment(directory / "cache", cpu=cpu)
    train_config, model_config = directory / "train.yaml", directory / "model.yaml"
    write_json(train_config, config)
    write_json(model_config, model)
    command = [
        interpreter,
        "-m",
        "lm_trainer",
        "--config",
        str(train_config),
        "--model-config",
        str(model_config),
    ]
    started = perf_counter()
    with (directory / "launcher.log").open("w") as log:
        process = subprocess.run(
            command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    wall = perf_counter() - started
    backend = config.get("backend", config.get("stages", [{}])[0].get("backend"))
    case = {
        "case": label,
        "length": length,
        "repeat": repeat,
        "backend": backend,
        "launcher_wall_seconds": wall,
        "returncode": process.returncode,
    }
    suites = config.get("stages", [config])
    if process.returncode != 0:
        log = (directory / "launcher.log").read_text(errors="replace")
        reason = (
            "OOM"
            if any(s in log.lower() for s in ("out of memory", "outofmemoryerror"))
            else "process_failed_see_local_log"
        )
        return {**case, "status": "FAIL", "reason": reason, "stages": None}
    reports = []
    for stage in suites:
        out = Path(stage.get("output_dir") or stage["checkpoint_dir"])
        metrics = json.loads((out / "performance.json").read_text())
        if stage.get("init_adapter"):
            m, sha = verify_artifact(out / "final", allow_fixture=True)
            assert m["fixture_only"] is True
            if stage["init_adapter"] == "continue_stage":
                assert m["parent_adapter_sha256"] == reports[-1]["artifact_sha256"]
        else:
            m, sha = {}, None  # older legacy contract has no fixture registry entry
        stats = summarize_updates(metrics["updates"], warmup)
        reports.append(
            {
                "stage_name": metrics.get("stage_name", "legacy"),
                "objective": stage.get("objective", "legacy"),
                "artifact_sha256": sha,
                "parent_adapter_sha256": m.get("parent_adapter_sha256"),
                "optimizer_updates": metrics["optimizer_updates"],
                "fixture_data": True,
                "identity": {
                    k: metrics.get(k)
                    for k in (
                        "device",
                        "driver",
                        "cuda",
                        "packages",
                        "source_sha256",
                        "lock_sha256",
                        "base_model",
                        "base_config_sha256",
                        "tokenizer_hash",
                        "template_policy",
                        "lora_config",
                        "compute_dtype",
                        "quantization",
                        "quantization_mode",
                        "data_sha256",
                    )
                },
                "controls": {
                    "microbatch": stage.get("batch_size", 1),
                    "gradient_accumulation_steps": stage["gradient_accumulation_steps"],
                    "gradient_checkpointing": stage["gradient_checkpointing"],
                    "optimizer": "torch.AdamW + linear scheduler",
                    "seed": stage["seed"],
                    "warmup_updates": warmup,
                    "token_budget": stage.get("token_budget"),
                    "target_loss_token_share": stage.get("target_loss_token_share"),
                    "requested_updates": steps,
                },
                "timing": {
                    k: metrics.get(k)
                    for k in (
                        "model_load_seconds",
                        "preprocess_seconds",
                        "save_seconds",
                        "initial_model_audit_seconds",
                        "preflight_seconds",
                        "dev_seconds",
                        "run_wall_seconds_including_setup_and_save",
                    )
                },
                "peak_allocated_bytes": metrics.get("peak_allocated_bytes"),
                "peak_reserved_bytes": metrics.get("peak_reserved_bytes"),
                "workload": metrics.get("actual_sampling", {}).get("sources"),
                "stop_reason": metrics.get("actual_sampling", {}).get(
                    "run_stop_reason"
                ),
                "steady_state": stats,
                "updates": metrics["updates"],
                "sample_class": "smoke_only"
                if stats["measured_updates"] < 10
                else "steady_sample",
            }
        )
    if label in ("mixed_cpt", "two_stage") and any(
        reports[0]["workload"][source]["sample_exposures"] == 0
        for source in ("chat", "notes")
    ):
        return {
            **case,
            "status": "FAIL",
            "reason": "CPT smoke did not consume both sources; tune shares or use more updates",
            "stages": reports,
        }
    return {**case, "status": "PASS", "stages": reports}


def parity(cases):
    results = []
    for a in cases:
        if a["backend"] != "hf" or a["case"] not in ("mixed_cpt", "direct_sft"):
            continue
        other = next(
            (
                b
                for b in cases
                if b["backend"] == "unsloth"
                and all(b[k] == a[k] for k in ("case", "length", "repeat"))
            ),
            None,
        )
        if other is None or any(c["status"] != "PASS" for c in (a, other)):
            results.append(
                {
                    "case": a["case"],
                    "length": a["length"],
                    "repeat": a["repeat"],
                    "status": "NOT RUN",
                    "reason": "both complete backend cases are required",
                }
            )
            continue
        left, right = (c["stages"][0] for c in (a, other))
        keys = (
            "batch_order_sha256",
            "batch_tensors_sha256",
            "loss_tokens",
            "input_tokens",
            "microbatches",
            "samples",
        )
        same = [[r[k] for k in keys] for r in left["updates"]] == [
            [r[k] for k in keys] for r in right["updates"]
        ]
        first_loss_delta = abs(left["updates"][0]["loss"] - right["updates"][0]["loss"])
        controls = left["controls"] == right["controls"] and all(
            left["identity"][k] == right["identity"][k]
            for k in (
                "device",
                "base_model",
                "tokenizer_hash",
                "template_policy",
                "source_sha256",
                "lora_config",
                "quantization",
                "compute_dtype",
                "data_sha256",
            )
        )
        results.append(
            {
                "case": a["case"],
                "length": a["length"],
                "repeat": a["repeat"],
                "status": "PASS"
                if same and controls and first_loss_delta <= 0.03
                else "FAIL",
                "batch_tensors_order_and_budgets_equal": same,
                "static_controls_equal": controls,
                "first_update_loss_absolute_delta": first_loss_delta,
                "loss_atol": 0.03,
                "isolates_backend_speedup": False,
            }
        )
    return results


def aggregate(cases):
    """Pool actual measured denominators across repetitions, not per-run rates."""
    groups = {}
    for case in cases:
        if case["status"] != "PASS":
            continue
        for stage in case["stages"]:
            key = (case["case"], case["length"], case["backend"], stage["stage_name"])
            groups.setdefault(key, []).append(stage)
    rows = []
    for (case, length, backend, name), runs in groups.items():
        measured = [
            r["steady_state"] for r in runs if r["steady_state"]["measured_updates"]
        ]
        seconds = sum(m["measured_seconds"] for m in measured)
        rows.append(
            {
                "case": case,
                "length": length,
                "backend": backend,
                "stage_name": name,
                "repeats": len(runs),
                "measured_seconds": seconds or None,
                "sample_class": "steady_repeated"
                if len(runs) >= 3
                and all(r["sample_class"] == "steady_sample" for r in runs)
                else "smoke_or_limited",
                "measured_updates": sum(m["measured_updates"] for m in measured),
                **{
                    f"{key}_per_second": sum(m[f"measured_{key}"] for m in measured)
                    / seconds
                    if seconds
                    else None
                    for key in ("input_tokens", "loss_tokens", "samples")
                },
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", required=True, choices=("L2", "L3", "L4"))
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--local-base", type=Path)
    parser.add_argument("--unsloth-python", type=Path)
    parser.add_argument("--legacy-baseline-root", type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--lengths", type=int, nargs="+")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--quantization", choices=("none", "4bit"))
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument(
        "--cpt-chat-share",
        type=float,
        default=0.05,
        help="Target CPT chat token share; small smoke runs must still sample long notes.",
    )
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--stop-token", action="append")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=(
            "mixed_cpt",
            "direct_sft",
            "two_stage",
            "legacy_after",
            "legacy_before",
        ),
    )
    args = parser.parse_args()
    if not 0 <= args.warmup < args.steps <= 1000 or args.repeats < 1 or args.rank < 1:
        parser.error("Invalid steps/warmup/repeats/rank.")
    if not 0 < args.cpt_chat_share < 1:
        parser.error("CPT chat share must be between zero and one.")
    if args.cpu and (args.level != "L2" or args.unsloth_python):
        parser.error("CPU is only for L2 and HF.")
    if args.level != "L2" and args.local_base is None:
        parser.error("L3/L4 require an existing local unquantized base. No downloads.")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    os.environ.update(environment(work / "prepare_cache", cpu=args.cpu))
    if args.local_base is not None:
        from lm_trainer.model.loader import load_tokenizer

        base = args.local_base.resolve(strict=True)
        count = parameter_elements(base)
        low, high = {
            "L2": (0, 100_000_000),
            "L3": (2_000_000_000, 9_000_000_000),
            "L4": (12_000_000_000, 16_000_000_000),
        }[args.level]
        if not low < count <= high:
            parser.error(
                "Local weights do not fit the requested level; 27B is excluded."
            )
        tokenizer = load_tokenizer(str(base), use_fast=True, local_files_only=True)
    else:
        base = work / "tiny_base"
        tokenizer = tiny_base(base)
        count = parameter_elements(base)
    shared_tokenizer = work / "pinned_tokenizer"
    tokenizer.save_pretrained(shared_tokenizer)
    from lm_trainer.train.modeling import load_stage_tokenizer

    tokenizer = load_stage_tokenizer({"tokenizer_name_or_path": str(shared_tokenizer)})
    lengths = args.lengths or (
        [256, 512, 1024] if args.level == "L2" else [512, 1024, 2048]
    )
    dtype = args.dtype or ("fp32" if args.cpu else "bf16")
    quant = args.quantization or ("none" if args.level == "L2" else "4bit")
    policy = {**POLICY, "stop_tokens": args.stop_token or POLICY["stop_tokens"]}
    if args.disable_thinking:
        policy["enable_thinking"] = False
    model = {
        "lora_r": args.rank,
        "lora_alpha": 2 * args.rank,
        "lora_target_modules": args.target_modules,
        "lora_dropout": 0.0,
        "lora_bias": "none",
    }
    backends = [("hf", sys.executable)]
    if args.unsloth_python:
        backends.append(("unsloth", str(args.unsloth_python.absolute())))
    cases = []
    for length in lengths:
        tr = workload(
            work / f"data_{length}",
            tokenizer,
            length=length,
            count=max(16, 3 * args.steps),
        )
        for repeat in range(1, args.repeats + 1):
            for backend, interpreter in backends:
                for label in (
                    "mixed_cpt",
                    "direct_sft",
                    "two_stage",
                    "legacy_after",
                    "legacy_before",
                ):
                    if args.cases and label not in args.cases:
                        continue
                    if label == "two_stage" and (
                        backend != "hf" or (args.level != "L2" and length < 1024)
                    ):
                        continue
                    if label == "direct_sft" and args.level != "L2" and length < 1024:
                        continue
                    if label.startswith("legacy") and (
                        length != 512
                        or (label == "legacy_before" and not args.legacy_baseline_root)
                    ):
                        continue
                    path = work / f"{label}_{length}_{backend}_{repeat}"
                    cpt, sft = stage_configs(
                        path,
                        base,
                        tr,
                        None,
                        length=length,
                        steps=args.steps,
                        bench_warmup=args.warmup,
                        backend=backend,
                        dtype=dtype,
                        quantization=quant,
                        policy=policy,
                    )
                    cpt["target_loss_token_share"] = {
                        "chat": args.cpt_chat_share,
                        "notes": 1 - args.cpt_chat_share,
                    }
                    for stage in (cpt, sft):
                        stage["tokenizer_name_or_path"] = str(shared_tokenizer)
                    if label == "two_stage":
                        sft.update(
                            init_adapter="continue_stage", parent="stage:mixed_cpt"
                        )
                        cfg = {"stages": [cpt, sft]}
                    elif label.startswith("legacy"):
                        raw = work / f"legacy_{length}.jsonl"
                        if not raw.exists():
                            lines = (
                                tr.parent / "chat.jsonl"
                            ).read_text().splitlines() + (
                                tr.parent / "notes.jsonl"
                            ).read_text().splitlines()
                            raw.write_text(
                                "".join(
                                    json.dumps(
                                        {"text": json.loads(line)["text"]},
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                    for line in lines
                                )
                            )
                        cfg = {
                            k: v
                            for k, v in cpt.items()
                            if k
                            in (
                                "model_name_or_path",
                                "quantization_mode",
                                "compute_dtype",
                                "max_length",
                                "max_steps",
                                "backend",
                                "batch_size",
                                "gradient_checkpointing",
                                "gradient_accumulation_steps",
                                "seed",
                                "lr",
                                "warmup_steps",
                                "benchmark_warmup_steps",
                                "save_steps",
                            )
                        }
                        cfg.update(
                            data_file_path=str(raw),
                            checkpoint_dir=str(path / "legacy"),
                            epochs=1,
                            shuffle=True,
                            auto_pad_batch=True,
                            num_workers=0,
                            pin_memory=False,
                        )
                    else:
                        cfg = cpt if label == "mixed_cpt" else sft
                    case = run_case(
                        path,
                        cfg,
                        model,
                        interpreter=interpreter,
                        cwd=args.legacy_baseline_root.resolve()
                        if label == "legacy_before"
                        else REPO,
                        warmup=args.warmup,
                        steps=args.steps,
                        label=label,
                        length=length,
                        repeat=repeat,
                        cpu=args.cpu,
                    )
                    cases.append(case)
                    print(
                        f"{label:12} {backend:7} length={length:4} rep={repeat}: {case['status']}",
                        flush=True,
                    )
                    write_json(
                        work / "benchmark.json",
                        {
                            "schema": "lm_trainer.benchmark.v1",
                            "level": args.level,
                            "fixture_only": True,
                            "base_saved_parameter_elements": count,
                            "cases": cases,
                            "cross_backend_parity": parity(cases),
                            "aggregate": aggregate(cases),
                            "notes": [
                                "Proof of engineering only; this data is fictional.",
                                "HF deterministic kernels; Unsloth patching, kernels and dependency pins differ.",
                                "Loading, saving, preprocessing and dev are excluded from measured update rate.",
                            ],
                        },
                    )
    return int(any(c["status"] == "FAIL" for c in [*cases, *parity(cases)]))


if __name__ == "__main__":
    sys.exit(main())
