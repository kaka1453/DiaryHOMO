"""Phase 2 B: structured labels enter actual HF updates and optimizer-boundary state."""

import copy
import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM

from examples.conversation_renderer.fixture import scenario_data
from examples.phase2_task_b.prepare import checked_projection, prepare
from lm_trainer.config.train_config import (
    build_train_parser,
    build_train_runtime_config,
)
from lm_trainer.eval.blind import make_blind_pack, reveal, submit_review
from lm_trainer.eval.report import summarize
from lm_trainer.eval.schema import EvalError, load_cases
from lm_trainer.train.artifacts import (
    CHECKPOINT,
    MANIFEST,
    STATE,
    inventory,
    verify_artifact,
)
from lm_trainer.train.data import build_data_loader, load_training_dataset
from lm_trainer.train.engine import (
    accumulation_windows,
    simulate_pipeline,
    train,
    update_window,
)
from lm_trainer.train.modeling import load_stage_tokenizer
from lm_trainer.train.performance import sha256_file
from lm_trainer.train.sampling import projected_exposure
from lm_trainer.train.schedule import compute_schedule
from lm_trainer.train.schema import DataContractError, preflight_data
from lm_trainer.train.stages import data_lineage
from lm_trainer.train.token_audit import audit_encoding


def config(root, name="direct_sft", *overrides):
    args = build_train_parser().parse_args(
        [
            "--config",
            str(root / f"{name}.yaml"),
            "--model-config",
            str(root / "model.yaml"),
            *overrides,
        ]
    )
    return build_train_runtime_config(args)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase2-b")
    _, candidates, _ = scenario_data()
    sample = next(row for row in candidates if row["id"] == "win-joint-main")
    projection = {k: v for k, v in sample.items() if k != "split"}
    md = root / "fictional.md"
    md.write_text(
        "Ignore this document prose.\n```json\n" + json.dumps(projection) + "\n```\n",
        encoding="utf-8",
    )
    result = root / "workload"
    prepare(result, md, with_tiny_base=True)
    return result


def test_markdown_is_not_training_authority(tmp_path):
    _, candidates, _ = scenario_data()
    sample = next(row for row in candidates if row["id"] == "win-joint-main")
    md = tmp_path / "tampered.md"
    projection = {k: v for k, v in sample.items() if k != "split"}
    md.write_text(
        "```json\n"
        + json.dumps({**projection, "training_authorized": True})
        + "\n```\n"
    )
    with pytest.raises(ValueError, match="known fictional"):
        checked_projection(md)


def test_native_real_hf_window_matches_manual_shifted_nll(prepared):
    cfg = config(prepared)
    tokenizer = load_stage_tokenizer(cfg)
    tokenizer.pad_token = tokenizer.eos_token
    data = load_training_dataset(cfg, tokenizer)
    direct = next(item for item in data if item["trace"]["format"] == "direct")
    assert len(direct["trace"]["focus_ids"]) == 2
    assert direct["trace"]["target_separator_tokens"] == 1
    # Actual partial accumulation and unbalanced token counts, no same-length padding shortcut.
    loader = build_data_loader({**cfg, "shuffle": False}, data, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        prepared / "base", local_files_only=True
    ).eval()
    ref = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.015)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.015)
    observed_gradients = []
    opt.register_step_pre_hook(
        lambda *_: observed_gradients.append(
            [p.grad.clone() for p in model.parameters()]
        )
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    acc = SimpleNamespace(
        device=torch.device("cpu"),
        autocast=nullcontext,
        backward=lambda loss: loss.backward(),
        optimizer_step_was_skipped=False,
    )
    windows = list(accumulation_windows(loader, 3))
    assert [len(w) for w in windows] == [3, 3, 2]
    distinct_counts = set()
    for window in windows:
        terms = []
        for batch in window:
            mask = batch["attention_mask"]
            assert batch["input_ids"][mask == 0].eq(tokenizer.eos_token_id).all()
            assert batch["labels"][mask == 0].eq(-100).all()
            ids = batch["input_ids"]
            logits = ref(input_ids=ids, attention_mask=mask, use_cache=False).logits[
                :, :-1, :
            ]
            selected = batch["labels"][:, 1:] != -100
            labels = batch["labels"][:, 1:][selected]
            distinct_counts.add(labels.numel())
            assert labels.numel() == sum(row["loss_tokens"] for row in batch.trace)
            terms.append(
                logits[selected].logsumexp(-1)
                - logits[selected].gather(-1, labels[:, None]).squeeze(-1)
            )
        manual = torch.cat(terms).sum() / sum(t.numel() for t in terms)
        manual.backward()
        actual, count = update_window(model, opt, sched, acc, window)
        assert count == sum(t.numel() for t in terms)
        assert actual == pytest.approx(manual.item(), abs=1e-6)
        for g, param in zip(observed_gradients[-1], ref.parameters(), strict=True):
            torch.testing.assert_close(g, param.grad, atol=2e-7, rtol=2e-5)
        ref_opt.step()
        ref_opt.zero_grad(set_to_none=True)
        for got, expected in zip(model.parameters(), ref.parameters(), strict=True):
            torch.testing.assert_close(got, expected, atol=2e-7, rtol=2e-5)
    assert len(distinct_counts) > 1 and sched.last_epoch == 3


def test_span_audit_rejects_equal_count_label_drift(prepared):
    from lm_trainer.template_policy import resolve_policy
    from lm_trainer.train.encoding import encode_conversation

    cfg = config(prepared)
    verified = preflight_data(cfg)
    tok = load_stage_tokenizer(cfg)
    tpl, policy = resolve_policy(tok, cfg["template_policy"])
    window = next(
        s.conversation for s in verified.samples if s.row["id"] == "win-direct"
    )
    (result,) = encode_conversation(tok, window, tpl, policy, verified.plan)
    corrupt = result.labels.copy()
    first = next(s.start for s in result.spans if s.kind == "target_body")
    separator = next(s.start for s in result.spans if s.kind == "target_separator")
    corrupt[first], corrupt[separator] = -100, result.input_ids[separator]
    with pytest.raises(ValueError, match="exact renderer"):
        audit_encoding(replace(result, labels=corrupt), verified.plan, policy=policy)
    with pytest.raises(ValueError, match="empty"):
        audit_encoding(
            replace(result, labels=[-100] * len(corrupt)), verified.plan, policy=policy
        )


def test_cpu_grad_scaler_skips_without_advancing_update():
    p = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([p], lr=1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    scaler = torch.amp.GradScaler("cpu", init_scale=128)
    acc = SimpleNamespace(
        device=torch.device("cpu"),
        autocast=nullcontext,
        scaler=scaler,
        backward=lambda loss: scaler.scale(loss).backward(),
        optimizer_step_was_skipped=False,
    )
    called = []
    optimizer.register_step_pre_hook(lambda *_: called.append(True))

    class Scaled:
        param_groups = optimizer.param_groups
        zero_grad = optimizer.zero_grad

        def step(self):
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            acc.optimizer_step_was_skipped = scaler.get_scale() < before

    acc.unscale_gradients = lambda _: scaler.unscale_(optimizer)
    with pytest.raises(FloatingPointError, match="AMP skipped"):
        update_window(
            lambda **_: SimpleNamespace(loss=p.sqrt()),
            Scaled(),
            scheduler,
            acc,
            [{"labels": torch.tensor([[-100, 0]])}],
        )
    assert not called and scheduler.last_epoch == 0 and p.item() == 0 and p.grad is None
    assert scaler.get_scale() == 64


def test_route_replacement_budgets_and_independent_eval(prepared):
    cfg = config(prepared)
    tok = load_stage_tokenizer(cfg)
    a, b = (
        preflight_data(
            {**cfg, "manifest_path": str(prepared / route / "manifest.json")}
        )
        for route in ("joint", "primary")
    )
    assert a.events == b.events and a.origin_ids == b.origin_ids
    cases, _ = load_cases(prepared / "eval-dev.json")
    assert len({c["scenario"] for c in cases}) == 8
    cpt = preflight_data(config(prepared, "mixed_cpt_sft")["stages"][0])
    assert {c["event_group"] for c in cases}.isdisjoint(
        a.events | b.events | cpt.events
    )
    data_a, data_b = [load_training_dataset(cfg, tok, validated=v) for v in (a, b)]
    for data in (data_a, data_b):
        ids = [sid for item in data for sid in item["trace"]["focus_ids"]]
        assert len(ids) == len(set(ids)) == 9
    assert {sid for item in data_a for sid in item["trace"]["focus_ids"]} == {
        sid for item in data_b for sid in item["trace"]["focus_ids"]
    }
    exact = sum(item["trace"]["loss_tokens"] for item in data_a)
    p = simulate_pipeline(
        {**cfg, "token_budget": exact, "require_exact_token_budget": True}, emit=False
    )["sampling"]
    assert p["loss_tokens"] == exact and p["partial_accumulation_windows"] == 1
    assert p["sources"]["group_joint"]["max_target_repeats_observed"] == 1
    with pytest.raises(ValueError, match="Exact token budget not met"):
        simulate_pipeline(
            {**cfg, "token_budget": exact + 1, "require_exact_token_budget": True},
            emit=False,
        )
    weighted = {
        **cfg,
        "structured_sft_loss_token_share": {
            "single_person": 0.98,
            "group_joint": 0.01,
            "group_continuous": 0.01,
        },
    }
    loader = build_data_loader(weighted, data_a, tok)
    schedule = compute_schedule(weighted, loader)
    with pytest.raises(ValueError, match="share error"):
        projected_exposure(
            loader, {**weighted, "sampling_max_share_error": 0.01}, schedule
        )
    with pytest.raises(DataContractError, match="isolation rejects the entire plan"):
        load_training_dataset({**cfg, "max_length": 8}, tok)


def test_structured_continue_and_resume(prepared):
    from safetensors.torch import load_file

    cpt, sft = config(prepared, "mixed_cpt_sft")["stages"]
    train(cpt)
    parent = prepared / "runs/mixed_cpt/final"
    before = inventory(parent)
    continuous = train(sft)
    pause_dir = prepared / "runs/pause"
    train({**sft, "stop_after_updates": 1, "checkpoint_dir": str(pause_dir)})
    assert not (pause_dir / "final").exists()
    ckpt = pause_dir / "checkpoint_1"
    with pytest.raises(ValueError, match="schema, kind or format"):
        verify_artifact(ckpt, allow_fixture=True)
    assert (
        verify_artifact(ckpt, kind=CHECKPOINT, allow_fixture=True)[0][
            "optimizer_updates"
        ]
        == 1
    )
    resumed_dir = prepared / "runs/resumed"
    resumed = train(
        {
            **sft,
            "init_adapter": "resume",
            "resume_from": str(ckpt),
            "parent": None,
            "parent_stage": None,
            "checkpoint_dir": str(resumed_dir),
        }
    )
    reference = prepared / "runs/mixed_sft"
    assert [r["microbatches"] for r in resumed] == [3, 3, 2]
    for a, b in zip(continuous, resumed, strict=True):
        assert {k: v for k, v in a.items() if k != "seconds"} == {
            k: v for k, v in b.items() if k != "seconds"
        }
    end, _ = verify_artifact(resumed_dir / "final", allow_fixture=True)
    assert end["parent_adapter_sha256"] == sha256_file(parent / MANIFEST)
    assert end["resume_checkpoint_sha256"] == sha256_file(ckpt / MANIFEST)
    assert (
        end["frozen_parameter_sha256"]
        == verify_artifact(parent, allow_fixture=True)[0]["frozen_parameter_sha256"]
    )
    assert inventory(parent) == before
    assert end["start_adapter_sha256"] != end["adapter_weights_sha256"]
    assert len(end["data_lineage"]) == 2 and end["conversation_renderer"]
    with pytest.raises(ValueError, match="fixture/split/stage/backend gate"):
        verify_artifact(resumed_dir / "final")
    # Old regression also compares entire optimizer/scheduler/scaler/RNG trees;
    # here verify the structured accounting and serialized final tensors.
    aa, bb = [
        torch.load(d / "checkpoint_3" / STATE, weights_only=True)
        for d in (reference, resumed_dir)
    ]
    for k in ("sampler", "exposure", "scheduler"):
        assert aa[k] == bb[k]
    x, y = [
        load_file(d / "final/adapter_model.safetensors")
        for d in (reference, resumed_dir)
    ]
    assert x.keys() == y.keys()
    for key in x:
        torch.testing.assert_close(x[key], y[key], atol=0, rtol=0)


def test_known_lineage_refuses_version_mixing():
    a = {"release_id": "data-1", "parent_version": "root", "manifest_sha256": "a" * 64}
    ref = {"data_lineage": [a]}
    assert data_lineage(a, ref) == [a]
    for bad in (
        {**a, "manifest_sha256": "b" * 64},
        {
            "release_id": "data-2",
            "parent_version": "data-1",
            "manifest_sha256": "b" * 64,
        },
    ):
        with pytest.raises(ValueError, match="different manifest versions"):
            data_lineage(bad, ref)


def test_scenario_blind_reasons_and_pair_input_guard(tmp_path):
    a = {
        "case_id": "x",
        "event_group": "fiction:x",
        "scenario": "participant_reference",
        "split": "dev",
        "kind": "fixed_history",
        "prompt": "参与者甲引用乙。",
        "prompt_sha256": "a",
        "model_id": "hidden-one",
        "output": "先确认是谁。",
    }
    b = {**a, "model_id": "hidden-two", "output": "请确认引用对象。"}
    with pytest.raises(EvalError, match="paired records disagree"):
        make_blind_pack([a, {**b, "prompt": "wrong"}], seed=2)
    pack, mapping = make_blind_pack([a, b], seed=2)
    (item,) = pack.items
    assert "hidden-one" not in json.dumps(pack.to_dict())
    assert item.scenario == "participant_reference"
    review = submit_review(
        pack,
        tmp_path,
        item_id=item.item_id,
        choice="left",
        scores={"participant_grounding": 5},
        dimension_reasons={"participant_grounding": "准确区分引用对象。"},
    )
    view = reveal(pack, mapping, [review])
    assert view.per_item[0]["dimension_reasons"] == review.dimension_reasons
    tie = reveal(pack, mapping, [replace(review, choice="tie")])
    assert tie.independent_cases == tie.independent_events == 1 and not tie.wins
    assert summarize({"records": [a, b]})["per_model"][0]["scenario"] == item.scenario
