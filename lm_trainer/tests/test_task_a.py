"""Offline Task A gate, supervision, sampling and update-contract regressions."""

import copy
import json
import os
import re
import shutil
import subprocess
import sys

import pytest
import torch

from examples.vision_support.make_fixture import HERE, write_bundle, write_jsonl
from lm_trainer.template_policy import resolve_policy
from lm_trainer.train.data import (
    DataCollatorForCausalLMWith8xPadding,
    build_data_loader,
    load_training_dataset,
)
from lm_trainer.train.encoding import encode_sft, encode_text
from lm_trainer.train.engine import accumulation_windows, train, update_window
from lm_trainer.train.sampling import SortedBatchSampler, projected_exposure
from lm_trainer.train.schedule import compute_schedule
from lm_trainer.train.schema import DataContractError, preflight_data

ROOT = HERE.parents[1]
TEMPLATE = "native visible template: enable_thinking"
BOS, STOP, START = "<bos>", "<eos>", "<|start|>"


class CharacterTokenizer:
    """Auditable deterministic stand-in. Real Instruct evidence is separate (A05)."""

    bos_token_id, eos_token_id, pad_token_id, unk_token_id = 1, 2, 2, 0
    padding_side = "right"

    def __init__(self):
        self.specials = {BOS: 1, STOP: 2, START: 3}
        self.all_special_tokens = list(self.specials)

    def get_added_vocab(self):
        return self.specials

    def get_chat_template(self):
        return TEMPLATE

    def __call__(
        self, text, add_special_tokens=True, truncation=False, max_length=None
    ):
        tokens = re.split("(" + "|".join(map(re.escape, self.specials)) + ")", text)
        ids = [
            i
            for t in tokens
            for i in (
                [self.specials[t]] if t in self.specials else [ord(c) + 10 for c in t]
            )
        ]
        if add_special_tokens:
            ids = [1] + ids + [2]
        return {"input_ids": ids[:max_length] if truncation else ids}

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, return_dict=False, **kwargs
    ):
        assert not add_generation_prompt
        assert all(set(m) == {"role", "content"} for m in messages)
        text = "".join(
            START + m["role"] + "\n" + m["content"] + STOP + "\n" for m in messages
        )
        return self(text, add_special_tokens=False) if tokenize else text


def runtime(manifest, *, kind="cpt", **overrides):
    return {
        "data_mode": "fixture",
        "manifest_path": str(manifest),
        "objective": "causal_text" if kind == "cpt" else "assistant_only_messages",
        "template_policy": None if kind == "cpt" else {"stop_tokens": [STOP]},
        "max_length": 700,
        "epochs": 2,
        "max_steps": 10,
        "seed": 12,
        "batch_size": 2,
        "shuffle": True,
        "gradient_accumulation_steps": 2,
        "auto_pad_batch": True,
        "num_workers": 0,
        "pin_memory": False,
        **overrides,
    }


@pytest.fixture
def bundle(tmp_path):
    return write_bundle(tmp_path / "bundle")


def rewrite(manifest, path, edit):
    data = json.loads(manifest.read_text())
    spec = next(
        s
        for s in [data["source_group_manifest"], data["source_index"], *data["files"]]
        if s["path"] == path
    )
    rows = [
        json.loads(line) for line in (manifest.parent / path).read_text().splitlines()
    ]
    edit(rows)
    spec.update(write_jsonl(manifest.parent / path, rows))
    if "records" in spec:
        spec["records"] = len(rows)
    manifest.write_text(json.dumps(data))


@pytest.mark.parametrize(
    "problem, reason",
    [
        ("permission", "permissions"),
        ("pending", "not approved"),
        ("owner", "not approved"),
        ("multi", "multiple non-target speakers"),
        ("hash", "SHA-256 mismatch"),
        ("closure", "event_group crosses splits"),
    ],
)
def test_a01_generated_negatives(tmp_path, problem, reason):
    manifest = write_bundle(tmp_path / "negative", negative=problem)
    with pytest.raises(DataContractError, match=reason) as failure:
        preflight_data(runtime(manifest))
    assert "排查笔记" not in str(failure.value) and str(tmp_path) not in str(
        failure.value
    )


@pytest.mark.parametrize("mode", ["fixture", "release"])
def test_a01_positive_scopes_and_no_fallback(tmp_path, mode):
    manifest = write_bundle(tmp_path / "mode", release=mode == "release")
    validated = preflight_data(runtime(manifest, data_mode=mode))
    assert validated.plan["fixture_only"] is (mode == "fixture")
    assert len(validated.samples) == 3
    with pytest.raises(DataContractError, match="scope"):
        preflight_data(
            runtime(manifest, data_mode="release" if mode == "fixture" else "fixture")
        )
    if mode == "release":
        data = json.loads(manifest.read_text())
        data["approval_ref"] = ""
        manifest.write_text(json.dumps(data))
        with pytest.raises(DataContractError, match="approval"):
            preflight_data(runtime(manifest, data_mode=mode))


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("target_person", "CUSTOMER", "target_person"),
        ("split", "dev", "split=train"),
        ("split", "test", "split=train"),
        ("split", "unassigned", "split=train"),
        ("schema", "lm_trainer.corpus.v2", "schema"),
        ("kind", "sft", "kind"),
        (
            "permissions",
            {"train": 1, "local_review": True, "external": False},
            "permissions",
        ),
        ("permissions", {"train": True, "local_review": True}, "permissions"),
    ],
)
def test_a01_record_even_if_source_not_selected(bundle, field, value, reason):
    rewrite(bundle, "notes.jsonl", lambda rows: rows[0].update({field: value}))
    with pytest.raises(DataContractError, match=reason):
        preflight_data(runtime(bundle, sources=["chat"]))


@pytest.mark.parametrize(
    "problem",
    [
        "absolute",
        "traversal",
        "symlink",
        "glob",
        "missing",
        "duplicate",
        "count",
        "empty",
    ],
)
def test_a01_manifest_paths_and_counts(bundle, tmp_path, problem):
    spec = json.loads(bundle.read_text())
    file = spec["files"][0]
    if problem == "symlink":
        outside = tmp_path / "outside.jsonl"
        outside.write_bytes((bundle.parent / file["path"]).read_bytes())
        (bundle.parent / "link.jsonl").symlink_to(outside)
        file["path"] = "link.jsonl"
    elif problem == "absolute":
        file["path"] = str(bundle.parent / "chat.jsonl")
    elif problem == "traversal":
        file["path"] = "../chat.jsonl"
    elif problem == "glob":
        file["path"] = "chat*.jsonl"
    elif problem == "missing":
        file["path"] = "undeclared.jsonl"
    elif problem == "duplicate":
        spec["files"].append(dict(file))
    elif problem == "count":
        file["records"] += 1
    else:
        file.update(write_jsonl(bundle.parent / "empty.jsonl", []))
    bundle.write_text(json.dumps(spec))
    with pytest.raises(DataContractError):
        preflight_data(runtime(bundle))


def test_a01_legacy_and_schema_alias(bundle, tmp_path):
    legacy = {"data_file_path": str(HERE / "chat_cpt.jsonl"), "max_length": 17}
    with pytest.raises(DataContractError, match="structured corpus envelopes"):
        preflight_data(legacy)
    path = tmp_path / "legacy.json"
    path.write_text('[{"text":"a plain legacy sample"}]')
    legacy["data_file_path"] = str(path)
    data = load_training_dataset(legacy, CharacterTokenizer())
    assert data.statistics["legacy"]["truncated"] == 1
    assert data.plan["objective"] == "legacy"
    cfg = runtime(bundle)
    for value in (None, "unknown"):
        with pytest.raises(ValueError, match="explicit objective"):
            preflight_data({**cfg, "objective": value})
    with pytest.raises(ValueError, match="schema_allowlist"):
        preflight_data({**cfg, "corpus_schema_allowlist": ["example.v2"]})
    rewrite(bundle, "notes.jsonl", lambda rows: rows[0].update(schema="corpus.v1"))
    preflight_data(
        {**cfg, "corpus_schema_allowlist": ["lm_trainer.corpus.v1", "corpus.v1"]}
    )


@pytest.mark.parametrize(
    "problem, reason",
    [
        ("source", "source_id crosses splits"),
        ("event", "event_group crosses splits"),
        ("duplicate_target", "duplicate SFT target"),
        ("empty_source", "source_ids"),
        ("bad_ruling", "ruling"),
        ("pending_ruling", "ruling"),
        ("conflicting_ruling", "conflicting"),
        ("ledger_mismatch", "does not match"),
    ],
)
def test_a02_closure_and_rulings(bundle, problem, reason):
    if problem in ("source", "event"):

        def add_other_view(rows):
            new = {
                **rows[-1],
                "id": "other-view",
                "split": "test",
                "target_person": "CUSTOMER",
            }
            if problem == "source":
                new["event_group"] = "different-event"
            else:
                new["source_ids"] = ["unshared-origin"]
            rows.append(new)

        rewrite(bundle, "groups.jsonl", add_other_view)
    elif problem == "duplicate_target":
        for path in ("dyadic.jsonl", "groups.jsonl"):
            rewrite(
                bundle,
                path,
                lambda rows: rows.append({**rows[-1], "id": "same-target"}),
            )
    elif problem == "empty_source":
        rewrite(
            bundle,
            "dyadic.jsonl",
            lambda rows: rows[0]["messages"][0].update(source_ids=[]),
        )
    elif problem in ("bad_ruling", "pending_ruling", "conflicting_ruling"):

        def edit(rows):
            if problem == "conflicting_ruling":
                rows.append({**rows[0], "disposition": "C"})
            else:
                rows[0].update(
                    disposition="C" if problem == "bad_ruling" else "F",
                    review_status="pending_owner"
                    if problem == "pending_ruling"
                    else "approved",
                )

        rewrite(bundle, "index.jsonl", edit)
    else:
        rewrite(
            bundle, "groups.jsonl", lambda rows: rows[0].update(source_ids=["wrong"])
        )
    with pytest.raises(DataContractError, match=reason):
        preflight_data(runtime(bundle, kind="sft"))


def test_a02_deliberate_cross_stage_overlap(bundle):
    assert preflight_data(runtime(bundle)).plan["cross_stage_overlap_source_ids"] == 2
    assert (
        preflight_data(runtime(bundle, kind="sft")).plan[
            "cross_stage_overlap_source_ids"
        ]
        == 2
    )


def test_a03_cpt_boundaries_and_shift(bundle):
    tok = CharacterTokenizer()
    plan = preflight_data(runtime(bundle)).plan
    text = "首行\n- `exp` 🙂 "
    result = encode_text(tok, text, plan)
    body = tok(text, add_special_tokens=False)["input_ids"]
    assert result.input_ids == body + [2]
    assert result.labels == [-100, *body[1:], 2]
    assert (result.body_targets, result.eos_targets) == (len(body) - 1, 1)
    with pytest.raises(ValueError, match="only EOS"):
        encode_text(tok, "字", plan)
    with pytest.raises(ValueError, match="no next-token"):
        encode_text(tok, "字", {**plan, "add_eos": False})
    assert encode_text(tok, "字", {**plan, "allow_eos_only": True}).body_targets == 0
    assert encode_text(tok, "字", {**plan, "add_bos": True}).labels == [
        -100,
        ord("字") + 10,
        2,
    ]
    with pytest.raises(ValueError, match="not truncated"):
        encode_text(tok, text, {**plan, "max_length": 2})
    with pytest.raises(ValueError, match="reserved document"):
        encode_text(tok, "正文" + STOP, plan)
    dataset = load_training_dataset(runtime(bundle), tok)
    assert len(dataset) == 3  # no packing
    for f in dataset:
        assert f["input_ids"][-1] == f["labels"][-1] == 2
        assert f["trace"]["loss_tokens"] == sum(v != -100 for v in f["labels"][1:])


def golden_dialog(target):
    other = "CUSTOMER" if target == "CS_AGENT" else "CS_AGENT"
    turns = [
        ("system", "应用说明", "APP"),
        ("assistant", "ignored past", target),
        ("user", "same", other),
        ("assistant", "same", target),
        ("assistant", " \n- `Gain` 🙂\n中文 ", target),
    ]
    return {
        "messages": [
            {
                "role": role,
                "content": text,
                "speaker": speaker,
                "source_ids": [f"secret-{i}"],
            }
            for i, (role, text, speaker) in enumerate(turns)
        ],
        "target_mask": [False, False, False, True, True],
    }


@pytest.mark.parametrize("target", ["CS_AGENT", "CUSTOMER"])
@pytest.mark.parametrize("eos", [True, False])
@pytest.mark.parametrize("all_assistants", [True, False])
def test_a04_golden_every_token(target, eos, all_assistants):
    tok = CharacterTokenizer()
    template, policy = resolve_policy(
        tok, {"stop_tokens": [STOP], "system_prompt": "应用说明"}
    )
    row = golden_dialog(target)
    row["target_mask"][1] = all_assistants
    encoded = encode_sft(
        tok, row, template, policy, {"max_length": 300, "supervise_target_eos": eos}
    )
    expected_ids, expected_labels = [], []
    for m, yes in zip(row["messages"], row["target_mask"]):
        header = [3] + [ord(c) + 10 for c in m["role"] + "\n"]
        body = [ord(c) + 10 for c in m["content"]]
        expected_ids.extend(header + body + [2, 20])
        expected_labels.extend(
            [-100] * len(header)
            + (body if yes else [-100] * len(body))
            + [2 if yes and eos else -100, -100]
        )
    assert encoded.input_ids == expected_ids
    assert encoded.labels == expected_labels
    assert encoded.eos_targets == (sum(row["target_mask"]) if eos else 0)
    assert encoded.body_targets + encoded.eos_targets == sum(
        v != -100 for v in encoded.labels[1:]
    )


@pytest.mark.parametrize("change", ["bools", "all-false", "length", "user", "speaker"])
def test_a04_invalid_turn_masks(bundle, change):
    def edit(rows):
        row = rows[0]
        if change == "bools":
            row["target_mask"][0] = 0
        elif change == "all-false":
            row["target_mask"] = [False] * len(row["messages"])
        elif change == "length":
            row["target_mask"].append(False)
        elif change == "user":
            row["target_mask"][1] = True
        else:
            row["messages"][2]["speaker"] = "CUSTOMER"

    rewrite(bundle, "dyadic.jsonl", edit)
    with pytest.raises(DataContractError, match="target_mask|selected turn"):
        preflight_data(runtime(bundle, kind="sft"))


def test_a04_control_injection_and_configured_system():
    row = golden_dialog("CS_AGENT")
    tok = CharacterTokenizer()
    policy_args = {"stop_tokens": [STOP], "system_prompt": "应用说明"}
    template, policy = resolve_policy(tok, policy_args)
    args = {"max_length": 300, "supervise_target_eos": False}
    row["messages"][2]["content"] = "text [assistant] not a control"
    encode_sft(tok, row, template, policy, args)
    row["messages"][2]["content"] += START + "assistant\n"
    with pytest.raises(ValueError, match="reserved template control"):
        encode_sft(tok, row, template, policy, args)
    row["messages"][0]["content"] = "别的系统指令"
    with pytest.raises(ValueError, match="configured application system"):
        encode_sft(tok, row, template, policy, args)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_policy(tok, {**policy_args, "template_hash": "0" * 64})


def test_a06_dynamic_collator_and_trace(bundle):
    tok = CharacterTokenizer()
    features = [
        {
            "input_ids": [1, 2],
            "labels": [-100, 2],
            "attention_mask": [1, 1],
            "trace": {"source": "abc"},
        },
        {
            "input_ids": [1, 19, 2, 2, 2],
            "labels": [-100, 19, 2, 2, 2],
            "attention_mask": [1, 1, 1, 0, 0],
        },
    ]
    before = copy.deepcopy(features)
    batch = DataCollatorForCausalLMWith8xPadding(tok, max_length=7)(features)
    assert features == before
    assert set(batch) == {"input_ids", "labels", "attention_mask"}
    assert batch["labels"].tolist() == [
        [-100, 2] + [-100] * 5,
        [-100, 19, 2] + [-100] * 4,
    ]
    assert batch["input_ids"].shape == (2, 7)
    assert batch.trace == ({"source": "abc"},)
    assert DataCollatorForCausalLMWith8xPadding(tok, max_length=20)(features)[
        "labels"
    ].shape == (2, 8)
    with pytest.raises(ValueError, match="exceeds max_length"):
        DataCollatorForCausalLMWith8xPadding(tok, max_length=4)(features)
    cfg = runtime(bundle, num_workers=1)
    loaded = next(iter(build_data_loader(cfg, load_training_dataset(cfg, tok), tok)))
    assert len(loaded.trace) == loaded["labels"].shape[0]
    tok.padding_side = "left"
    with pytest.raises(ValueError, match="right padding"):
        DataCollatorForCausalLMWith8xPadding(tok)


def test_a07_reject_overlong_and_unprovable_alignment():
    class MergingTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            return super().__call__(text.replace("\nsame", "Xsame"), **kwargs)

    class TrimmingTokenizer(CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return super().apply_chat_template(
                [{**m, "content": m["content"].strip()} for m in messages], **kwargs
            )

    row = golden_dialog("CS_AGENT")
    for tok, limit, reason in [
        (CharacterTokenizer(), 10, "no target truncated"),
        (MergingTokenizer(), 300, "cross-boundary merge"),
        (TrimmingTokenizer(), 300, "transforms content"),
    ]:
        template, policy = resolve_policy(
            tok, {"stop_tokens": [STOP], "system_prompt": "应用说明"}
        )
        with pytest.raises(ValueError, match=reason):
            encode_sft(
                tok,
                row,
                template,
                policy,
                {"max_length": limit, "supervise_target_eos": False},
            )


def test_a08_qwen2_nll_gradient_and_step_equivalence():
    from accelerate import Accelerator
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(37)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            attention_dropout=0.0,
            use_cache=False,
        )
    ).eval()
    reference = copy.deepcopy(model)
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1)
    recorded = []

    class CaptureSGD(torch.optim.SGD):
        def step(self, *args, **kwargs):
            recorded[:] = [p.grad.detach().clone() for p in model.parameters()]
            return super().step(*args, **kwargs)

    optimizer = CaptureSGD(model.parameters(), lr=0.05)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    collate = DataCollatorForCausalLMWith8xPadding(CharacterTokenizer(), max_length=13)
    features = []
    for i, n in enumerate((5, 11, 8, 12, 6)):
        tokens = list(range(3, n + 3))
        labels = [
            token if j and ((j + i) % (i + 2) != 0) else -100
            for j, token in enumerate(tokens)
        ]
        features.append(
            {"input_ids": tokens, "attention_mask": [1] * n, "labels": labels}
        )
    micros = [collate([f]) for f in features]
    for i, window in enumerate(accumulation_windows(micros, 3)):
        assert len(window) == (3, 2)[i]
        combined = collate(features[i * 3 : i * 3 + len(window)])
        result = reference(**combined)
        independent = torch.nn.functional.cross_entropy(
            result.logits[:, :-1].flatten(0, 1).float(),
            combined["labels"][:, 1:].reshape(-1),
        )
        torch.testing.assert_close(result.loss, independent)
        independent.backward()
        loss, targets = update_window(model, optimizer, scheduler, accelerator, window)
        assert targets == int(combined["labels"][:, 1:].ne(-100).sum())
        assert loss == pytest.approx(independent.item(), abs=1e-6)
        for got, p in zip(recorded, reference.parameters()):
            torch.testing.assert_close(got, p.grad, atol=2e-7, rtol=2e-5)
        ref_optimizer.step()
        ref_optimizer.zero_grad(set_to_none=True)
        for got, want in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(got, want, atol=2e-7, rtol=2e-5)
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(ValueError, match="no next-token targets"):
        update_window(
            model,
            optimizer,
            scheduler,
            accelerator,
            [{"labels": torch.tensor([[7, -100]])}],
        )
    for got, want in zip(model.parameters(), before):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert scheduler.last_epoch == 2


def test_a09_sampler_boundaries_and_mid_epoch_resume():
    data = [{"input_ids": [1] * size} for size in range(1, 12)]

    def sampler(seed=42):
        return SortedBatchSampler(data, 3, seed=seed)

    a = sampler()
    continuous = list(a)
    assert continuous == list(sampler()) != list(sampler(43))
    assert sorted(map(len, continuous)) == [2, 3, 3, 3]
    assert all(max(b) - min(b) == len(b) - 1 for b in continuous)
    partial = sampler()
    it = iter(partial)
    assert next(it) == continuous[0]
    state = json.loads(json.dumps(partial.state_dict()))
    assert list(it) == continuous[1:]
    restored = sampler()
    restored.load_state_dict(state)
    assert list(restored) == continuous[1:]
    assert list(restored) == list(a)
    with pytest.raises(ValueError, match="incompatible"):
        sampler(1).load_state_dict(state)


def test_a09_budget_repeats_shares_and_exposure(bundle):
    tok = CharacterTokenizer()
    cfg = runtime(
        bundle,
        token_budget=310,
        target_loss_token_share={"chat": 0.3, "notes": 0.7},
        max_sample_repeats=2,
        max_effective_epochs=2,
    )
    dataset = load_training_dataset(cfg, tok)
    loader = build_data_loader(cfg, dataset, tok)
    schedule = compute_schedule(cfg, loader)
    plan = projected_exposure(loader, cfg, schedule)
    assert schedule["planned_epochs"] == 1
    traces = [t for batch in loader for t in batch.trace]
    assert (
        plan["loss_tokens"]
        == sum(t["loss_tokens"] for t in traces)
        <= cfg["token_budget"]
    )
    assert (
        plan["unused_budget_loss_tokens"] == cfg["token_budget"] - plan["loss_tokens"]
    )
    assert plan["sources"]["notes"]["planned_share"] == 0.7
    for v in plan["sources"].values():
        assert v["effective_epochs"] <= 2
        assert v["sample_exposures"] <= v["samples"] * 2
    assert sum(v["actual_share"] for v in plan["sources"].values()) == pytest.approx(
        1.0
    )
    assert "max_sample_repeats" in plan["sampling_exclusions"]
    assert plan["stop_reason"] == "no_eligible_complete_sample"
    assert cfg["max_steps"] > schedule["planned_training_steps"]
    resume = build_data_loader(cfg, dataset, tok).batch_sampler
    start = iter(resume)
    next(start)
    state = resume.state_dict()
    remaining = list(start)
    mirror = build_data_loader(cfg, dataset, tok).batch_sampler
    mirror.load_state_dict(state)
    assert list(mirror) == remaining
    cfg["token_budget"] = 1
    with pytest.raises(ValueError, match="no complete sample"):
        build_data_loader(cfg, load_training_dataset(cfg, tok), tok)


def test_a10_dry_run_preflight_dispatch_and_no_weights(bundle, tmp_path):
    root = tmp_path / "assets"
    for p in ("config", "data/train", "assets"):
        (root / p).mkdir(parents=True)
    shutil.copytree(bundle.parent, root / "data/train/fixture")
    cfg = HERE / "train_cpt.yaml"
    (root / "config/train.yaml").write_text(
        cfg.read_text()
        .replace("task_a_fixture/manifest.json", "data/train/fixture/manifest.json")
        .replace("path: base", "path: assets/base")
    )
    shutil.copyfile(
        HERE.parent / "industrial_support/model.yaml", root / "config/model.yaml"
    )
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "lm_trainer",
        "--asset-root",
        str(root),
        "--backend",
        "unsloth",
    ]
    env = dict(os.environ, LM_TRAINER_UNSLOTH_PYTHON="/definitely-no-python")
    dry = subprocess.run(
        [*cmd, "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    assert json.loads(dry.stdout)["objective"] == "causal_text"
    result = subprocess.run(
        [*cmd, "--validate-release"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    assert json.loads(result.stdout)["fixture_only"] is True
    assert str(tmp_path) not in result.stdout and not (root / "runs").exists()
    bad = root / "data/train/fixture/notes.jsonl"
    with bad.open("a") as stream:
        stream.write("\n")
    result = subprocess.run(
        [*cmd, "--validate-release"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    assert result.returncode == 2
    assert json.loads(result.stderr)["reason"] == "SHA-256 mismatch"
    with pytest.raises(DataContractError, match="SHA-256"):
        train(runtime(root / "data/train/fixture/manifest.json", backend="unsloth"))


def test_a04_native_generation_mask_disambiguates_selected_history():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast

    vocab = {s: i for i, s in enumerate([START, STOP, *sorted(ByteLevel.alphabet())])}
    raw = Tokenizer(BPE(vocab, merges=[]))
    raw.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        eos_token=STOP,
        pad_token=STOP,
        additional_special_tokens=[START],
    )
    tok.chat_template = (
        "{% for m in messages %}{{ '<|start|>' ~ m.role ~ '\\n' }}"
        "{% if m.role == 'assistant' %}{% generation %}{{ m.content }}{% endgeneration %}"
        "{% else %}{{ m.content }}{% endif %}{{ '<eos>\\n' }}{% endfor %}"
    )
    row = {
        "messages": golden_dialog("CS_AGENT")["messages"][1:4],
        "target_mask": [False, False, True],
    }
    template, policy = resolve_policy(tok, {"stop_tokens": [STOP]})
    native = tok.apply_chat_template(
        [{k: m[k] for k in ("role", "content")} for m in row["messages"]],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    result = encode_sft(
        tok, row, template, policy, {"max_length": 200, "supervise_target_eos": True}
    )
    assert sum(native["assistant_masks"]) > result.body_targets == len("same")
    target_stop = max(
        i for i, role in enumerate(result.roles) if role == "assistant_stop"
    )
    assert result.labels == [
        token if role == "assistant_target" or i == target_stop else -100
        for i, (token, role) in enumerate(zip(result.input_ids, result.roles))
    ]


def test_a01_actual_fixture_artifact_markers_and_consumed_sampling(
    bundle, tmp_path, monkeypatch
):
    from peft import get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from lm_trainer.train import metrics, modeling

    base, output = tmp_path / "base", tmp_path / "run"
    base.mkdir()
    config = Qwen2Config(
        vocab_size=7,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        pad_token_id=2,
        eos_token_id=2,
        max_position_embeddings=300,
        use_cache=False,
    )
    config.save_pretrained(base)

    class TinyTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            return {
                "input_ids": [
                    (i % 4) + 3 if i > 2 else i
                    for i in super().__call__(text, **kwargs)["input_ids"]
                ]
            }

        def save_pretrained(self, path):
            (path / "tokenizer_config.json").write_text('{"fixture": true}')

    def components(cfg):
        model = Qwen2ForCausalLM(config)
        return get_peft_model(model, modeling.create_lora_config(cfg)), TinyTokenizer()

    monkeypatch.setattr(modeling, "load_training_components", components)
    monkeypatch.setattr(metrics, "plot_metrics", None)
    cfg = runtime(
        bundle,
        model_name_or_path=str(base),
        checkpoint_dir=str(output),
        max_steps=1,
        save_steps=1,
        lora_r=2,
        lora_alpha=4,
        lora_target_modules=["q_proj"],
        lora_dropout=0.0,
        lora_bias="none",
        backend="hf",
        compute_dtype="no",
        warmup_steps=0,
        quantization_mode="none",
        lr=0.001,
    )
    [update] = train(cfg)
    for checkpoint in ("step_1", "final"):
        manifest = json.loads(
            (output / checkpoint / "training_manifest.json").read_text()
        )
        assert (
            manifest["fixture_only"] is True and manifest["objective"] == "causal_text"
        )
    perf = json.loads((output / "performance.json").read_text())
    assert (
        perf["fixture_only"] is True
        and perf["actual_sampling"]["loss_tokens"] == update["loss_tokens"]
    )
    assert perf["planned_sampling"] == perf["actual_sampling"]
    assert perf["actual_sampling"]["run_stop_reason"] == "max_steps"
    assert "demo_msg_" not in json.dumps(perf)
    with pytest.raises(FileExistsError):
        train(cfg)


def test_a04_customer_view_is_validated_and_uses_swapped_roles(bundle):
    manifest = json.loads(bundle.read_text())
    manifest["target_person"] = "CUSTOMER"
    manifest["files"] = [spec for spec in manifest["files"] if spec["kind"] == "sft"]
    bundle.write_text(json.dumps(manifest))

    def change_perspective(rows):
        row = rows[0]
        row["target_person"] = "CUSTOMER"
        row["messages"] = row["messages"][:6]
        for m in row["messages"]:
            m["role"] = "assistant" if m["speaker"] == "CUSTOMER" else "user"
        row["target_mask"] = [m["role"] == "assistant" for m in row["messages"]]

    rewrite(bundle, "dyadic.jsonl", change_perspective)
    rows = [
        json.loads(line)
        for line in (bundle.parent / "dyadic.jsonl").read_text().splitlines()
    ]
    row = rows[0]
    rewrite(
        bundle,
        "groups.jsonl",
        lambda entries: entries[-1].update(
            target_person="CUSTOMER",
            source_ids=[sid for m in row["messages"] for sid in m["source_ids"]],
        ),
    )
    manifest = json.loads(bundle.read_text())
    del manifest["source_index"]
    bundle.write_text(json.dumps(manifest))
    [sample] = load_training_dataset(runtime(bundle, kind="sft"), CharacterTokenizer())
    assert sample["trace"]["body_tokens"] == sum(
        len(m["content"]) for m in row["messages"] if m["role"] == "assistant"
    )
