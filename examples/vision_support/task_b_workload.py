"""Deterministic, entirely fictional Task B workload. No corpus files are read."""

from __future__ import annotations

import random
from pathlib import Path

from examples.vision_support.make_fixture import write_jsonl
from lm_trainer.train.performance import write_json
from lm_trainer.train.schema import CORPUS_SCHEMA, GROUP_SCHEMA, RELEASE_SCHEMA

STOP = "<|im_end|>"
START = "<|im_start|>"
POLICY = {"stop_tokens": [STOP]}
TEMPLATE = (
    "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '<|im_start|>' + "
    "m['content'] + '<|im_end|>' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant<|im_start|>' }}{% endif %}"
)
TOPICS = (
    ("图像发暗", "检查光源亮度，再记录曝光时间。"),
    ("触发漏拍", "对比触发计数和存图数量。"),
    ("边缘模糊", "使用虚构标准样件确认焦距。"),
    ("工位误报", "检查工位照明角度和阈值。"),
    ("镜头反光", "调整照明角度并固定样件。"),
    ("图像偏移", "先复查安装位置和标定结果。"),
)


def tiny_base(destination, *, seed=805):
    """Random local Qwen2 and a deterministic byte-level native chat tokenizer."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    destination.mkdir(parents=True, exist_ok=False)
    special = ["<pad>", "<unk>", STOP, START]
    vocab = {s: i for i, s in enumerate([*special, *sorted(ByteLevel.alphabet())])}
    raw = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    raw.pre_tokenizer = ByteLevel(add_prefix_space=False)
    raw.decoder = ByteDecoder()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token=STOP,
        additional_special_tokens=[START],
        chat_template=TEMPLATE,
    )
    tokenizer.save_pretrained(destination)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=len(vocab),
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=2048,
                bos_token_id=None,
                eos_token_id=2,
                pad_token_id=0,
                tie_word_embeddings=False,
            )
        )
    model.save_pretrained(destination)
    return tokenizer


def _fit(render, count, text, supplement, limit):
    """Find a close, complete content prefix measured by the actual tokenizer."""
    low, high = 0, limit
    best = render(text)
    if count(best) > limit:
        raise ValueError("Workload minimum dialogue exceeds requested context.")
    while low <= high:
        mid = (low + high) // 2
        candidate = render(text + supplement * mid)
        if count(candidate) <= limit:
            best, low = candidate, mid + 1
        else:
            high = mid - 1
    return best


def workload(destination, tokenizer, *, split="train", length=512, count=32, seed=71):
    if split not in ("train", "dev") or count < 1 or length < 128:
        raise ValueError("Workload requires train/dev, count >= 1, context >= 128.")
    destination.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed + (1 if split == "dev" else 0))
    files, ledger = [], []
    for source, kind in (("chat", "cpt"), ("notes", "cpt"), ("dyadic_chat", "sft")):
        rows = []
        for i in range(count):
            marker = f"fictional-{split}-{source}-{i}"
            issue, answer = rng.choice(TOPICS)
            common = {
                "schema": CORPUS_SCHEMA,
                "id": marker,
                "kind": kind,
                "target_person": "CS_AGENT",
                "event_group": marker,
                "source_index_ref": f"synthetic:{marker}",
                "split": split,
                "permissions": {"local_review": True, "train": True, "external": False},
                "review_status": "approved",
                "fixture_only": True,
            }
            if kind == "cpt":
                ids = [f"{marker}-0"]
                if source == "chat":
                    text = ("好。", "请复查。", answer, issue + "。" + answer)[i % 4]
                else:
                    text = _fit(
                        lambda x: x,
                        lambda x: (
                            len(tokenizer(x, add_special_tokens=False)["input_ids"]) + 1
                        ),
                        "虚构排查笔记：" + issue + "。",
                        "拍照、核对、记录。",
                        (length, length // 2, length // 3)[i % 3],
                    )
                rows.append({**common, "text": text, "source_ids": ids})
            else:
                ids = [f"{marker}-{turn}" for turn in range(4)]
                mask = [False, i % 2 == 0, False, True]

                def render(text, mask=mask, issue=issue, ids=ids):
                    contents = (
                        text if not mask[1] else issue,
                        "明白。" if not mask[1] else text,
                        "然后呢？",
                        "请复查。",
                    )
                    return [
                        {
                            "role": "user" if j % 2 == 0 else "assistant",
                            "speaker": "CUSTOMER" if j % 2 == 0 else "CS_AGENT",
                            "content": content,
                            "source_ids": [ids[j]],
                        }
                        for j, content in enumerate(contents)
                    ]

                def size(messages):
                    return len(
                        tokenizer.apply_chat_template(
                            [{k: m[k] for k in ("role", "content")} for m in messages],
                            tokenize=True,
                            add_generation_prompt=False,
                            return_dict=True,
                        )["input_ids"]
                    )

                messages = _fit(
                    render,
                    size,
                    issue[:2],
                    "检查光源并记录。",
                    length if i % 3 == 0 else max(128, length // 2),
                )
                rows.append({**common, "messages": messages, "target_mask": mask})
            ledger.append(
                {
                    **{
                        k: common[k]
                        for k in ("id", "kind", "target_person", "event_group", "split")
                    },
                    "schema": GROUP_SCHEMA,
                    "source_ids": ids,
                }
            )
        files.append(
            {
                **write_jsonl(destination / f"{source}.jsonl", rows),
                "source": source,
                "kind": kind,
                "split": split,
                "records": count,
            }
        )
    manifest = destination / "manifest.json"
    write_json(
        manifest,
        {
            "schema": RELEASE_SCHEMA,
            "release_id": f"fictional_task_b_v1_{split}",
            "target_person": "CS_AGENT",
            "parent_version": "fictional_task_b_v1",
            "scope": "fixture",
            "fixture_only": True,
            "training_authorized": False,
            "review_status": "approved",
            "files": files,
            "source_group_manifest": write_jsonl(destination / "groups.jsonl", ledger),
        },
    )
    return manifest


def stage_configs(
    root: Path,
    base: Path,
    train: Path,
    dev: Path | None,
    *,
    length=256,
    steps=4,
    warmup=0,
    bench_warmup=1,
    backend="hf",
    dtype="fp32",
    quantization="none",
    policy=None,
):
    policy = policy or POLICY
    common = {
        "model_name_or_path": str(base),
        "data_mode": "fixture",
        "manifest_path": str(train),
        "backend": backend,
        "compute_dtype": dtype,
        "quantization_mode": quantization,
        "max_length": length,
        "max_steps": steps,
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": "standard",
        "warmup_steps": warmup,
        "benchmark_warmup_steps": bench_warmup,
        "lr": 0.003,
        "seed": 71,
        "save_steps": steps + 1,
        "template_policy": policy,
        "packing": False,
    }
    if dev:
        common["dev_manifest_path"] = str(dev)
    cpt = {
        **common,
        "name": "mixed_cpt",
        "objective": "causal_text",
        "init_adapter": "new",
        "sources": ["chat", "notes"],
        "target_loss_token_share": {"chat": 0.35, "notes": 0.65},
        "output_dir": str(root / "cpt"),
        "token_budget": length * steps * 2,
    }
    sft = {
        **common,
        "name": "dialogue_sft",
        "objective": "assistant_only_messages",
        "init_adapter": "new",
        "sources": ["dyadic_chat"],
        "supervise_target_eos": True,
        "output_dir": str(root / "sft"),
        "token_budget": length * steps * 2,
    }
    return cpt, sft


def main():
    import argparse
    import os
    import shutil

    import yaml

    parser = argparse.ArgumentParser(
        description="Prepare a new offline fictional Task B training example."
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    tokenizer = tiny_base(target / "base")
    workload(target / "data/train", tokenizer, length=256, count=16)
    workload(target / "data/dev", tokenizer, split="dev", length=256, count=3)
    templates = Path(__file__).resolve().parent / "task_b_config"
    for name in ("stages.yaml", "model.yaml"):
        shutil.copyfile(templates / name, target / name)
    cfg = yaml.safe_load((target / "stages.yaml").read_text())
    cpt, sft = cfg.pop("stages")
    write_json(target / "cpt.yaml", {**cfg, **cpt})
    write_json(
        target / "sft_new.yaml",
        {
            **cfg,
            **sft,
            "init_adapter": "new",
            "parent": None,
            "output_dir": "runs/direct_sft_seed72",
        },
    )
    print("Fictional Task B configs and random tiny Qwen2 prepared.")


if __name__ == "__main__":
    main()
