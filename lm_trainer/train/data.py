from __future__ import annotations

import math
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from lm_trainer.template_policy import public_policy, resolve_policy
from lm_trainer.train.encoding import IGNORE, EncodingError, encode_sft, encode_text
from lm_trainer.train.sampling import SortedBatchSampler, sampling_plan
from lm_trainer.train.schema import DataContractError, preflight_data


class CausalBatch(dict):
    """Only forward tensors are keys; trace survives worker IPC and pinning."""

    def __init__(self, tensors, trace):
        super().__init__(tensors)
        self.trace = trace


class DataCollatorForCausalLMWith8xPadding:
    def __init__(
        self,
        tokenizer,
        max_length=None,
        pad_to_max_length=False,
        label_pad_token_id=IGNORE,
    ):
        if tokenizer.padding_side != "right":
            raise ValueError("Causal collator supports right padding only")
        if pad_to_max_length and max_length is None:
            raise ValueError("Fixed padding requires max_length")
        if label_pad_token_id != IGNORE:
            raise ValueError("Causal labels require ignore_index=-100")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length

    def __call__(self, features):
        if not features:
            raise ValueError("Cannot collate an empty batch")
        longest = max(len(f["input_ids"]) for f in features)
        if not longest or (self.max_length is not None and longest > self.max_length):
            raise ValueError("Empty sequence or sequence exceeds max_length")
        if self.pad_to_max_length:
            padded_len = self.max_length
        else:
            padded_len = (longest + 7) // 8 * 8
            if self.max_length is not None:
                padded_len = min(padded_len, self.max_length)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            raise ValueError("Tokenizer must define a pad_token_id")
        inputs, masks, labels, trace = [], [], [], []
        for f in features:
            ids, mask, lab = f["input_ids"], f["attention_mask"], f["labels"]
            if (
                not len(ids)
                or not (len(ids) == len(mask) == len(lab))
                or any(v not in (0, 1) for v in mask)
            ):
                raise ValueError(
                    "Features require aligned input_ids, attention_mask and preconstructed labels"
                )
            amount = padded_len - len(ids)
            inputs.append([*ids, *([pad] * amount)])
            masks.append([*mask, *([0] * amount)])
            labels.append(
                [v if active else IGNORE for v, active in zip(lab, mask)]
                + [IGNORE] * amount
            )
            if "trace" in f:
                trace.append(f["trace"])
        return CausalBatch(
            {
                key: torch.tensor(rows, dtype=torch.long)
                for key, rows in (
                    ("input_ids", inputs),
                    ("attention_mask", masks),
                    ("labels", labels),
                )
            },
            tuple(trace),
        )


class EncodedDataset(list):
    def __init__(self, plan):
        super().__init__()
        self.plan = plan
        self.statistics = {}
        self.template_policy = None


def load_training_dataset(runtime, tokenizer, *, validated=None):
    validated = validated if validated is not None else preflight_data(runtime)
    plan = validated.plan
    dataset = EncodedDataset(plan)
    is_sft = plan["objective"] == "assistant_only_messages"
    if is_sft:
        try:
            template, policy = resolve_policy(tokenizer, plan["template_policy"])
        except ValueError as exc:
            raise DataContractError(str(exc), path="template_policy") from None
        dataset.template_policy = public_policy(policy)
    lengths = defaultdict(list)
    for sample in validated.samples:
        row, source = sample.row, sample.source
        try:
            result = (
                encode_sft(tokenizer, row, template, policy, plan)
                if is_sft
                else encode_text(tokenizer, row["text"], plan)
            )
        except EncodingError as exc:
            sample.fail(str(exc))
        except (ValueError, TypeError, RuntimeError):
            sample.fail("tokenizer failed to encode the sample")
        source_ids = (
            tuple(
                dict.fromkeys(sid for m in row["messages"] for sid in m["source_ids"])
            )
            if is_sft
            else tuple(row.get("source_ids", ()))
        )
        trace = {
            "sample_id": row.get("id"),
            "source": source,
            "source_ids": source_ids,
            "event_group": row.get("event_group"),
            "source_index_ref": row.get("source_index_ref"),
            "body_tokens": result.body_targets,
            "eos_tokens": result.eos_targets,
            "loss_tokens": result.body_targets + result.eos_targets,
            "input_tokens": len(result.input_ids),
            "truncated": result.truncated,
        }
        dataset.append(
            {
                "input_ids": result.input_ids,
                "attention_mask": [1] * len(result.input_ids),
                "labels": result.labels,
                "trace": trace,
            }
        )
        lengths[source].append(trace["input_tokens"])
        stats = dataset.statistics.setdefault(
            source,
            {
                **plan["statistics"][source],
                "unique_body_tokens": 0,
                "unique_eos_tokens": 0,
                "unique_loss_tokens": 0,
                "unique_input_tokens": 0,
                "rejected": 0,
                "truncated": 0,
                "truncation_reasons": {},
                "eos_only_samples": 0,
            },
        )
        for key in ("body_tokens", "eos_tokens", "loss_tokens", "input_tokens"):
            stats[f"unique_{key}"] += trace[key]
        if result.truncated:
            stats["truncated"] += 1
            stats["truncation_reasons"]["legacy_max_length"] = stats["truncated"]
        stats["eos_only_samples"] += result.body_targets == 0
    for source, stats in dataset.statistics.items():
        seq = sorted(lengths[source])
        stats["length_quantiles"] = {
            **{f"p{p}": seq[math.ceil(p * len(seq) / 100) - 1] for p in (50, 90, 99)},
            "max": seq[-1],
        }
        stats["eos_only_sample_share"] = stats["eos_only_samples"] / stats["samples"]
        stats["eos_target_share"] = (
            stats["unique_eos_tokens"] / stats["unique_loss_tokens"]
        )
    return dataset


def build_data_loader(runtime: dict, dataset, tokenizer):
    data_collator = DataCollatorForCausalLMWith8xPadding(
        tokenizer=tokenizer,
        max_length=runtime["max_length"],
        pad_to_max_length=not runtime["auto_pad_batch"],
    )
    try:
        ordered, mixture = sampling_plan(dataset, runtime)
    except ValueError as exc:
        raise DataContractError(str(exc), path="sampling_plan") from None
    sampler = SortedBatchSampler(
        dataset,
        runtime["batch_size"],
        runtime["shuffle"],
        runtime["seed"],
        ordered=ordered,
    )
    sampler.planned_epochs = 1 if ordered is not None else runtime["epochs"]
    sampler.statistics = dataset.statistics
    sampler.mixture = mixture
    sampler.token_budget = dataset.plan["token_budget"]
    generator = None
    if runtime.get("init_adapter"):
        # Iterator/worker seeds must not consume the training model RNG, even
        # when resume creates an extra iterator in the middle of an epoch.
        generator = torch.Generator().manual_seed(runtime["seed"])
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=data_collator,
        num_workers=runtime["num_workers"],
        pin_memory=runtime["pin_memory"],
        generator=generator,
    )
