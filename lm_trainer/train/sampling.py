"""Deterministic batch-boundary sampling and complete-sample token budgeting."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
from collections import defaultdict, deque
from itertools import islice

from torch.utils.data import Sampler


def _batches(indices, size):
    return [list(indices[i : i + size]) for i in range(0, len(indices), size)]


class SortedBatchSampler(Sampler):
    """Yield intact batches; cursor is the next dispatched batch in this epoch.

    One live iterator at a time. At the yielded end of an epoch the cursor
    remains len(self) until iteration is exhausted; the next epoch starts at 0.
    """

    def __init__(self, dataset, batch_size, shuffle=True, seed=42, *, ordered=None):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size, self.shuffle, self.seed = batch_size, shuffle, seed
        self.iteration = self.cursor = 0
        self._active = False
        lengths = [len(item["input_ids"]) for item in dataset]
        indices = (
            sorted(range(len(dataset)), key=lambda i: -lengths[i])
            if ordered is None
            else ordered
        )
        self._batches = _batches(indices, batch_size)
        self._shuffle_batches = shuffle and ordered is None
        fingerprint = hashlib.sha256()
        for item in dataset:
            fingerprint.update(
                json.dumps(
                    (item["input_ids"], item.get("labels"), item.get("trace")),
                    sort_keys=True,
                ).encode()
            )
        self._key = hashlib.sha256(
            json.dumps(
                [
                    fingerprint.hexdigest(),
                    batch_size,
                    shuffle,
                    seed,
                    self._batches,
                ]
            ).encode()
        ).hexdigest()

    def batches_at(self, epoch):
        batches = [batch.copy() for batch in self._batches]
        if self._shuffle_batches:
            random.Random(self.seed + epoch).shuffle(batches)
        return batches

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        if self._active:
            raise RuntimeError("Only one live sampler iterator is supported")
        self._active = True
        try:
            batches = self.batches_at(self.iteration)
            while self.cursor < len(batches):
                batch = batches[self.cursor]
                self.cursor += 1
                yield batch
            self.iteration += 1
            self.cursor = 0
        finally:
            self._active = False

    def state_dict(self):
        return {
            "version": 1,
            "plan_key": self._key,
            "epoch": self.iteration,
            "next_batch": self.cursor,
        }

    def load_state_dict(self, state):
        if self._active:
            raise RuntimeError("Cannot restore a sampler with a live iterator")
        if (
            not isinstance(state, dict)
            or type(state.get("version")) is not int
            or state["version"] != 1
            or state.get("plan_key") != self._key
            or type(state.get("epoch")) is not int
            or state["epoch"] < 0
            or type(state.get("next_batch")) is not int
            or not 0 <= state["next_batch"] <= len(self)
        ):
            raise ValueError("Invalid or incompatible sampler state")
        self.iteration, self.cursor = state["epoch"], state["next_batch"]


def sampling_plan(dataset, runtime):
    """Return an optional fixed order for a one-pass, stage-wide sampling plan.

    Choose the source with the earliest virtual token finish among eligible
    samples. Cost is the shifted supervised token count, not examples or bytes.
    Each source cycles through its shuffled queue; no sample is ever sliced.
    """
    plan = dataset.plan
    sources = plan["sources"]
    unique = {s: dataset.statistics[s]["unique_loss_tokens"] for s in sources}
    mix = (
        runtime.get("structured_sft_loss_token_share")
        or plan["target_loss_token_share"]
    )
    if mix != "natural" and set(mix) != set(sources):
        raise ValueError(
            "Sampling share keys must match all selected, approved sources"
        )
    shares = (
        {s: n / sum(unique.values()) for s, n in unique.items()}
        if mix == "natural"
        else mix
    )
    budget = plan["token_budget"]
    repeat_cap, epoch_cap = plan["max_sample_repeats"], plan["max_effective_epochs"]
    if mix == "natural" and all(
        value is None for value in (budget, repeat_cap, epoch_cap)
    ):
        return None, {
            "target_loss_token_share": shares,
            "stop_reason": "natural_epochs",
            "sampling_exclusions": {},
        }

    epochs = runtime["epochs"]
    repeat_cap = epochs if repeat_cap is None else repeat_cap
    epoch_cap = epochs if epoch_cap is None else epoch_cap
    # Without an explicit budget, keep the nominal stage token work bounded
    # even when the user allows oversampling an individual source.
    capacity = budget if budget is not None else epochs * sum(unique.values())
    remaining = capacity
    queues = {s: deque() for s in sources}
    used, repeats, excluded = defaultdict(int), defaultdict(int), defaultdict(int)
    source_exclusions = {s: defaultdict(int) for s in sources}
    costs = [item["trace"]["loss_tokens"] for item in dataset]
    for i, item in enumerate(dataset):
        queues[item["trace"]["source"]].append(i)
    if runtime["shuffle"]:
        rng = random.Random(plan["seed"])
        for source, queue in queues.items():
            shuffled = list(queue)
            rng.shuffle(shuffled)
            queues[source] = deque(shuffled)
    # Unreachable thresholds need not convert huge user caps through float floor.
    caps = {
        s: min(capacity, math.floor(min(capacity, epoch_cap * count)))
        for s, count in unique.items()
    }
    heap = []

    def enqueue(source, order):
        queue = queues[source]
        while queue:
            i = queue[0]
            if repeats[i] >= repeat_cap:
                reason = "max_sample_repeats"
            elif costs[i] > remaining:
                reason = "insufficient_budget"
            elif costs[i] + used[source] > caps[source]:
                reason = "max_effective_epochs"
            else:
                heapq.heappush(
                    heap, ((used[source] + costs[i]) / shares[source], order, source)
                )
                break
            excluded[reason] += 1
            source_exclusions[source][reason] += 1
            queue.popleft()

    for order, source in enumerate(sources):
        enqueue(source, order)
    selected = []
    while heap and remaining:
        _, order, source = heapq.heappop(heap)
        i = queues[source][0]
        if costs[i] <= remaining:
            queues[source].rotate(-1)
            selected.append(i)
            remaining -= costs[i]
            used[source] += costs[i]
            repeats[i] += 1
        enqueue(source, order)
    if not selected:
        raise ValueError(
            "Sampling plan has no complete sample within the token budget/exposure caps"
        )
    return selected, {
        "target_loss_token_share": shares,
        "stop_reason": "token_budget_reached"
        if not remaining
        else "no_eligible_complete_sample",
        "sampling_exclusions": dict(excluded),
        **(
            {
                "sampling_exclusions_by_source": {
                    s: dict(c) for s, c in source_exclusions.items()
                }
            }
            if runtime.get("init_adapter")
            else {}
        ),
        "capacity_loss_tokens": capacity,
        "capacity_unused_loss_tokens": remaining,
        "max_sample_repeats": repeat_cap,
        "max_effective_epochs": epoch_cap,
    }


def _trace_key(source, identity):
    return hashlib.sha256(
        json.dumps([source, identity], ensure_ascii=False).encode()
    ).hexdigest()


class Exposure:
    """Summarize consumed batches without exporting sample or origin IDs."""

    def __init__(self, sampler, *, track_unique=False):
        self.sampler = sampler
        self.track_unique = track_unique
        self.repeats = defaultdict(int)
        self.target_repeats = defaultdict(int)
        self.events = defaultdict(set)
        self.tail_windows = 0
        self.last_microbatches = 0
        self.unique = {
            s: dict.fromkeys(
                (
                    "samples",
                    "loss_tokens",
                    "body_tokens",
                    "eos_tokens",
                    "input_tokens",
                    "target_separator_tokens",
                ),
                0,
            )
            for s in sampler.statistics
        }
        self.counts = {
            s: {
                "samples": 0,
                "loss_tokens": 0,
                "body_tokens": 0,
                "eos_tokens": 0,
                "input_tokens": 0,
                "target_separator_tokens": 0,
                "max_sample_repeats_observed": 0,
                "max_target_repeats_observed": 0,
                "truncated": 0,
            }
            for s in sampler.statistics
        }
        self.updates = 0

    def observe(self, traces, *, microbatches=None, accumulation=None):
        self.updates += 1
        if microbatches is not None:
            self.last_microbatches = microbatches
            self.tail_windows += microbatches < accumulation
        for trace in traces:
            counts = self.counts[trace["source"]]
            counts["samples"] += 1
            counts["truncated"] += trace.get("truncated", False)
            first = False
            if self.track_unique:
                source = trace["source"]
                key = _trace_key(source, trace["sample_id"])
                self.repeats[key] += 1
                first = self.repeats[key] == 1
                counts["max_sample_repeats_observed"] = max(
                    counts["max_sample_repeats_observed"], self.repeats[key]
                )
                for sid in set(trace.get("target_source_ids", ())):
                    target = _trace_key(source, sid)
                    self.target_repeats[target] += 1
                    counts["max_target_repeats_observed"] = max(
                        counts["max_target_repeats_observed"],
                        self.target_repeats[target],
                    )
                if trace.get("event_group") is not None:
                    self.events[source].add(_trace_key(source, trace["event_group"]))
                if first:
                    self.unique[source]["samples"] += 1
            for key in (
                "loss_tokens",
                "body_tokens",
                "eos_tokens",
                "input_tokens",
                "target_separator_tokens",
            ):
                counts[key] += trace.get(key, 0)
                if first:
                    self.unique[trace["source"]][key] += trace.get(key, 0)

    def state_dict(self):
        return {
            "counts": self.counts,
            "unique": self.unique,
            "repeats": dict(self.repeats),
            "target_repeats": dict(self.target_repeats),
            "events": {s: sorted(ids) for s, ids in self.events.items()},
            "tail_windows": self.tail_windows,
            "last_microbatches": self.last_microbatches,
            "updates": self.updates,
        }

    def load_state_dict(self, state):
        if set(state["counts"]) != set(self.counts) or set(state["unique"]) != set(
            self.unique
        ):
            raise ValueError("Incompatible exposure counters.")
        self.counts, self.unique = state["counts"], state["unique"]
        self.repeats = defaultdict(int, state["repeats"])
        self.target_repeats = defaultdict(int, state["target_repeats"])
        self.events = defaultdict(
            set, {s: set(ids) for s, ids in state["events"].items()}
        )
        self.tail_windows = state["tail_windows"]
        self.last_microbatches = state["last_microbatches"]
        self.updates = state["updates"]

    def report(self):
        sampler = self.sampler
        total = sum(count["loss_tokens"] for count in self.counts.values())
        budget = sampler.token_budget
        return {
            **sampler.mixture,
            "token_budget": budget,
            "unused_budget_loss_tokens": None if budget is None else budget - total,
            "optimizer_windows": self.updates,
            "partial_accumulation_windows": self.tail_windows,
            "last_window_microbatches": self.last_microbatches,
            "loss_tokens": total,
            "sources": {
                s: {
                    **sampler.statistics[s],
                    "sample_exposures": c["samples"],
                    "sampled_loss_tokens": c["loss_tokens"],
                    "sampled_body_tokens": c["body_tokens"],
                    "sampled_eos_tokens": c["eos_tokens"],
                    "sampled_input_tokens": c["input_tokens"],
                    "sampled_target_separator_tokens": c["target_separator_tokens"],
                    "requested_loss_tokens": (budget if budget is not None else total)
                    * sampler.mixture["target_loss_token_share"][s],
                    "share_error": abs(
                        (c["loss_tokens"] / total if total else 0)
                        - sampler.mixture["target_loss_token_share"][s]
                    ),
                    **(
                        {
                            **{
                                f"consumed_unique_{k}": v
                                for k, v in self.unique[s].items()
                            },
                            "sampled_truncated": c["truncated"],
                            "consumed_independent_event_groups": len(
                                self.events.get(s, ())
                            ),
                            "repeat_sample_exposures": c["samples"]
                            - self.unique[s]["samples"],
                            "max_sample_repeats_observed": c[
                                "max_sample_repeats_observed"
                            ],
                            "max_target_repeats_observed": c[
                                "max_target_repeats_observed"
                            ],
                            "unconsumed_unique_samples": sampler.statistics[s][
                                "samples"
                            ]
                            - self.unique[s]["samples"],
                            "sampling_exclusions": sampler.mixture.get(
                                "sampling_exclusions_by_source", {}
                            ).get(s, {}),
                        }
                        if self.track_unique
                        else {}
                    ),
                    "effective_epochs": c["loss_tokens"]
                    / sampler.statistics[s]["unique_loss_tokens"],
                    "planned_share": sampler.mixture["target_loss_token_share"][s],
                    "actual_share": c["loss_tokens"] / total if total else 0,
                }
                for s, c in self.counts.items()
            },
        }


def projected_exposure(loader, runtime, schedule):
    sampler = loader.batch_sampler
    report = Exposure(sampler, track_unique=bool(runtime.get("init_adapter")))
    for epoch in range(schedule["planned_epochs"]):
        batches = iter(sampler.batches_at(epoch))
        while report.updates < schedule["planned_training_steps"]:
            window = list(islice(batches, runtime["gradient_accumulation_steps"]))
            if not window:
                break
            report.observe(
                (loader.dataset[i]["trace"] for indices in window for i in indices),
                microbatches=len(window),
                accumulation=runtime["gradient_accumulation_steps"],
            )
        if report.updates >= schedule["planned_training_steps"]:
            break
    limited = schedule["planned_training_steps"] < schedule["total_possible_steps"]
    aggregate = report.report()
    if (
        runtime.get("require_exact_token_budget")
        and aggregate["unused_budget_loss_tokens"] != 0
    ):
        raise ValueError(
            f"Exact token budget not met: {aggregate['loss_tokens']}/{sampler.token_budget}; "
            "adjust budget, repeat/exposure caps, or max_steps."
        )
    tolerance = runtime.get("sampling_max_share_error")
    if tolerance is not None:
        errors = {
            s: row["share_error"]
            for s, row in aggregate["sources"].items()
            if row["share_error"] > tolerance + 1e-12
        }
        if errors:
            raise ValueError(
                f"Sampling share error exceeds tolerance: {errors}; adjust budget or mixture."
            )
    return {
        **aggregate,
        "run_stop_reason": "max_steps"
        if limited
        else (
            sampler.mixture["stop_reason"]
            if runtime.get("init_adapter")
            else "sampling_plan_complete"
        ),
    }
