from __future__ import annotations

import math
import random

from datasets import load_dataset
from torch.utils.data import DataLoader, Sampler

from lm_trainer.model.loader import load_tokenizer


class DataCollatorForCausalLMWith8xPadding:
    def __init__(
        self,
        tokenizer,
        max_length=None,
        pad_to_max_length=False,
        label_pad_token_id=-100,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        batch_max_len = (
            self.max_length
            if self.pad_to_max_length
            else max(len(f["input_ids"]) for f in features)
        )
        padded_len = int(math.ceil(batch_max_len / 8) * 8)
        batch = self.tokenizer.pad(
            features,
            padding="max_length",
            max_length=padded_len,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = self.label_pad_token_id
        batch["labels"] = labels
        return batch


class SortedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.iteration = 0
        self.lengths = [len(item["input_ids"]) for item in dataset]

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices.sort(key=lambda idx: self.lengths[idx], reverse=True)
        if self.shuffle:
            batch_indices = [
                indices[i : i + self.batch_size]
                for i in range(0, len(indices), self.batch_size)
            ]
            rng = random.Random(self.seed + self.iteration)
            rng.shuffle(batch_indices)
            indices = [idx for batch in batch_indices for idx in batch]
        self.iteration += 1
        return iter(indices)

    def __len__(self):
        return len(self.dataset)


def load_training_dataset(runtime: dict, tokenizer):
    dataset = load_dataset("json", data_files={"train": runtime["data_file_path"]})[
        "train"
    ]
    if "text" not in dataset.column_names:
        raise ValueError(f"训练数据缺少 text 字段: {dataset.column_names}")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=runtime["max_length"],
            return_attention_mask=True,
        )

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


def build_data_loader(runtime: dict, dataset, tokenizer):
    data_collator = DataCollatorForCausalLMWith8xPadding(
        tokenizer=tokenizer,
        max_length=runtime["max_length"],
        pad_to_max_length=not runtime["auto_pad_batch"],
        label_pad_token_id=-100,
    )
    sampler = SortedBatchSampler(
        dataset=dataset,
        batch_size=runtime["batch_size"],
        shuffle=runtime["shuffle"],
        seed=runtime["seed"],
    )
    return DataLoader(
        dataset,
        batch_size=runtime["batch_size"],
        sampler=sampler,
        collate_fn=data_collator,
        num_workers=runtime["num_workers"],
        pin_memory=runtime["pin_memory"],
    )
