"""Compatibility interface for lm_trainer.train.data."""

from lm_trainer.train.data import (
    DataCollatorForCausalLMWith8xPadding,
    SortedBatchSampler,
    load_training_dataset,
    build_data_loader,
)

import json
from diary_core.config.train_config import (
    build_train_parser,
    build_train_runtime_config,
)
from lm_trainer.model.loader import load_tokenizer


def main() -> None:
    parser = build_train_parser()
    args = parser.parse_args()
    runtime = build_train_runtime_config(args)
    summary = {
        "data_file_path": runtime["data_file_path"],
        "batch_size": runtime["batch_size"],
        "max_length": runtime["max_length"],
        "auto_pad_batch": runtime["auto_pad_batch"],
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    tokenizer = load_tokenizer(runtime["model_name_or_path"], use_fast=False)
    dataset = load_training_dataset(runtime, tokenizer)
    train_loader = build_data_loader(runtime, dataset, tokenizer)
    first_batch = next(iter(train_loader))
    summary.update(
        {
            "dataset_size": len(dataset),
            "batch_input_shape": list(first_batch["input_ids"].shape),
            "batch_labels_shape": list(first_batch["labels"].shape),
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
