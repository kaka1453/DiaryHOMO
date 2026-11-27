# Task B consumer fixtures — NOT for production

Both adapters were trained on generated, fictional vision-support examples with a randomly initialized tiny Qwen2. See [`index.json`](index.json) for the pinned manifest digests and base fingerprint. `sft/` inherits `cpt/`. The adapters are not standalone language models and their outputs have **no customer-support validity**.

From the repository root, use the pinned HF environment:

```sh
mkdir -p .cache
work=$(mktemp -d "$PWD/.cache/task-b-consumer.XXXXXX")
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 uv run --no-sync -m examples.vision_support.task_b_workload --output "$work/prepared"
```

Use `prepared/base` with `sft/` after verifying the base fingerprint and the manifest via [`TASK_B_CONTRACT.md`](../../../lm_trainer/TASK_B_CONTRACT.md). The tiny base is generated, not downloaded. `allow_fixture=True` is necessary and must only be used in fixture consumers.
