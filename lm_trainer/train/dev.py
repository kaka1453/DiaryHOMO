"""Stage-end dev evaluation (no test entry point), token-weighted by source."""

import math
from collections import defaultdict


def evaluate_dev(stage, model, tokenizer, accelerator):
    import torch

    from lm_trainer.template_policy import chat_template_kwargs, resolve_policy
    from lm_trainer.train.data import (
        DataCollatorForCausalLMWith8xPadding,
        load_training_dataset,
    )

    model.eval()
    result = {}
    with torch.inference_mode():
        if stage.dev is not None:
            dataset = load_training_dataset(
                stage.runtime, tokenizer, validated=stage.dev
            )
            collate = DataCollatorForCausalLMWith8xPadding(
                tokenizer, stage.runtime["max_length"], False
            )
            sources = defaultdict(
                lambda: {"loss_tokens": 0, "nll_sum": 0.0, "samples": 0}
            )
            for item in dataset:
                batch = collate([item])
                with accelerator.autocast():
                    loss = float(
                        model(
                            **{k: v.to(accelerator.device) for k, v in batch.items()}
                        ).loss
                    )
                if not math.isfinite(loss):
                    raise FloatingPointError("Non-finite dev loss.")
                trace = item["trace"]
                summary = sources[trace["source"]]
                summary["loss_tokens"] += trace["loss_tokens"]
                summary["nll_sum"] += loss * trace["loss_tokens"]
                summary["samples"] += 1
            for summary in sources.values():
                summary["loss"] = summary.pop("nll_sum") / summary["loss_tokens"]
            result.update(
                split="dev",
                manifest_sha256=stage.dev.plan["manifest_sha256"],
                sources=dict(sources),
            )
        if stage.runtime["dev_probe"]:
            _, policy = resolve_policy(tokenizer, stage.runtime["template_policy"])
            messages = [
                {"role": "user", "content": "虚构咨询：相机图像发暗，先检查什么？"}
            ]
            if policy["system_prompt"]:
                messages.insert(
                    0, {"role": "system", "content": policy["system_prompt"]}
                )
            kwargs = chat_template_kwargs(policy, for_training=False)
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, return_tensors="pt", return_dict=True, **kwargs
            )
            length = inputs["input_ids"].shape[-1]
            allowance = min(16, stage.runtime["max_length"] - length)
            if allowance < 1:
                raise ValueError("Dev probe does not fit within max_length.")
            from transformers import GenerationConfig

            config = GenerationConfig(
                do_sample=False,
                max_new_tokens=allowance,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=policy["stop_token_ids"],
            )
            with accelerator.autocast():
                output = model.generate(
                    **inputs.to(accelerator.device), generation_config=config
                )
            ids = output[0, length:].tolist()
            probe = {
                "probe_id": "fictional_camera_dark_v1",
                "generated_tokens": len(ids),
                "stop_matched": bool(ids and ids[-1] in policy["stop_token_ids"]),
            }
            if stage.runtime["save_probe_text"]:
                probe["fixture_only"] = True
                probe["text"] = tokenizer.decode(ids, skip_special_tokens=True)
            result["probe"] = probe
    return result or None
