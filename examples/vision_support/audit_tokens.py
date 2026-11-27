"""Print the fictional SFT fixture's position/ID/token/label/role table (opt-in)."""

import argparse
import json
from itertools import groupby
from pathlib import Path

from lm_trainer.template_policy import resolve_policy
from lm_trainer.train.encoding import encode_sft
from lm_trainer.train.schema import preflight_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Local tokenizer directory, never a Hub ID",
    )
    parser.add_argument("--stop-token", action="append", default=None)
    parser.add_argument("--thinking-flag-unsupported", action="store_true")
    parser.add_argument("--without-eos", action="store_true")
    args = parser.parse_args()
    if not args.tokenizer.is_dir():
        parser.error("local tokenizer directory is missing")
    from lm_trainer.model.loader import load_tokenizer

    bundle = Path(__file__).parent / "task_a_fixture" / "manifest.json"
    validated = preflight_data(
        {
            "data_mode": "fixture",
            "manifest_path": str(bundle),
            "objective": "assistant_only_messages",
            "max_length": 1024,
            "supervise_target_eos": not args.without_eos,
            "template_policy": {
                "stop_tokens": args.stop_token or ["<|im_end|>"],
                "enable_thinking": None if args.thinking_flag_unsupported else False,
            },
        }
    )
    tokenizer = load_tokenizer(
        str(args.tokenizer), use_fast=True, local_files_only=True
    )
    template, policy = resolve_policy(tokenizer, validated.plan["template_policy"])
    print(f"Template SHA-256: `{policy['template_hash']}`\n")
    result = encode_sft(
        tokenizer, validated.samples[0].row, template, policy, validated.plan
    )
    # Decode individual body spans independently and check before printing.
    sample = validated.samples[0].row
    expected = [
        m["content"] for m, yes in zip(sample["messages"], sample["target_mask"]) if yes
    ]
    actual = []
    for role, indices in groupby(
        range(len(result.roles)), key=result.roles.__getitem__
    ):
        positions = list(indices)
        if role == "assistant_target":
            assert all(result.labels[i] == result.input_ids[i] for i in positions)
            actual.append(
                tokenizer.decode(
                    [result.input_ids[i] for i in positions],
                    clean_up_tokenization_spaces=False,
                )
            )
        elif role == "assistant_stop":
            assert all(
                result.labels[i] == -100 or result.labels[i] in policy["stop_token_ids"]
                for i in positions
            )
        else:
            assert all(result.labels[i] == -100 for i in positions)
    assert actual == expected
    assert result.eos_targets == (0 if args.without_eos else sum(sample["target_mask"]))
    print("| index | token_id | token_str | label | role |\n|---:|---:|---|---:|---|")
    for i, (token, label, role) in enumerate(
        zip(result.input_ids, result.labels, result.roles)
    ):
        fragment = json.dumps(
            tokenizer.convert_ids_to_tokens(token), ensure_ascii=False
        )
        fragment = (
            fragment.replace("&", "&amp;")
            .replace("|", "&#124;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        print(f"| {i} | {token} | <code>{fragment}</code> | {label} | {role} |")
    print(f"\nShifted targets: body={result.body_targets}, EOS={result.eos_targets}.")


if __name__ == "__main__":
    main()
