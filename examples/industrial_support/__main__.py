"""Look up the ready indicator for a fictional device and hardware revision."""

import argparse
import json
from pathlib import Path


def answer(model, revision, manuals):
    if not revision:
        return "请先确认铭牌上的硬件版本，再核对就绪指示灯。"
    matches = [
        row for row in manuals if (row["model"], row["revision"]) == (model, revision)
    ]
    if len(matches) != 1:
        return "该型号及版本的手册证据不足，请核对铭牌或转人工。"
    record = matches[0]
    return f"{model} 硬件 {revision} 的就绪灯为{record['indicator']}。依据 {record['source']}。"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--revision")
    parser.add_argument(
        "--manuals", type=Path, default=Path(__file__).with_name("manuals.json")
    )
    args = parser.parse_args()
    print(
        answer(
            args.model,
            args.revision,
            json.loads(args.manuals.read_text(encoding="utf-8")),
        )
    )


if __name__ == "__main__":
    main()
