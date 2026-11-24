"""Compatibility interface for lm_trainer.model.quantization."""

from lm_trainer.model.quantization import (
    normalize_quantization_mode,
    build_quantization_config,
)

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantization config smoke test")
    parser.add_argument("--mode", default="8bit")
    args = parser.parse_args()
    config = build_quantization_config(args.mode)
    print(
        json.dumps(
            {
                "mode": normalize_quantization_mode(args.mode),
                "enabled": config is not None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
