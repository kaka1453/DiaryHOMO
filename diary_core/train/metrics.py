"""Compatibility interface for lm_trainer.train.metrics."""

from lm_trainer.train.metrics import (
    plot_metrics,
    empty_metrics,
    sample_metrics,
)

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Metrics plotting smoke test")
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()
    metrics = sample_metrics()
    if args.save_path:
        plot_metrics(metrics, args.save_path)
    print(
        json.dumps(
            {"keys": list(metrics), "saved": args.save_path is not None},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
