from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_metrics(metrics: dict[str, list[float]], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    titles = ["Loss", "Perplexity", "Learning Rate", "Step Time", "Throughput"]
    colors = ["b", "orange", "red", "purple", "brown"]
    keys = ["loss", "perplexity", "lr", "step_time", "throughput"]

    for ax, title, key, color in zip(axes, titles, keys, colors):
        ax.plot(range(1, len(metrics[key]) + 1), metrics[key], marker="o", color=color)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)

    if len(axes) > len(keys):
        for ax in axes[len(keys):]:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def empty_metrics() -> dict[str, list[float]]:
    return {
        "loss": [],
        "perplexity": [],
        "lr": [],
        "step_time": [],
        "throughput": [],
    }


def sample_metrics() -> dict[str, list[float]]:
    return {
        "loss": [1.0, 0.8],
        "perplexity": [2.7, 2.2],
        "lr": [1.0e-5, 9.0e-6],
        "step_time": [0.5, 0.45],
        "throughput": [6.0, 6.5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Metrics plotting smoke test")
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()
    metrics = sample_metrics()
    if args.save_path:
        plot_metrics(metrics, args.save_path)
    print(json.dumps({"keys": list(metrics), "saved": args.save_path is not None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
