"""Pruning-rate accuracy plots for BoolQ and PIQA."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PRUNING_RATES = ["20", "30", "40", "50"]
ALGORITHMS = ["Magnitude", "DLP", "LLM-Pruner", "Proposed"]
COLORS = ["#5A7B9C", "#789E73", "#D4A373", "#BC4749"]

ACCURACY_DATA = {
    "BoolQ": {
        "Magnitude": [64.2, 62.5, 58.4, 56.1],
        "DLP": [66.1, 65.2, 61.8, 60.9],
        "LLM-Pruner": [65.4, 64.7, 60.2, 59.8],
        "Proposed": [66.8, 65.9, 62.7, 62.4],
    },
    "PIQA": {
        "Magnitude": [71.5, 67.2, 62.2, 56.5],
        "DLP": [75.8, 74.5, 70.8, 64.5],
        "LLM-Pruner": [77.0, 71.6, 69.8, 63.2],
        "Proposed": [77.2, 74.8, 71.1, 64.8],
    },
}


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.width"] = 1.2
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 12


def plot_dataset(dataset_name: str, dataset: dict[str, list[float]], output_dir: Path) -> Path:
    x = np.arange(len(PRUNING_RATES))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, algorithm in enumerate(ALGORITHMS):
        offset = (i - 1.5) * width
        ax.bar(
            x + offset,
            dataset[algorithm],
            width,
            label=algorithm,
            color=COLORS[i],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

    ax.set_xlabel("剪枝率 (%)", fontweight="bold")
    ax.set_ylabel("准确率 (%)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(PRUNING_RATES)
    ax.set_ylim(50, 80)
    ax.grid(axis="y", linestyle="--", color="gray", alpha=0.3, zorder=0)
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    output_path = output_dir / f"pruning_accuracy_{dataset_name.lower()}.png"
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_csv(output_dir: Path) -> Path:
    path = output_dir / "pruning_accuracy_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "algorithm", "pruning_rate_percent", "accuracy_percent"])
        for dataset_name, dataset in ACCURACY_DATA.items():
            for algorithm in ALGORITHMS:
                for rate, value in zip(PRUNING_RATES, dataset[algorithm]):
                    writer.writerow([dataset_name, algorithm, rate, value])
    return path


def run(output_dir: Path) -> list[Path]:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [plot_dataset(name, data, output_dir) for name, data in ACCURACY_DATA.items()]
    files.append(write_csv(output_dir))
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BoolQ/PIQA pruning accuracy plots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "chapter4_lightweighting" / "pruning_accuracy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = run(args.output_dir)
    print("Pruning accuracy experiment completed.")
    for file in files:
        print(f"  - {file.resolve()}")


if __name__ == "__main__":
    main()

