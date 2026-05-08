"""Pruning-rate perplexity plots for WikiText2 and PTB."""

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

PPL_DATA = {
    "WikiText2": {
        "Magnitude": [17.8, 22.5, 29.2, 38.5],
        "DLP": [17.4, 21.8, 28.1, 37.2],
        "LLM-Pruner": [17.6, 21.3, 27.5, 36.4],
        "Proposed": [17.0, 20.5, 26.2, 35.1],
    },
    "PTB": {
        "Magnitude": [32.8, 38.5, 49.5, 67.2],
        "DLP": [30.5, 37.1, 48.2, 65.8],
        "LLM-Pruner": [29.8, 36.4, 47.1, 64.5],
        "Proposed": [29.2, 35.1, 45.8, 61.2],
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
    ax.set_ylabel("困惑度 (PPL)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(PRUNING_RATES)
    ax.set_ylim((10, 45) if dataset_name == "WikiText2" else (20, 75))
    ax.grid(axis="y", linestyle="--", color="gray", alpha=0.3, zorder=0)
    ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    output_path = output_dir / f"pruning_ppl_{dataset_name.lower()}.png"
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_csv(output_dir: Path) -> Path:
    path = output_dir / "pruning_ppl_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "algorithm", "pruning_rate_percent", "perplexity"])
        for dataset_name, dataset in PPL_DATA.items():
            for algorithm in ALGORITHMS:
                for rate, value in zip(PRUNING_RATES, dataset[algorithm]):
                    writer.writerow([dataset_name, algorithm, rate, value])
    return path


def run(output_dir: Path) -> list[Path]:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [plot_dataset(name, data, output_dir) for name, data in PPL_DATA.items()]
    files.append(write_csv(output_dir))
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WikiText2/PTB pruning perplexity plots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "chapter4_lightweighting" / "pruning_ppl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = run(args.output_dir)
    print("Pruning perplexity experiment completed.")
    for file in files:
        print(f"  - {file.resolve()}")


if __name__ == "__main__":
    main()

