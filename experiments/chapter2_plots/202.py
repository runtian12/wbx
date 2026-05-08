"""Bandwidth-constrained inference comparison plots.

This script generates the three figures used for the bandwidth experiment:
accuracy, latency, and throughput under 10/30/50 Mbps network constraints.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BANDWIDTHS = ["10 Mbps", "30 Mbps", "50 Mbps"]
ALGORITHMS = ["Proposed", "IACI", "PSOCI", "Neur", "DeeBERT"]
COLORS = ["#1f77b4", "#4c72b0", "#55a868", "#8172b2", "#ccb974"]

ACCURACY_DATA = {
    "Proposed": [72.4, 74.5, 76.2],
    "IACI": [68.8, 71.2, 72.6],
    "PSOCI": [67.5, 69.8, 71.4],
    "Neur": [62.5, 64.8, 66.2],
    "DeeBERT": [64.2, 66.5, 68.0],
}

LATENCY_DATA = {
    "Proposed": [510, 420, 340],
    "IACI": [740, 610, 520],
    "PSOCI": [810, 680, 580],
    "Neur": [890, 750, 640],
    "DeeBERT": [620, 510, 430],
}

THROUGHPUT_DATA = {
    "Proposed": [18.2, 21.5, 24.8],
    "IACI": [13.4, 15.8, 18.5],
    "PSOCI": [12.2, 14.5, 16.8],
    "Neur": [10.5, 12.6, 14.5],
    "DeeBERT": [15.6, 18.8, 21.2],
}


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.dpi": 300,
        }
    )


def plot_bar_chart(data_dict, ylabel, filename, ylim=None) -> None:
    x = np.arange(len(BANDWIDTHS))
    width = 0.15
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, alg in enumerate(ALGORITHMS):
        offset = (i - len(ALGORITHMS) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            data_dict[alg],
            width,
            label=alg,
            color=COLORS[i],
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(BANDWIDTHS)

    if ylim:
        ax.set_ylim(ylim)

    ax.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=0.8,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def write_summary_csv(output_dir: Path) -> Path:
    summary_path = output_dir / "bandwidth_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["scenario", "algorithm", "bandwidth", "accuracy_percent", "latency_ms", "throughput_tokens_s"])
        for alg in ALGORITHMS:
            for index, bandwidth in enumerate(BANDWIDTHS):
                writer.writerow(
                    [
                        "bandwidth",
                        alg,
                        bandwidth,
                        ACCURACY_DATA[alg][index],
                        LATENCY_DATA[alg][index],
                        THROUGHPUT_DATA[alg][index],
                    ]
                )
    return summary_path


def run(output_dir: Path) -> list:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = [
        output_dir / "bw_accuracy_bar.png",
        output_dir / "bw_latency_bar.png",
        output_dir / "bw_throughput_bar.png",
    ]

    plot_bar_chart(ACCURACY_DATA, "推理准确度 (%)", generated_files[0], ylim=(30, 95))
    plot_bar_chart(LATENCY_DATA, "推理时延 (ms)", generated_files[1], ylim=(0, 1150))
    plot_bar_chart(THROUGHPUT_DATA, "推理吞吐量 (Tokens/s)", generated_files[2], ylim=(0, 32))
    generated_files.append(write_summary_csv(output_dir))

    return generated_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate bandwidth-constrained inference plots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "chapter2_plots" / "bandwidth",
        help="Directory used to store generated figures and metric CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = run(args.output_dir)
    print("Bandwidth experiment completed.")
    for file in files:
        print(f"  - {file.resolve()}")


if __name__ == "__main__":
    main()

