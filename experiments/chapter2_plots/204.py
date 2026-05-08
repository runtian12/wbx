"""Hardware-platform inference comparison plots.

This script generates the three figures used for the hardware experiment:
accuracy, latency, and throughput across Jetson Orin Nano, Jetson Orin NX,
and RTX 3060.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HARDWARES = ["Jetson Orin Nano", "Jetson Orin NX", "RTX 3060"]
ALGORITHMS = ["Proposed", "IACI", "PSOCI", "Neur", "DeeBERT"]
COLORS = ["#1f77b4", "#4c72b0", "#55a868", "#8172b2", "#ccb974"]

ACCURACY_DATA = {
    "Proposed": [75.2, 75.2, 75.2],
    "IACI": [72.8, 72.8, 72.8],
    "PSOCI": [71.5, 71.5, 71.5],
    "Neur": [48.2, 60.2, 63.5],
    "DeeBERT": [52.5, 62.5, 65.2],
}

LATENCY_DATA = {
    "Proposed": [831, 561, 341],
    "IACI": [1301, 882, 568],
    "PSOCI": [1410, 965, 612],
    "Neur": [1671, 1124, 701],
    "DeeBERT": [1042, 715, 456],
}

THROUGHPUT_DATA = {
    "Proposed": [13.0, 21.5, 32.4],
    "IACI": [7.1, 13.8, 18.5],
    "PSOCI": [5.6, 12.4, 16.2],
    "Neur": [4.2, 10.5, 14.8],
    "DeeBERT": [8.4, 15.2, 22.8],
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
    x = np.arange(len(HARDWARES))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 6))

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
    ax.set_xticklabels(HARDWARES)

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
    summary_path = output_dir / "hardware_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["scenario", "algorithm", "hardware", "accuracy_percent", "latency_ms", "throughput_tokens_s"])
        for alg in ALGORITHMS:
            for index, hardware in enumerate(HARDWARES):
                writer.writerow(
                    [
                        "hardware",
                        alg,
                        hardware,
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
        output_dir / "hardware_accuracy.png",
        output_dir / "hardware_latency.png",
        output_dir / "hardware_throughput.png",
    ]

    plot_bar_chart(ACCURACY_DATA, "推理准确度 (%)", generated_files[0], ylim=(30, 95))
    plot_bar_chart(LATENCY_DATA, "推理时延 (ms)", generated_files[1], ylim=(0, 2000))
    plot_bar_chart(THROUGHPUT_DATA, "推理吞吐量 (Tokens/s)", generated_files[2], ylim=(0, 45))
    generated_files.append(write_summary_csv(output_dir))

    return generated_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hardware-platform inference plots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "chapter2_plots" / "hardware",
        help="Directory used to store generated figures and metric CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = run(args.output_dir)
    print("Hardware experiment completed.")
    for file in files:
        print(f"  - {file.resolve()}")


if __name__ == "__main__":
    main()

