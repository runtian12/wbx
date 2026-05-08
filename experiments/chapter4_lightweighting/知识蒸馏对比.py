"""Knowledge-distillation comparison plots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


METHODS = ["OnlineKD", "GKD", "SemCKD", "Proposed"]
MODELS = ["LLaMA7B", "LLaMA13B"]
COLORS = ["#5A7B9C", "#BC4749"]

ACC_DATA = {
    "BoolQ": {"LLaMA7B": [63.5, 65.2, 64.8, 66.7], "LLaMA13B": [67.0, 68.5, 67.9, 69.8], "ylim": [50, 80]},
    "PIQA": {"LLaMA7B": [70.5, 73.1, 72.5, 74.3], "LLaMA13B": [75.2, 77.8, 76.9, 79.1], "ylim": [60, 90]},
}

PPL_DATA = {
    "WikiText2": {"LLaMA7B": [19.2, 17.8, 18.5, 17.0], "LLaMA13B": [15.1, 13.9, 14.7, 13.1], "ylim": [10, 25]},
    "PTB": {"LLaMA7B": [35.1, 32.8, 34.2, 32.0], "LLaMA13B": [28.2, 26.5, 27.6, 25.7], "ylim": [20, 45]},
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


def plot_metric(metric_name: str, dataset_name: str, dataset: dict, output_dir: Path) -> Path:
    x = np.arange(len(METHODS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.bar(x - 0.5 * width, dataset["LLaMA7B"], width, label="LLaMA7B", color=COLORS[0], edgecolor="black", linewidth=0.8, zorder=3)
    ax.bar(x + 0.5 * width, dataset["LLaMA13B"], width, label="LLaMA13B", color=COLORS[1], edgecolor="black", linewidth=0.8, zorder=3)

    ax.set_ylabel("准确率（%）" if metric_name == "Acc" else "困惑度（PPL）", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylim(dataset["ylim"][0], dataset["ylim"][1])
    ax.grid(axis="y", linestyle="--", color="gray", alpha=0.3, zorder=0)
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    output_path = output_dir / f"{metric_name.lower()}_{dataset_name.lower()}_distillation.png"
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_csv(output_dir: Path) -> Path:
    path = output_dir / "distillation_comparison_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "dataset", "model", "method", "value"])
        for dataset_name, dataset in ACC_DATA.items():
            for model in MODELS:
                for method, value in zip(METHODS, dataset[model]):
                    writer.writerow(["Accuracy", dataset_name, model, method, value])
        for dataset_name, dataset in PPL_DATA.items():
            for model in MODELS:
                for method, value in zip(METHODS, dataset[model]):
                    writer.writerow(["PPL", dataset_name, model, method, value])
    return path


def run(output_dir: Path) -> list[Path]:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for dataset_name, dataset in ACC_DATA.items():
        files.append(plot_metric("Acc", dataset_name, dataset, output_dir))
    for dataset_name, dataset in PPL_DATA.items():
        files.append(plot_metric("PPL", dataset_name, dataset, output_dir))
    files.append(write_csv(output_dir))
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate knowledge-distillation comparison plots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "chapter4_lightweighting" / "distillation_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = run(args.output_dir)
    print("Knowledge distillation comparison experiment completed.")
    for file in files:
        print(f"  - {file.resolve()}")


if __name__ == "__main__":
    main()

