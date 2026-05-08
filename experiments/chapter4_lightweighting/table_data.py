"""Structured data from Section 4.1.6 result tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PRUNING_RATES = ["20", "30", "40", "50"]
BASELINES = ["Magnitude", "DLP", "LLM-Pruner"]

TABLE_4_3 = [
    ("Jetson Orin Nano", "Magnitude", "Accuracy", [63.7, 61.5, 57.8, 54.9]),
    ("Jetson Orin Nano", "Magnitude", "PPL", [18.1, 22.8, 29.5, 38.9]),
    ("Jetson Orin Nano", "DLP", "Accuracy", [65.4, 64.2, 60.9, 59.8]),
    ("Jetson Orin Nano", "DLP", "PPL", [17.6, 22.0, 28.3, 37.5]),
    ("Jetson Orin Nano", "LLM-Pruner", "Accuracy", [64.8, 63.9, 59.7, 58.5]),
    ("Jetson Orin Nano", "LLM-Pruner", "PPL", [17.8, 21.5, 27.7, 36.6]),
    ("Jetson Orin Nano", "Proposed", "Accuracy", [67.2, 66.1, 63.1, 61.9]),
    ("Jetson Orin Nano", "Proposed", "PPL", [17.2, 20.7, 26.5, 35.3]),
    ("Jetson Orin NX", "Magnitude", "Accuracy", [66.2, 63.8, 60.5, 57.1]),
    ("Jetson Orin NX", "Magnitude", "PPL", [16.8, 21.2, 27.1, 35.8]),
    ("Jetson Orin NX", "DLP", "Accuracy", [68.1, 66.9, 63.7, 62.4]),
    ("Jetson Orin NX", "DLP", "PPL", [16.3, 20.4, 26.0, 34.2]),
    ("Jetson Orin NX", "LLM-Pruner", "Accuracy", [67.5, 66.4, 62.2, 61.3]),
    ("Jetson Orin NX", "LLM-Pruner", "PPL", [16.5, 19.9, 25.5, 33.1]),
    ("Jetson Orin NX", "Proposed", "Accuracy", [69.8, 68.7, 65.5, 64.6]),
    ("Jetson Orin NX", "Proposed", "PPL", [15.8, 18.9, 24.1, 31.8]),
    ("RTX 4060", "Magnitude", "Accuracy", [68.5, 66.2, 62.8, 59.4]),
    ("RTX 4060", "Magnitude", "PPL", [14.5, 18.2, 23.5, 30.2]),
    ("RTX 4060", "DLP", "Accuracy", [70.4, 69.1, 65.9, 64.6]),
    ("RTX 4060", "DLP", "PPL", [14.1, 17.6, 22.8, 29.1]),
    ("RTX 4060", "LLM-Pruner", "Accuracy", [69.8, 68.7, 64.5, 63.6]),
    ("RTX 4060", "LLM-Pruner", "PPL", [14.2, 17.2, 22.1, 28.3]),
    ("RTX 4060", "Proposed", "Accuracy", [72.3, 71.2, 68.1, 67.0]),
    ("RTX 4060", "Proposed", "PPL", [13.5, 16.1, 20.5, 26.8]),
]

TABLE_4_4 = [
    ("LLaMA-13B", "Proposed", "Accuracy", "BoolQ", [81.5, 78.3, 73.6, 67.4]),
    ("LLaMA-13B", "Proposed", "Accuracy", "PIQA", [82.3, 79.5, 75.1, 70.2]),
    ("LLaMA-13B", "Proposed", "PPL", "WikiText2", [13.5, 15.2, 18.1, 23.4]),
    ("LLaMA-13B", "Proposed", "PPL", "PTB", [15.8, 17.6, 20.5, 25.8]),
    ("LLaMA-13B", "w/o Hardware-aware Pruning", "Accuracy", "BoolQ", [79.8, 75.1, 68.8, 61.5]),
    ("LLaMA-13B", "w/o Hardware-aware Pruning", "Accuracy", "PIQA", [80.5, 76.2, 70.8, 64.1]),
    ("LLaMA-13B", "w/o Hardware-aware Pruning", "PPL", "WikiText2", [14.8, 18.3, 23.5, 31.2]),
    ("LLaMA-13B", "w/o Hardware-aware Pruning", "PPL", "PTB", [17.1, 20.5, 25.6, 33.5]),
    ("LLaMA-13B", "w/o Selective Distillation", "Accuracy", "BoolQ", [78.6, 74.2, 69.5, 63.8]),
    ("LLaMA-13B", "w/o Selective Distillation", "Accuracy", "PIQA", [79.7, 75.8, 72.1, 66.5]),
    ("LLaMA-13B", "w/o Selective Distillation", "PPL", "WikiText2", [15.7, 18.9, 20.2, 26.5]),
    ("LLaMA-13B", "w/o Selective Distillation", "PPL", "PTB", [18.2, 22.1, 22.8, 29.1]),
    ("LLaMA-7B", "Proposed", "Accuracy", "BoolQ", [78.2, 74.6, 69.1, 62.5]),
    ("LLaMA-7B", "Proposed", "Accuracy", "PIQA", [79.1, 75.8, 71.2, 65.8]),
    ("LLaMA-7B", "Proposed", "PPL", "WikiText2", [16.2, 18.5, 22.4, 28.7]),
    ("LLaMA-7B", "Proposed", "PPL", "PTB", [18.5, 20.4, 25.1, 31.5]),
    ("LLaMA-7B", "w/o Hardware-aware Pruning", "Accuracy", "BoolQ", [76.4, 71.2, 64.8, 57.3]),
    ("LLaMA-7B", "w/o Hardware-aware Pruning", "Accuracy", "PIQA", [77.2, 72.4, 66.5, 59.2]),
    ("LLaMA-7B", "w/o Hardware-aware Pruning", "PPL", "WikiText2", [17.8, 22.1, 28.5, 37.4]),
    ("LLaMA-7B", "w/o Hardware-aware Pruning", "PPL", "PTB", [19.9, 23.5, 30.2, 40.1]),
    ("LLaMA-7B", "w/o Selective Distillation", "Accuracy", "BoolQ", [75.5, 70.8, 65.2, 59.4]),
    ("LLaMA-7B", "w/o Selective Distillation", "Accuracy", "PIQA", [76.6, 72.2, 68.5, 62.1]),
    ("LLaMA-7B", "w/o Selective Distillation", "PPL", "WikiText2", [18.9, 22.8, 24.1, 31.2]),
    ("LLaMA-7B", "w/o Selective Distillation", "PPL", "PTB", [20.9, 25.6, 26.8, 34.2]),
]

TABLE_4_5 = [
    ("LLaMA-7B", "Magnitude", "Accuracy", [68.4, 65.3, 61.2, 56.4]),
    ("LLaMA-7B", "Magnitude", "PPL", [16.5, 19.4, 24.1, 31.2]),
    ("LLaMA-7B", "DLP", "Accuracy", [70.2, 68.1, 64.5, 60.1]),
    ("LLaMA-7B", "DLP", "PPL", [15.8, 18.6, 23.2, 29.8]),
    ("LLaMA-7B", "LLM-Pruner", "Accuracy", [69.7, 67.5, 63.8, 59.5]),
    ("LLaMA-7B", "LLM-Pruner", "PPL", [16.1, 18.9, 23.6, 30.4]),
    ("LLaMA-7B", "Proposed", "Accuracy", [72.5, 70.6, 67.2, 63.4]),
    ("LLaMA-7B", "Proposed", "PPL", [14.9, 17.5, 21.8, 27.6]),
    ("LLaMA-13B", "Magnitude", "Accuracy", [72.1, 69.8, 65.4, 60.2]),
    ("LLaMA-13B", "Magnitude", "PPL", [13.5, 15.8, 19.4, 25.1]),
    ("LLaMA-13B", "DLP", "Accuracy", [74.5, 72.2, 68.5, 64.1]),
    ("LLaMA-13B", "DLP", "PPL", [12.8, 14.6, 18.2, 23.5]),
    ("LLaMA-13B", "LLM-Pruner", "Accuracy", [73.8, 71.5, 67.9, 63.6]),
    ("LLaMA-13B", "LLM-Pruner", "PPL", [13.1, 15.1, 18.8, 24.2]),
    ("LLaMA-13B", "Proposed", "Accuracy", [77.2, 75.1, 72.4, 68.8]),
    ("LLaMA-13B", "Proposed", "PPL", [11.2, 13.5, 16.4, 21.5]),
]


def ensure_output_dir(name: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "outputs" / "chapter4_lightweighting" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_json(path: Path, payload) -> Path:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return path


def write_text(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def higher_is_better(metric: str) -> bool:
    return metric == "Accuracy"


def improvement(proposed: float, baseline: float, metric: str) -> float:
    if higher_is_better(metric):
        return (proposed - baseline) / baseline * 100
    return (baseline - proposed) / baseline * 100

