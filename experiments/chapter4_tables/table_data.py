"""Structured data from the Chapter 4.2 result tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ALGORITHMS = ["Neur", "DeeBERT", "IACI", "PSOCI", "Proposed"]
BASELINES = ["Neur", "DeeBERT", "IACI", "PSOCI"]
THRESHOLDS = ["0.50", "0.55", "0.60", "0.65"]

TABLE_4_8 = [
    ("LLaMA-7B", "GSM8K", "Acc", {"Neur": 63.5, "DeeBERT": 61.2, "IACI": 68.4, "PSOCI": 66.7, "Proposed": 72.5}),
    ("LLaMA-7B", "GSM8K", "Lat", {"Neur": 852, "DeeBERT": 615, "IACI": 784, "PSOCI": 824, "Proposed": 568}),
    ("LLaMA-7B", "GSM8K", "Tpt", {"Neur": 7.8, "DeeBERT": 21.2, "IACI": 17.5, "PSOCI": 15.1, "Proposed": 28.5}),
    ("LLaMA-7B", "Alpaca", "Acc", {"Neur": 68.5, "DeeBERT": 62.3, "IACI": 76.8, "PSOCI": 75.2, "Proposed": 78.2}),
    ("LLaMA-7B", "Alpaca", "Lat", {"Neur": 864, "DeeBERT": 624, "IACI": 778, "PSOCI": 831, "Proposed": 572}),
    ("LLaMA-7B", "Alpaca", "Tpt", {"Neur": 14.8, "DeeBERT": 24.5, "IACI": 19.8, "PSOCI": 17.6, "Proposed": 44.6}),
    ("LLaMA-7B", "HumanEval", "Acc", {"Neur": 50.2, "DeeBERT": 46.8, "IACI": 61.5, "PSOCI": 59.6, "Proposed": 65.8}),
    ("LLaMA-7B", "HumanEval", "Lat", {"Neur": 880, "DeeBERT": 642, "IACI": 856, "PSOCI": 894, "Proposed": 577}),
    ("LLaMA-7B", "HumanEval", "Tpt", {"Neur": 6.5, "DeeBERT": 18.6, "IACI": 15.2, "PSOCI": 13.8, "Proposed": 35.2}),
    ("LLaMA3-8B", "GSM8K", "Acc", {"Neur": 75.2, "DeeBERT": 73.8, "IACI": 81.5, "PSOCI": 79.4, "Proposed": 86.2}),
    ("LLaMA3-8B", "GSM8K", "Lat", {"Neur": 871, "DeeBERT": 632, "IACI": 797, "PSOCI": 842, "Proposed": 581}),
    ("LLaMA3-8B", "GSM8K", "Tpt", {"Neur": 4.2, "DeeBERT": 11.5, "IACI": 9.5, "PSOCI": 8.2, "Proposed": 15.6}),
    ("LLaMA3-8B", "Alpaca", "Acc", {"Neur": 79.2, "DeeBERT": 72.5, "IACI": 87.5, "PSOCI": 85.8, "Proposed": 89.6}),
    ("LLaMA3-8B", "Alpaca", "Lat", {"Neur": 880, "DeeBERT": 671, "IACI": 813, "PSOCI": 849, "Proposed": 588}),
    ("LLaMA3-8B", "Alpaca", "Tpt", {"Neur": 8.1, "DeeBERT": 13.5, "IACI": 10.8, "PSOCI": 9.5, "Proposed": 24.2}),
    ("LLaMA3-8B", "HumanEval", "Acc", {"Neur": 62.5, "DeeBERT": 58.4, "IACI": 74.2, "PSOCI": 71.8, "Proposed": 79.5}),
    ("LLaMA3-8B", "HumanEval", "Lat", {"Neur": 891, "DeeBERT": 687, "IACI": 832, "PSOCI": 851, "Proposed": 575}),
    ("LLaMA3-8B", "HumanEval", "Tpt", {"Neur": 3.5, "DeeBERT": 10.1, "IACI": 8.2, "PSOCI": 7.5, "Proposed": 19.1}),
]

TABLE_4_9 = [
    ("LLaMA-7B", "GSM8K", "Acc", [72.4, 72.3, 72.0, 71.4]),
    ("LLaMA-7B", "GSM8K", "Lat", [610, 568, 545, 520]),
    ("LLaMA-7B", "GSM8K", "Tpt", [25.4, 28.5, 30.2, 32.1]),
    ("LLaMA-7B", "Alpaca", "Acc", [78.1, 78.2, 77.8, 77.1]),
    ("LLaMA-7B", "Alpaca", "Lat", [615, 572, 550, 530]),
    ("LLaMA-7B", "Alpaca", "Tpt", [27.5, 28.6, 31.8, 35.2]),
    ("LLaMA-7B", "HumanEval", "Acc", [65.6, 65.8, 65.2, 64.5]),
    ("LLaMA-7B", "HumanEval", "Lat", [620, 577, 558, 535]),
    ("LLaMA-7B", "HumanEval", "Tpt", [18.6, 23.2, 25.4, 26.8]),
    ("LLaMA3-8B", "GSM8K", "Acc", [76.9, 76.7, 75.8, 74.1]),
    ("LLaMA3-8B", "GSM8K", "Lat", [625, 581, 558, 535]),
    ("LLaMA3-8B", "GSM8K", "Tpt", [23.9, 25.6, 26.8, 28.2]),
    ("LLaMA3-8B", "Alpaca", "Acc", [82.5, 81.6, 81.1, 80.5]),
    ("LLaMA3-8B", "Alpaca", "Lat", [630, 588, 565, 540]),
    ("LLaMA3-8B", "Alpaca", "Tpt", [21.5, 24.2, 25.8, 27.5]),
    ("LLaMA3-8B", "HumanEval", "Acc", [69.4, 68.5, 67.9, 66.1]),
    ("LLaMA3-8B", "HumanEval", "Lat", [618, 575, 552, 530]),
    ("LLaMA3-8B", "HumanEval", "Tpt", [16.8, 19.1, 20.8, 22.4]),
]

TABLE_4_10 = [
    ("Jetson Orin Nano", "IACI", 66.8, 1315, 7.1),
    ("Jetson Orin Nano", "PSOCI", 67.5, 1426, 5.6),
    ("Jetson Orin Nano", "Neur", 56.1, 1658, 4.2),
    ("Jetson Orin Nano", "DeeBERT", 57.6, 1042, 8.4),
    ("Jetson Orin Nano", "Proposed", 72.4, 811, 13.0),
    ("Jetson Orin NX", "IACI", 72.8, 882, 13.8),
    ("Jetson Orin NX", "PSOCI", 71.5, 965, 12.4),
    ("Jetson Orin NX", "Neur", 60.2, 1124, 10.5),
    ("Jetson Orin NX", "DeeBERT", 62.5, 715, 15.2),
    ("Jetson Orin NX", "Proposed", 75.2, 550, 21.5),
    ("RTX 3060", "IACI", 74.1, 568, 18.5),
    ("RTX 3060", "PSOCI", 72.3, 615, 16.2),
    ("RTX 3060", "Neur", 62.5, 712, 14.8),
    ("RTX 3060", "DeeBERT", 65.2, 456, 22.8),
    ("RTX 3060", "Proposed", 77.2, 350, 32.4),
]

TABLE_4_11 = [
    ("LLaMA-7B", "Proposed", "GSM8K", 72.5, 568, 28.5),
    ("LLaMA-7B", "Proposed", "Alpaca", 78.2, 572, 44.6),
    ("LLaMA-7B", "Proposed", "HumanEval", 65.8, 577, 35.2),
    ("LLaMA-7B", "w/o DD", "GSM8K", 70.9, 685, 24.5),
    ("LLaMA-7B", "w/o DD", "Alpaca", 76.8, 693, 39.8),
    ("LLaMA-7B", "w/o DD", "HumanEval", 64.3, 689, 30.4),
    ("LLaMA-7B", "w/o CDV", "GSM8K", 70.6, 718, 22.4),
    ("LLaMA-7B", "w/o CDV", "Alpaca", 76.3, 720, 34.6),
    ("LLaMA-7B", "w/o CDV", "HumanEval", 63.9, 732, 27.1),
    ("LLaMA-7B", "w/o Both", "GSM8K", 68.1, 892, 13.8),
    ("LLaMA-7B", "w/o Both", "Alpaca", 74.4, 890, 20.5),
    ("LLaMA-7B", "w/o Both", "HumanEval", 60.7, 889, 17.6),
    ("LLaMA3-8B", "Proposed", "GSM8K", 86.2, 581, 15.6),
    ("LLaMA3-8B", "Proposed", "Alpaca", 89.6, 588, 24.2),
    ("LLaMA3-8B", "Proposed", "HumanEval", 79.5, 575, 19.1),
    ("LLaMA3-8B", "w/o DD", "GSM8K", 84.6, 698, 13.4),
    ("LLaMA3-8B", "w/o DD", "Alpaca", 87.8, 695, 21.1),
    ("LLaMA3-8B", "w/o DD", "HumanEval", 77.6, 699, 15.2),
    ("LLaMA3-8B", "w/o CDV", "GSM8K", 84.1, 730, 12.1),
    ("LLaMA3-8B", "w/o CDV", "Alpaca", 87.1, 736, 18.6),
    ("LLaMA3-8B", "w/o CDV", "HumanEval", 77.2, 747, 13.4),
    ("LLaMA3-8B", "w/o Both", "GSM8K", 81.7, 912, 9.4),
    ("LLaMA3-8B", "w/o Both", "Alpaca", 84.9, 920, 14.8),
    ("LLaMA3-8B", "w/o Both", "HumanEval", 73.9, 920, 10.2),
]


def ensure_output_dir(name: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "outputs" / "chapter4_tables" / name
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


def metric_higher_is_better(metric: str) -> bool:
    return metric in {"Acc", "Tpt"}


def best_baseline(values: dict[str, float], metric: str) -> tuple[str, float]:
    candidates = [(alg, values[alg]) for alg in BASELINES]
    if metric_higher_is_better(metric):
        return max(candidates, key=lambda item: item[1])
    return min(candidates, key=lambda item: item[1])


def improvement_ratio(proposed: float, baseline: float, metric: str) -> float:
    if metric_higher_is_better(metric):
        return (proposed - baseline) / baseline * 100
    return (baseline - proposed) / baseline * 100

