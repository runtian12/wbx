from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .config import CandidateSolution, ModelTopology
from .resource_estimator import ResourceEstimator


def build_resource_summary(estimator: ResourceEstimator, topology: ModelTopology) -> Dict[str, float]:
    return {
        "num_layers": float(topology.num_layers),
        "hidden_size": float(topology.hidden_size),
        "intermediate_size": float(topology.intermediate_size),
        "num_heads": float(topology.num_heads),
        "static_memory_mb": estimator.static_memory_bytes(topology) / 1024.0 / 1024.0,
        "dynamic_memory_mb": estimator.dynamic_memory_bytes(topology) / 1024.0 / 1024.0,
        "total_memory_mb": estimator.total_memory_bytes(topology) / 1024.0 / 1024.0,
        "total_flops": estimator.total_flops(topology),
        "latency_ms": estimator.latency_sec(topology) * 1000.0,
        "feasible": float(estimator.feasible(topology)),
    }


def build_solution_report(
    estimator: ResourceEstimator,
    base_topology: ModelTopology,
    best_solution: CandidateSolution,
    parameter_summary: Dict[str, Any],
    eval_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "base_topology": build_resource_summary(estimator, base_topology),
        "pruned_topology": build_resource_summary(estimator, best_solution.topology),
        "best_rates": best_solution.rates.as_dict(),
        "objective": best_solution.objective,
        "feasible": best_solution.feasible,
        "parameters": parameter_summary,
        "evaluation": eval_summary,
    }


def save_json_report(report: Dict[str, Any], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def save_markdown_report(report: Dict[str, Any], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Lightweighting Experiment Report",
        "",
        "## Best Pruning Rates",
        "",
    ]
    for key, value in report["best_rates"].items():
        lines.append(f"- `{key}`: {value:.4f}")
    lines.extend([
        "",
        "## Resource Estimate",
        "",
        "| Metric | Base | Pruned |",
        "| --- | ---: | ---: |",
    ])
    base = report["base_topology"]
    pruned = report["pruned_topology"]
    for key in ["total_memory_mb", "latency_ms", "total_flops", "feasible"]:
        lines.append(f"| `{key}` | {base[key]:.4f} | {pruned[key]:.4f} |")
    lines.extend([
        "",
        "## Parameter Summary",
        "",
    ])
    for key, value in report["parameters"].items():
        if isinstance(value, float):
            lines.append(f"- `{key}`: {value:.6f}")
        else:
            lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## Evaluation",
        "",
    ])
    for key, value in report["evaluation"].items():
        if isinstance(value, float):
            lines.append(f"- `{key}`: {value:.6f}")
        else:
            lines.append(f"- `{key}`: {value}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

