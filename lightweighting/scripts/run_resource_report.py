from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from chapter2_lightweighting.config_loader import load_run_config
from chapter2_lightweighting.config import ModelTopology, RateVector
from chapter2_lightweighting.resource_estimator import ResourceEstimator
from chapter2_lightweighting.reporting import build_resource_summary, save_json_report


def topology_from_rate(base: ModelTopology, rate: RateVector) -> ModelTopology:
    hidden = max(8, int(round(base.hidden_size * (1.0 - rate.rho_emb))))
    num_heads = max(1, int(round(base.num_heads * (1.0 - rate.rho_head))))
    hidden = max(num_heads, (hidden // num_heads) * num_heads)
    inter = max(8, int(round(base.intermediate_size * (1.0 - rate.rho_ffn))))
    return ModelTopology(
        num_layers=base.num_layers,
        hidden_size=hidden,
        intermediate_size=inter,
        num_heads=num_heads,
    )


def search_feasible_rate(base: ModelTopology, estimator: ResourceEstimator, cfg) -> RateVector:
    random.seed(cfg.search.seed)
    bounds = cfg.search.bounds
    best_rate = RateVector(bounds.emb_min, bounds.head_min, bounds.ffn_min)
    best_score = float("inf")
    for _ in range(max(8, cfg.search.population_size * cfg.search.iterations)):
        rate = RateVector(
            rho_emb=random.uniform(bounds.emb_min, bounds.emb_max),
            rho_head=random.uniform(bounds.head_min, bounds.head_max),
            rho_ffn=random.uniform(bounds.ffn_min, bounds.ffn_max),
        )
        topo = topology_from_rate(base, rate)
        mem = estimator.total_memory_bytes(topo)
        latency = estimator.latency_sec(topo)
        violation = max(0.0, mem - cfg.vehicle.max_memory_bytes) / max(cfg.vehicle.max_memory_bytes, 1.0)
        violation += max(0.0, latency - cfg.vehicle.max_latency_sec) / max(cfg.vehicle.max_latency_sec, 1.0)
        compression = rate.rho_emb + rate.rho_head + rate.rho_ffn
        score = violation * 100.0 + compression
        if estimator.feasible(topo):
            score = compression
        if score < best_score:
            best_score = score
            best_rate = rate
    return best_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a resource report from config without loading model weights.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "demo_cpu.json"))
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    if cfg.model_topology is None:
        raise ValueError("配置文件必须提供 model_topology，才能在不加载模型权重的情况下生成资源报告。")

    estimator = ResourceEstimator(cfg.vehicle)
    best_rate = search_feasible_rate(cfg.model_topology, estimator, cfg)
    best_topology = topology_from_rate(cfg.model_topology, best_rate)
    report = {
        "baseline": build_resource_summary(estimator, cfg.model_topology),
        "searched_solution": {
            "rates": best_rate.as_dict(),
            "topology": build_resource_summary(estimator, best_topology),
            "feasible": estimator.feasible(best_topology),
        },
        "manual_20_percent": {
            "rates": {"emb": 0.2, "head": 0.2, "ffn": 0.2},
            "topology": build_resource_summary(
                estimator,
                ModelTopology(
                    num_layers=cfg.model_topology.num_layers,
                    hidden_size=max(1, int(cfg.model_topology.hidden_size * 0.8)),
                    intermediate_size=max(1, int(cfg.model_topology.intermediate_size * 0.8)),
                    num_heads=max(1, int(cfg.model_topology.num_heads * 0.8)),
                ),
            ),
        },
    }
    output_dir = Path(cfg.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json_report(report, str(output_dir / "resource_report.json"))
    print(f"Resource report saved to: {output_dir / 'resource_report.json'}")


if __name__ == "__main__":
    main()
