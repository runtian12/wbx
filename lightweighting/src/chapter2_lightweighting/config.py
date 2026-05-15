from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class VehicleConstraint:
    """车端资源约束。"""

    max_memory_bytes: float
    effective_flops_per_sec: float
    max_latency_sec: float
    max_input_len: int = 100
    max_output_len: int = 1000
    weight_bits: int = 16


@dataclass
class SearchBounds:
    """三类组件剪枝率边界。"""

    emb_min: float = 0.0
    emb_max: float = 0.5
    head_min: float = 0.0
    head_max: float = 0.5
    ffn_min: float = 0.0
    ffn_max: float = 0.5


@dataclass
class SearchConfig:
    """进化搜索参数。"""

    population_size: int = 24
    iterations: int = 20
    elite_ratio: float = 0.25
    mutation_prob: float = 0.3
    mutation_scale: float = 0.08
    seed: int = 42
    bounds: SearchBounds = field(default_factory=SearchBounds)


@dataclass
class DistillConfig:
    """选择性知识蒸馏参数。"""

    keep_ratio: float = 0.4
    epochs: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.01
    temperature: float = 1.0
    grad_clip: float = 1.0
    log_every: int = 20


@dataclass
class ModelTopology:
    """剪枝前/后的拓扑描述。"""

    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass
class RateVector:
    rho_emb: float
    rho_head: float
    rho_ffn: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "emb": self.rho_emb,
            "head": self.rho_head,
            "ffn": self.rho_ffn,
        }


@dataclass
class CandidateSolution:
    rates: RateVector
    objective: float
    feasible: bool
    topology: ModelTopology


@dataclass
class StructuredUnit:
    """一个可剪枝的结构化单元。"""

    unit_id: str
    unit_type: str
    layer_idx: int
    local_idx: int
    score: float = 0.0
