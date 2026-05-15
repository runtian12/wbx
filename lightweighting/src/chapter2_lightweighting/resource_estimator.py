from __future__ import annotations

from .config import ModelTopology, VehicleConstraint


class ResourceEstimator:
    """
    对应论文第 2.3.1 节的工程化资源估计器。
    这里采用近似公式，便于做剪枝率搜索和可行性判断。
    """

    def __init__(self, constraint: VehicleConstraint):
        self.constraint = constraint

    def static_memory_bytes(self, topo: ModelTopology) -> float:
        b = self.constraint.weight_bits
        l = topo.num_layers
        h = topo.num_heads
        dh = topo.head_dim
        d = topo.hidden_size
        dff = topo.intermediate_size
        return 2.0 * b * l * h * dh * (d + dff) / 8.0

    def dynamic_memory_bytes(self, topo: ModelTopology) -> float:
        b = self.constraint.weight_bits
        l = topo.num_layers
        n = self.constraint.max_input_len
        o = self.constraint.max_output_len
        d = topo.hidden_size
        return 2.0 * l * b * (n + o) * d / 8.0

    def prefill_flops(self, topo: ModelTopology) -> float:
        l = topo.num_layers
        n = self.constraint.max_input_len
        d = topo.hidden_size
        dff = topo.intermediate_size
        flops_msa = 4 * n * (d ** 2) + 2 * (n ** 2) * d
        flops_ffn = 2 * n * d * dff
        return l * (flops_msa + flops_ffn)

    def decode_flops(self, topo: ModelTopology) -> float:
        l = topo.num_layers
        n = self.constraint.max_input_len
        o = self.constraint.max_output_len
        d = topo.hidden_size
        dff = topo.intermediate_size
        total = 0.0
        for step in range(1, o + 1):
            seq_len = n + step
            per_token = 4 * (d ** 2) + 2 * seq_len * d + 2 * d * dff
            total += per_token
        return l * total

    def total_memory_bytes(self, topo: ModelTopology) -> float:
        return self.static_memory_bytes(topo) + self.dynamic_memory_bytes(topo)

    def total_flops(self, topo: ModelTopology) -> float:
        return self.prefill_flops(topo) + self.decode_flops(topo)

    def latency_sec(self, topo: ModelTopology) -> float:
        return self.total_flops(topo) / max(self.constraint.effective_flops_per_sec, 1.0)

    def feasible(self, topo: ModelTopology) -> bool:
        mem_ok = self.total_memory_bytes(topo) <= self.constraint.max_memory_bytes
        lat_ok = self.latency_sec(topo) <= self.constraint.max_latency_sec
        return bool(mem_ok and lat_ok)
