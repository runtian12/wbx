from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .adapters import LlamaStyleAdapter
from .config import CandidateSolution, ModelTopology, RateVector, SearchConfig, StructuredUnit
from .resource_estimator import ResourceEstimator


class StructuredPruningScorer:
    def __init__(self, model: nn.Module, adapter: LlamaStyleAdapter, device: torch.device):
        self.model = model
        self.adapter = adapter
        self.device = device
        self.topology = adapter.get_topology()

    def build_units(self) -> List[StructuredUnit]:
        units: List[StructuredUnit] = []
        hidden = self.topology.hidden_size
        heads = self.topology.num_heads
        inter = self.topology.intermediate_size

        for idx in range(hidden):
            units.append(StructuredUnit(unit_id=f"emb_{idx}", unit_type="emb", layer_idx=-1, local_idx=idx))

        for group in self.adapter.iter_layer_groups():
            layer_idx = group["layer_idx"]
            for h in range(heads):
                units.append(StructuredUnit(unit_id=f"head_{layer_idx}_{h}", unit_type="head", layer_idx=layer_idx, local_idx=h))
            for j in range(inter):
                units.append(StructuredUnit(unit_id=f"ffn_{layer_idx}_{j}", unit_type="ffn", layer_idx=layer_idx, local_idx=j))
        return units

    @staticmethod
    def _sum_abs_wg(param: Optional[torch.Tensor], index: int, axis: int) -> float:
        if param is None or param.grad is None:
            return 0.0
        data = (param.detach() * param.grad.detach()).abs()
        if axis == 0:
            if index >= data.shape[0]:
                return 0.0
            return float(data[index].sum().item())
        if axis == 1:
            if index >= data.shape[1]:
                return 0.0
            return float(data[:, index].sum().item())
        raise ValueError("axis 只支持 0 或 1")

    def _score_emb_unit(self, emb_idx: int) -> float:
        score = 0.0
        if self.adapter.embed_tokens is not None:
            score += self._sum_abs_wg(self.adapter.embed_tokens.weight, emb_idx, axis=1)
        if self.adapter.lm_head is not None and hasattr(self.adapter.lm_head, "weight"):
            score += self._sum_abs_wg(self.adapter.lm_head.weight, emb_idx, axis=1)

        for group in self.adapter.iter_layer_groups():
            for name in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]:
                score += self._sum_abs_wg(group[name].weight, emb_idx, axis=1)
            for name in ["o_proj", "down_proj"]:
                score += self._sum_abs_wg(group[name].weight, emb_idx, axis=0)
            for ln_name in ["input_ln", "post_attn_ln"]:
                ln = group[ln_name]
                if ln is not None and hasattr(ln, "weight") and ln.weight.grad is not None and emb_idx < ln.weight.numel():
                    score += float((ln.weight.detach()[emb_idx] * ln.weight.grad.detach()[emb_idx]).abs().item())

        if self.adapter.final_norm is not None and hasattr(self.adapter.final_norm, "weight"):
            w = self.adapter.final_norm.weight
            if w.grad is not None and emb_idx < w.numel():
                score += float((w.detach()[emb_idx] * w.grad.detach()[emb_idx]).abs().item())
        return score

    def _score_head_unit(self, group: Dict[str, nn.Module], head_idx: int) -> float:
        head_dim = self.topology.head_dim
        start = head_idx * head_dim
        end = start + head_dim
        score = 0.0
        for name in ["q_proj", "k_proj", "v_proj"]:
            mod = group[name]
            if mod.weight.grad is not None:
                data = (mod.weight.detach() * mod.weight.grad.detach()).abs()
                score += float(data[start:end, :].sum().item())
        o_proj = group["o_proj"]
        if o_proj.weight.grad is not None:
            data = (o_proj.weight.detach() * o_proj.weight.grad.detach()).abs()
            score += float(data[:, start:end].sum().item())
        return score

    def _score_ffn_unit(self, group: Dict[str, nn.Module], ffn_idx: int) -> float:
        score = 0.0
        for name in ["gate_proj", "up_proj"]:
            score += self._sum_abs_wg(group[name].weight, ffn_idx, axis=0)
        score += self._sum_abs_wg(group["down_proj"].weight, ffn_idx, axis=1)
        return score

    def compute_first_order_scores(
        self,
        calibration_loader: DataLoader,
        max_batches: int = 8,
    ) -> Tuple[List[StructuredUnit], Dict[str, float], Dict[Tuple[str, str], float]]:
        self.model.train()
        units = self.build_units()
        unit_map = {u.unit_id: u for u in units}

        type_scores = {"emb": 0.0, "head": 0.0, "ffn": 0.0}
        fisher_records: List[Dict[str, float]] = []

        for step, batch in enumerate(calibration_loader):
            if step >= max_batches:
                break
            self.model.zero_grad(set_to_none=True)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            outputs.loss.backward()

            batch_type_masses = {"emb": 0.0, "head": 0.0, "ffn": 0.0}

            for emb_idx in range(self.topology.hidden_size):
                uid = f"emb_{emb_idx}"
                s = self._score_emb_unit(emb_idx)
                unit_map[uid].score += s
                type_scores["emb"] += s
                batch_type_masses["emb"] += s

            for group in self.adapter.iter_layer_groups():
                layer_idx = group["layer_idx"]
                for head_idx in range(self.topology.num_heads):
                    uid = f"head_{layer_idx}_{head_idx}"
                    s = self._score_head_unit(group, head_idx)
                    unit_map[uid].score += s
                    type_scores["head"] += s
                    batch_type_masses["head"] += s
                for ffn_idx in range(self.topology.intermediate_size):
                    uid = f"ffn_{layer_idx}_{ffn_idx}"
                    s = self._score_ffn_unit(group, ffn_idx)
                    unit_map[uid].score += s
                    type_scores["ffn"] += s
                    batch_type_masses["ffn"] += s

            fisher_records.append(batch_type_masses)

        coupling = self._estimate_type_coupling(fisher_records)
        return list(unit_map.values()), type_scores, coupling

    @staticmethod
    def _estimate_type_coupling(records: List[Dict[str, float]]) -> Dict[Tuple[str, str], float]:
        types = ["emb", "head", "ffn"]
        coupling: Dict[Tuple[str, str], float] = {}
        if not records:
            for u in types:
                for v in types:
                    coupling[(u, v)] = 0.0
            return coupling

        for u in types:
            for v in types:
                vals = [rec[u] * rec[v] for rec in records]
                coupling[(u, v)] = float(sum(vals) / len(vals))
        return coupling


class EvolutionaryRateSearcher:
    def __init__(
        self,
        base_topology: ModelTopology,
        estimator: ResourceEstimator,
        type_scores: Dict[str, float],
        coupling: Dict[Tuple[str, str], float],
        config: SearchConfig,
    ):
        self.base_topology = base_topology
        self.estimator = estimator
        self.type_scores = type_scores
        self.coupling = coupling
        self.config = config
        random.seed(config.seed)

    def _clip_rate(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _random_rate(self) -> RateVector:
        b = self.config.bounds
        return RateVector(
            rho_emb=random.uniform(b.emb_min, b.emb_max),
            rho_head=random.uniform(b.head_min, b.head_max),
            rho_ffn=random.uniform(b.ffn_min, b.ffn_max),
        )

    def _mutate(self, rate: RateVector) -> RateVector:
        b = self.config.bounds
        vals = [rate.rho_emb, rate.rho_head, rate.rho_ffn]
        bounds = [(b.emb_min, b.emb_max), (b.head_min, b.head_max), (b.ffn_min, b.ffn_max)]
        new_vals = []
        for value, (lo, hi) in zip(vals, bounds):
            if random.random() < self.config.mutation_prob:
                value += random.gauss(0.0, self.config.mutation_scale)
            new_vals.append(self._clip_rate(value, lo, hi))
        return RateVector(*new_vals)

    @staticmethod
    def _crossover(a: RateVector, b: RateVector) -> RateVector:
        return RateVector(
            rho_emb=(a.rho_emb + b.rho_emb) / 2.0,
            rho_head=(a.rho_head + b.rho_head) / 2.0,
            rho_ffn=(a.rho_ffn + b.rho_ffn) / 2.0,
        )

    def _topology_from_rate(self, rate: RateVector) -> ModelTopology:
        hidden = max(8, int(round(self.base_topology.hidden_size * (1.0 - rate.rho_emb))))
        num_heads = max(1, int(round(self.base_topology.num_heads * (1.0 - rate.rho_head))))
        hidden = max(num_heads, (hidden // num_heads) * num_heads)
        if hidden == 0:
            hidden = num_heads
        inter = max(8, int(round(self.base_topology.intermediate_size * (1.0 - rate.rho_ffn))))
        return ModelTopology(
            num_layers=self.base_topology.num_layers,
            hidden_size=hidden,
            intermediate_size=inter,
            num_heads=num_heads,
        )

    def _objective(self, rate: RateVector) -> CandidateSolution:
        rho = rate.as_dict()
        topology = self._topology_from_rate(rate)
        feasible = self.estimator.feasible(topology)

        linear_term = sum(rho[t] * self.type_scores[t] for t in ["emb", "head", "ffn"])
        quad_term = 0.0
        for u in ["emb", "head", "ffn"]:
            for v in ["emb", "head", "ffn"]:
                quad_term += 0.5 * rho[u] * rho[v] * self.coupling[(u, v)]

        penalty = 0.0
        if not feasible:
            mem = self.estimator.total_memory_bytes(topology)
            lat = self.estimator.latency_sec(topology)
            penalty += max(0.0, mem - self.estimator.constraint.max_memory_bytes) * 1e-6
            penalty += max(0.0, lat - self.estimator.constraint.max_latency_sec) * 1e6
        obj = linear_term + quad_term + penalty
        return CandidateSolution(rate, obj, feasible, topology)

    def search(self) -> CandidateSolution:
        population = [self._random_rate() for _ in range(self.config.population_size)]
        best: Optional[CandidateSolution] = None

        for _ in range(self.config.iterations):
            scored = [self._objective(ind) for ind in population]
            scored.sort(key=lambda x: x.objective)
            if best is None or scored[0].objective < best.objective:
                best = scored[0]

            elite_num = max(2, int(self.config.population_size * self.config.elite_ratio))
            elites = [cand.rates for cand in scored[:elite_num]]

            new_population = elites.copy()
            while len(new_population) < self.config.population_size:
                p1, p2 = random.sample(elites, 2)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_population.append(child)
            population = new_population

        assert best is not None
        return best


class StructuredMaskBuilder:
    def __init__(self, units: List[StructuredUnit], topology: ModelTopology):
        self.units = units
        self.topology = topology

    @staticmethod
    def _keep_topk(units: List[StructuredUnit], keep_num: int) -> List[StructuredUnit]:
        units = sorted(units, key=lambda x: x.score, reverse=True)
        return units[:keep_num]

    def build_masks(self, rate: RateVector) -> Dict[str, torch.Tensor]:
        emb_units = [u for u in self.units if u.unit_type == "emb"]
        head_units = [u for u in self.units if u.unit_type == "head"]
        ffn_units = [u for u in self.units if u.unit_type == "ffn"]

        keep_emb = max(1, int(round(len(emb_units) * (1.0 - rate.rho_emb))))
        keep_head = max(1, int(round(len(head_units) * (1.0 - rate.rho_head))))
        keep_ffn = max(1, int(round(len(ffn_units) * (1.0 - rate.rho_ffn))))

        emb_mask = torch.zeros(self.topology.hidden_size, dtype=torch.bool)
        head_mask = torch.zeros(self.topology.num_layers, self.topology.num_heads, dtype=torch.bool)
        ffn_mask = torch.zeros(self.topology.num_layers, self.topology.intermediate_size, dtype=torch.bool)

        for u in self._keep_topk(emb_units, keep_emb):
            emb_mask[u.local_idx] = True
        for u in self._keep_topk(head_units, keep_head):
            head_mask[u.layer_idx, u.local_idx] = True
        for u in self._keep_topk(ffn_units, keep_ffn):
            ffn_mask[u.layer_idx, u.local_idx] = True

        return {"emb": emb_mask, "head": head_mask, "ffn": ffn_mask}


class StructuredPruner:
    def __init__(self, model: nn.Module, adapter: LlamaStyleAdapter):
        self.model = model
        self.adapter = adapter
        self.topology = adapter.get_topology()

    @staticmethod
    def _zero_linear_rows(linear: nn.Linear, indices: torch.Tensor):
        if indices.numel() == 0:
            return
        linear.weight.data[indices, :] = 0
        if linear.bias is not None and linear.bias.numel() == linear.out_features:
            linear.bias.data[indices] = 0

    @staticmethod
    def _zero_linear_cols(linear: nn.Linear, indices: torch.Tensor):
        if indices.numel() == 0:
            return
        linear.weight.data[:, indices] = 0

    @staticmethod
    def _zero_norm_indices(norm: nn.Module, indices: torch.Tensor):
        if norm is None or not hasattr(norm, "weight") or indices.numel() == 0:
            return
        norm.weight.data[indices] = 0
        if hasattr(norm, "bias") and norm.bias is not None:
            norm.bias.data[indices] = 0

    def apply_masks_in_place(self, masks: Dict[str, torch.Tensor]) -> nn.Module:
        emb_drop = (~masks["emb"]).nonzero(as_tuple=False).flatten()
        if self.adapter.embed_tokens is not None:
            self.adapter.embed_tokens.weight.data[:, emb_drop] = 0
        if self.adapter.lm_head is not None and hasattr(self.adapter.lm_head, "weight"):
            self.adapter.lm_head.weight.data[:, emb_drop] = 0
        if self.adapter.final_norm is not None:
            self._zero_norm_indices(self.adapter.final_norm, emb_drop)

        head_dim = self.topology.head_dim
        for group in self.adapter.iter_layer_groups():
            layer_idx = group["layer_idx"]

            for name in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]:
                self._zero_linear_cols(group[name], emb_drop)
            for name in ["o_proj", "down_proj"]:
                self._zero_linear_rows(group[name], emb_drop)
            self._zero_norm_indices(group["input_ln"], emb_drop)
            self._zero_norm_indices(group["post_attn_ln"], emb_drop)

            dropped_heads = (~masks["head"][layer_idx]).nonzero(as_tuple=False).flatten().tolist()
            if dropped_heads:
                row_indices = []
                for h in dropped_heads:
                    row_indices.extend(list(range(h * head_dim, (h + 1) * head_dim)))
                row_indices = torch.tensor(row_indices, dtype=torch.long, device=group["q_proj"].weight.device)
                for name in ["q_proj", "k_proj", "v_proj"]:
                    self._zero_linear_rows(group[name], row_indices)
                self._zero_linear_cols(group["o_proj"], row_indices)

            dropped_ffn = (~masks["ffn"][layer_idx]).nonzero(as_tuple=False).flatten().to(group["gate_proj"].weight.device)
            self._zero_linear_rows(group["gate_proj"], dropped_ffn)
            self._zero_linear_rows(group["up_proj"], dropped_ffn)
            self._zero_linear_cols(group["down_proj"], dropped_ffn)
        return self.model
