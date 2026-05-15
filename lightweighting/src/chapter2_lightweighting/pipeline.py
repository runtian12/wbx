from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .adapters import LlamaStyleAdapter
from .config import CandidateSolution, DistillConfig, SearchConfig, VehicleConstraint
from .distillation import SelectiveDistiller
from .pruning import EvolutionaryRateSearcher, StructuredMaskBuilder, StructuredPruner, StructuredPruningScorer
from .resource_estimator import ResourceEstimator


class HardwareAwareLightweightingPipeline:
    """第二章完整流程封装。"""

    def __init__(
        self,
        model: nn.Module,
        teacher_model: nn.Module,
        reference_model: nn.Module,
        constraint: VehicleConstraint,
        search_config: SearchConfig,
        distill_config: DistillConfig,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.teacher_model = teacher_model.to(self.device)
        self.reference_model = reference_model.to(self.device)
        self.constraint = constraint
        self.search_config = search_config
        self.distill_config = distill_config
        self.adapter = LlamaStyleAdapter(self.model)
        self.topology = self.adapter.get_topology()
        self.estimator = ResourceEstimator(constraint)

    def run(
        self,
        calibration_loader: DataLoader,
        distill_loader: DataLoader,
        max_calibration_batches: int = 8,
    ) -> Tuple[nn.Module, CandidateSolution, Dict[str, torch.Tensor]]:
        scorer = StructuredPruningScorer(self.model, self.adapter, self.device)
        units, type_scores, coupling = scorer.compute_first_order_scores(
            calibration_loader=calibration_loader,
            max_batches=max_calibration_batches,
        )

        searcher = EvolutionaryRateSearcher(
            base_topology=self.topology,
            estimator=self.estimator,
            type_scores=type_scores,
            coupling=coupling,
            config=self.search_config,
        )
        best = searcher.search()
        print("[Search] best rates:", best.rates.as_dict())
        print("[Search] feasible:", best.feasible)
        print("[Search] pruned topology:", best.topology)
        print("[Search] est_memory(MB):", self.estimator.total_memory_bytes(best.topology) / 1024 / 1024)
        print("[Search] est_latency(ms):", self.estimator.latency_sec(best.topology) * 1000)

        mask_builder = StructuredMaskBuilder(units, self.topology)
        masks = mask_builder.build_masks(best.rates)

        pruner = StructuredPruner(self.model, self.adapter)
        student = pruner.apply_masks_in_place(masks)

        distiller = SelectiveDistiller(
            teacher_model=self.teacher_model,
            reference_model=self.reference_model,
            student_model=student,
            device=self.device,
            config=self.distill_config,
        )
        distilled_student = distiller.train(distill_loader)
        return distilled_student, best, masks
