from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from chapter2_lightweighting import (
    DistillConfig,
    HardwareAwareLightweightingPipeline,
    SearchConfig,
    VehicleConstraint,
)


# 这里的 calibration_loader 和 distill_loader 需要替换成你自己的 DataLoader。
# batch 至少应包含 input_ids、attention_mask、labels。

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_model = AutoModelForCausalLM.from_pretrained("your_teacher_model_path").to(device)
    base_model = AutoModelForCausalLM.from_pretrained("your_base_model_path").to(device)
    reference_model = AutoModelForCausalLM.from_pretrained("your_reference_model_path").to(device)

    calibration_loader = None  # TODO: 替换成你的校准集 DataLoader
    distill_loader = None      # TODO: 替换成你的蒸馏集 DataLoader

    pipeline = HardwareAwareLightweightingPipeline(
        model=base_model,
        teacher_model=teacher_model,
        reference_model=reference_model,
        constraint=VehicleConstraint(
            max_memory_bytes=8 * 1024 ** 3,
            effective_flops_per_sec=2e12,
            max_latency_sec=0.8,
            max_input_len=100,
            max_output_len=1000,
            weight_bits=16,
        ),
        search_config=SearchConfig(
            population_size=24,
            iterations=20,
        ),
        distill_config=DistillConfig(
            keep_ratio=0.4,
            epochs=1,
            lr=2e-5,
        ),
        device=device,
    )

    student_model, best_solution, masks = pipeline.run(
        calibration_loader=calibration_loader,
        distill_loader=distill_loader,
        max_calibration_batches=8,
    )

    print("最优剪枝率:", best_solution.rates.as_dict())
    print("是否满足硬件约束:", best_solution.feasible)
    torch.save(student_model.state_dict(), "student_model_pruned_distilled.pt")
    torch.save(masks, "structured_masks.pt")


if __name__ == "__main__":
    main()
