from __future__ import annotations

import torch

from chapter2_lightweighting import (
    DistillConfig,
    HardwareAwareLightweightingPipeline,
    SearchConfig,
    VehicleConstraint,
)
from chapter2_lightweighting.data_utils import build_demo_causal_lm_loader

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise RuntimeError("当前环境缺少 transformers，请先安装。") from exc


def main(model_name: str = "hf-internal-testing/tiny-random-LlamaForCausalLM"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    reference = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    texts = [
        "请介绍一下车云协同推理的基本思想。",
        "结构化剪枝和知识蒸馏之间有什么关系？",
        "智能座舱中的大语言模型部署为什么困难？",
        "请解释一下多头注意力和前馈网络的作用。",
    ]

    calib_loader = build_demo_causal_lm_loader(tokenizer, texts, max_length=64, batch_size=2)
    distill_loader = build_demo_causal_lm_loader(tokenizer, texts * 4, max_length=64, batch_size=2)

    pipeline = HardwareAwareLightweightingPipeline(
        model=base_model,
        teacher_model=teacher,
        reference_model=reference,
        constraint=VehicleConstraint(
            max_memory_bytes=8 * 1024 ** 3,
            effective_flops_per_sec=2e12,
            max_latency_sec=0.8,
            max_input_len=100,
            max_output_len=1000,
            weight_bits=16,
        ),
        search_config=SearchConfig(population_size=8, iterations=4),
        distill_config=DistillConfig(keep_ratio=0.4, epochs=1, lr=1e-5),
        device=device,
    )

    _, best, _ = pipeline.run(
        calibration_loader=calib_loader,
        distill_loader=distill_loader,
        max_calibration_batches=2,
    )
    print("Demo finished. Best rates:", best.rates.as_dict())


if __name__ == "__main__":
    main()
