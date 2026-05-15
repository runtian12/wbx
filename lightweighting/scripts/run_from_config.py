from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from chapter2_lightweighting.config_loader import load_run_config
from chapter2_lightweighting.data_utils import build_text_file_causal_lm_loader
from chapter2_lightweighting.metrics import count_nonzero_parameters, count_parameters, evaluate_causal_lm_loss, parameter_sparsity
from chapter2_lightweighting.pipeline import HardwareAwareLightweightingPipeline
from chapter2_lightweighting.reporting import build_solution_report, save_json_report, save_markdown_report

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise RuntimeError("当前环境缺少 transformers，请先执行 pip install -r requirements.txt。") from exc


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hardware-aware model lightweighting from a JSON config.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "demo_cpu.json"), help="Path to JSON config.")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    set_seed(cfg.runtime.seed)
    output_dir = Path(cfg.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.runtime.device)
    tokenizer_name = cfg.model.tokenizer or cfg.model.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=cfg.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.teacher_model,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.reference_model,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    calib_loader = build_text_file_causal_lm_loader(
        tokenizer=tokenizer,
        path=cfg.data.calibration_texts_path,
        max_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
    )
    distill_loader = build_text_file_causal_lm_loader(
        tokenizer=tokenizer,
        path=cfg.data.distill_texts_path,
        max_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
    )

    before_total = count_parameters(base_model)
    before_nonzero = count_nonzero_parameters(base_model)

    pipeline = HardwareAwareLightweightingPipeline(
        model=base_model,
        teacher_model=teacher_model,
        reference_model=reference_model,
        constraint=cfg.vehicle,
        search_config=cfg.search,
        distill_config=cfg.distill,
        device=device,
    )

    student_model, best_solution, masks = pipeline.run(
        calibration_loader=calib_loader,
        distill_loader=distill_loader,
        max_calibration_batches=cfg.runtime.max_calibration_batches,
    )

    eval_summary = evaluate_causal_lm_loss(
        model=student_model,
        loader=distill_loader,
        device=device,
        max_batches=cfg.runtime.max_calibration_batches,
    )
    parameter_summary = {
        "total_parameters": before_total,
        "nonzero_parameters_before": before_nonzero,
        "nonzero_parameters_after": count_nonzero_parameters(student_model),
        "parameter_sparsity_after": parameter_sparsity(student_model),
    }
    report = build_solution_report(
        estimator=pipeline.estimator,
        base_topology=pipeline.topology,
        best_solution=best_solution,
        parameter_summary=parameter_summary,
        eval_summary=eval_summary,
    )

    if cfg.output.save_model_state:
        torch.save(student_model.state_dict(), output_dir / "student_pruned_distilled_state.pt")
    if cfg.output.save_masks:
        torch.save(masks, output_dir / "structured_masks.pt")
    if cfg.output.save_report:
        save_json_report(report, str(output_dir / "report.json"))
        save_markdown_report(report, str(output_dir / "report.md"))

    print(f"Lightweighting finished. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
