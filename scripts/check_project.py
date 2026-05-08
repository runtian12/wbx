"""Check that the thesis-code repository has the files needed for reproduction.

This script intentionally avoids importing heavy ML packages. It verifies the
directory layout and reports optional dependencies if they are already installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "REPRODUCIBILITY.md",
    "requirements-common.txt",
    "experiments/chapter2_plots/202.py",
    "experiments/chapter2_plots/204.py",
    "experiments/chapter2_plots/run_bandwidth_experiment.py",
    "experiments/chapter2_plots/run_hardware_plot_experiment.py",
    "experiments/chapter2_plots/run_chapter2_experiments.py",
    "experiments/chapter2_plots/requirements.txt",
    "experiments/chapter2_plots/README.md",
    "experiments/chapter4_tables/table_data.py",
    "experiments/chapter4_tables/run_table_4_8_model_dataset.py",
    "experiments/chapter4_tables/run_table_4_9_threshold.py",
    "experiments/chapter4_tables/run_table_4_10_hardware.py",
    "experiments/chapter4_tables/run_table_4_11_ablation.py",
    "experiments/chapter4_tables/README.md",
    "chapter2_project/README.md",
    "chapter2_project/chapter2_project/run_demo.py",
    "chapter2_project/chapter2_project/requirements.txt",
    "chapter2_project/chapter2_project/chapter2_lightweighting/pipeline.py",
    "egasd/README.md",
    "egasd/example_usage.py",
    "egasd/train_acceptance_head.py",
    "egasd/data/README.md",
    "egasd/data/sample_train.json",
    "PAD-main/README.md",
    "PAD-main/setup.md",
    "PAD-main/.env.example",
    "PAD-main/requirements.txt",
    "PAD-main/requirements-vllm.txt",
    "PAD-main/data/README.md",
    "PAD-main/dataset_generation.py",
    "PAD-main/train_classifier.py",
    "PAD-main/decode.py",
    "SpecDec_pp-main/README.md",
    "SpecDec_pp-main/requirements.txt",
    "SpecDec_pp-main/specdec_pp/evaluate.py",
]

OPTIONAL_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "numpy",
    "tqdm",
    "sklearn",
    "wandb",
    "vllm",
]


def has_package(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> int:
    print(f"Repository root: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")

    missing = []
    for rel_path in REQUIRED_PATHS:
        path = ROOT / rel_path
        if not path.exists():
            missing.append(rel_path)

    if missing:
        print("\nMissing required files:")
        for rel_path in missing:
            print(f"  - {rel_path}")
    else:
        print("\nRequired files: OK")

    print("\nOptional Python packages:")
    for package in OPTIONAL_PACKAGES:
        status = "installed" if has_package(package) else "not installed"
        print(f"  - {package}: {status}")

    print("\nLarge artifacts are intentionally excluded from Git:")
    print("  - model checkpoints (*.pt, *.pth, *.safetensors)")
    print("  - generated datasets")
    print("  - output/results/logs/wandb directories")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
