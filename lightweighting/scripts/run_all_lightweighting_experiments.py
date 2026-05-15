from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNERS = [
    PROJECT_ROOT / "experiments" / "chapter4_lightweighting" / "run_convergence_experiment.py",
    PROJECT_ROOT / "experiments" / "chapter4_lightweighting" / "run_pruning_accuracy_experiment.py",
    PROJECT_ROOT / "experiments" / "chapter4_lightweighting" / "run_pruning_ppl_experiment.py",
    PROJECT_ROOT / "experiments" / "chapter4_lightweighting" / "run_distillation_comparison_experiment.py",
    PROJECT_ROOT / "experiments" / "chapter4_lightweighting" / "run_table_4_3_hardware_constraints.py",
    PROJECT_ROOT / "experiments" / "chapter4_lightweighting" / "run_table_4_4_ablation.py",
    PROJECT_ROOT / "experiments" / "chapter4_lightweighting" / "run_table_4_5_kvret.py",
]


def main() -> None:
    for runner in RUNNERS:
        print(f"[Run] {runner.name}")
        subprocess.check_call([sys.executable, str(runner)], cwd=str(PROJECT_ROOT))
    print("All lightweighting experiments finished.")


if __name__ == "__main__":
    main()

