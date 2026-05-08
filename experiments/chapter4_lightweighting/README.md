# Chapter 4.1 Lightweighting Experiments

This directory contains independent runners for Section 4.1.6 of the thesis.

## Plot Runners

| Runner | Source script | Output |
| --- | --- | --- |
| `run_convergence_experiment.py` | `收敛.py` | Figure 4-6 convergence curve and CSV. |
| `run_pruning_accuracy_experiment.py` | `101.py` | BoolQ/PIQA pruning accuracy plots and CSV. |
| `run_pruning_ppl_experiment.py` | `102.py` | WikiText2/PTB pruning perplexity plots and CSV. |
| `run_distillation_comparison_experiment.py` | `知识蒸馏对比.py` | Knowledge-distillation comparison plots and CSV. |

## Table Runners

| Runner | Thesis table | Output |
| --- | --- | --- |
| `run_table_4_3_hardware_constraints.py` | Table 4-3 | Raw CSV and Proposed-vs-best-baseline summary. |
| `run_table_4_4_ablation.py` | Table 4-4 | Raw CSV and ablation delta summary. |
| `run_table_4_5_kvret.py` | Table 4-5 | Raw CSV and Proposed-vs-best-baseline summary. |

## Usage

Run each experiment independently from the repository root:

```bash
python experiments/chapter4_lightweighting/run_convergence_experiment.py
python experiments/chapter4_lightweighting/run_pruning_accuracy_experiment.py
python experiments/chapter4_lightweighting/run_pruning_ppl_experiment.py
python experiments/chapter4_lightweighting/run_distillation_comparison_experiment.py
python experiments/chapter4_lightweighting/run_table_4_3_hardware_constraints.py
python experiments/chapter4_lightweighting/run_table_4_4_ablation.py
python experiments/chapter4_lightweighting/run_table_4_5_kvret.py
```

Generated artifacts are stored under:

```text
outputs/chapter4_lightweighting/
```

