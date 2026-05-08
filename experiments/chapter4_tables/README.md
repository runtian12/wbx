# Chapter 4.2 Table Experiments

This directory contains one independent runner for each experimental table in Section 4.2 of the thesis.

## Runners

| Runner | Thesis table | Output |
| --- | --- | --- |
| `run_table_4_8_model_dataset.py` | Table 4-8 | Raw CSV and Proposed-vs-best-baseline summary. |
| `run_table_4_9_threshold.py` | Table 4-9 | Raw CSV and threshold trend summary. |
| `run_table_4_10_hardware.py` | Table 4-10 | Raw CSV and hardware-platform comparison summary. |
| `run_table_4_11_ablation.py` | Table 4-11 | Raw CSV and ablation delta summary. |

## Usage

Run each experiment separately from this directory or from PyCharm:

```bash
python experiments/chapter4_tables/run_table_4_8_model_dataset.py
python experiments/chapter4_tables/run_table_4_9_threshold.py
python experiments/chapter4_tables/run_table_4_10_hardware.py
python experiments/chapter4_tables/run_table_4_11_ablation.py
```

Generated artifacts are stored under:

```text
outputs/chapter4_tables/
```

