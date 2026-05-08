# Chapter 2 Plotting Experiments

This directory binds the uploaded thesis codebase with the two plotting scripts used for Chapter 2 experimental results.

## Files

| File | Purpose |
| --- | --- |
| `202.py` | Generates bandwidth-constrained inference figures. |
| `204.py` | Generates hardware-platform inference figures. |
| `run_chapter2_experiments.py` | Runs `202.py` and `204.py` in sequence and records a reproducible log. |
| `requirements.txt` | Minimal dependencies for the plotting scripts. |

## Run

From the repository root:

```bash
pip install -r experiments/chapter2_plots/requirements.txt
python experiments/chapter2_plots/run_chapter2_experiments.py
```

Outputs are written to:

```text
outputs/chapter2_plots/
├── bandwidth/
│   ├── bw_accuracy_bar.png
│   ├── bw_latency_bar.png
│   ├── bw_throughput_bar.png
│   └── bandwidth_metrics.csv
├── hardware/
│   ├── hardware_accuracy.png
│   ├── hardware_latency.png
│   ├── hardware_throughput.png
│   └── hardware_metrics.csv
└── run_log.json
```

The `outputs/` directory is ignored by Git because these files are generated artifacts.

