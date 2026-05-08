"""Reproduce Table 4-3: performance under different hardware constraints."""

from __future__ import annotations

from collections import defaultdict

from table_data import BASELINES, PRUNING_RATES, TABLE_4_3, ensure_output_dir, higher_is_better, improvement, write_csv, write_json, write_text


def main() -> None:
    output_dir = ensure_output_dir("table_4_3_hardware_constraints")

    raw_rows = []
    grouped = defaultdict(dict)
    for hardware, algorithm, metric, values in TABLE_4_3:
        for rate, value in zip(PRUNING_RATES, values):
            raw_rows.append([hardware, algorithm, metric, rate, value])
            grouped[(hardware, metric, rate)][algorithm] = value

    summary_rows = []
    for (hardware, metric, rate), values in sorted(grouped.items()):
        baseline_items = [(algorithm, values[algorithm]) for algorithm in BASELINES]
        if higher_is_better(metric):
            best_algorithm, best_value = max(baseline_items, key=lambda item: item[1])
        else:
            best_algorithm, best_value = min(baseline_items, key=lambda item: item[1])
        proposed = values["Proposed"]
        summary_rows.append(
            [hardware, metric, rate, best_algorithm, best_value, proposed, round(improvement(proposed, best_value, metric), 2)]
        )

    files = [
        write_csv(output_dir / "table_4_3_raw.csv", ["hardware", "algorithm", "metric", "pruning_rate_percent", "value"], raw_rows),
        write_csv(
            output_dir / "table_4_3_proposed_vs_best_baseline.csv",
            ["hardware", "metric", "pruning_rate_percent", "best_baseline", "best_baseline_value", "proposed", "improvement_percent"],
            summary_rows,
        ),
        write_json(output_dir / "table_4_3_summary.json", summary_rows),
    ]

    avg_improvement = sum(row[-1] for row in summary_rows) / len(summary_rows)
    files.append(
        write_text(
            output_dir / "summary.txt",
            [
                "Table 4-3 reproduction: hardware resource constraints.",
                f"Compared rows: {len(summary_rows)}",
                f"Average Proposed improvement over best baseline: {avg_improvement:.2f}%",
            ],
        )
    )

    print("Table 4-3 experiment completed.")
    for file in files:
        print(f"  - {file.resolve()}")


if __name__ == "__main__":
    main()

