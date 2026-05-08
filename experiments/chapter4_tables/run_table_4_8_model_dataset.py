"""Reproduce Table 4-8: performance across models and datasets."""

from __future__ import annotations

from table_data import ALGORITHMS, TABLE_4_8, best_baseline, ensure_output_dir, improvement_ratio, write_csv, write_json, write_text


def main() -> None:
    output_dir = ensure_output_dir("table_4_8_model_dataset")

    raw_rows = []
    summary_rows = []
    for model, dataset, metric, values in TABLE_4_8:
        raw_rows.append([model, dataset, metric] + [values[alg] for alg in ALGORITHMS])
        baseline_alg, baseline_value = best_baseline(values, metric)
        proposed = values["Proposed"]
        summary_rows.append(
            [
                model,
                dataset,
                metric,
                baseline_alg,
                baseline_value,
                proposed,
                round(improvement_ratio(proposed, baseline_value, metric), 2),
            ]
        )

    files = [
        write_csv(output_dir / "table_4_8_raw.csv", ["model", "dataset", "metric"] + ALGORITHMS, raw_rows),
        write_csv(
            output_dir / "table_4_8_proposed_vs_best_baseline.csv",
            ["model", "dataset", "metric", "best_baseline", "best_baseline_value", "proposed", "improvement_percent"],
            summary_rows,
        ),
        write_json(output_dir / "table_4_8_summary.json", summary_rows),
    ]

    avg_improvement = sum(row[-1] for row in summary_rows) / len(summary_rows)
    files.append(
        write_text(
            output_dir / "summary.txt",
            [
                "Table 4-8 reproduction: model and dataset performance.",
                f"Compared rows: {len(summary_rows)}",
                f"Average Proposed improvement over the best baseline: {avg_improvement:.2f}%",
            ],
        )
    )

    print("Table 4-8 experiment completed.")
    for file in files:
        print(f"  - {file.resolve()}")


if __name__ == "__main__":
    main()

