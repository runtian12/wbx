"""Reproduce Table 4-4: ablation study for the lightweighting method."""

from __future__ import annotations

from collections import defaultdict

from table_data import PRUNING_RATES, TABLE_4_4, ensure_output_dir, higher_is_better, write_csv, write_json, write_text


def main() -> None:
    output_dir = ensure_output_dir("table_4_4_ablation")

    raw_rows = []
    grouped = defaultdict(dict)
    for model, variant, metric, dataset, values in TABLE_4_4:
        for rate, value in zip(PRUNING_RATES, values):
            raw_rows.append([model, variant, metric, dataset, rate, value])
            grouped[(model, dataset, metric, rate)][variant] = value

    summary_rows = []
    for (model, dataset, metric, rate), values in sorted(grouped.items()):
        proposed = values["Proposed"]
        for variant, value in values.items():
            if variant == "Proposed":
                continue
            if higher_is_better(metric):
                delta = proposed - value
            else:
                delta = value - proposed
            summary_rows.append([model, dataset, metric, rate, variant, value, proposed, round(delta, 2)])

    files = [
        write_csv(output_dir / "table_4_4_raw.csv", ["model", "variant", "metric", "dataset", "pruning_rate_percent", "value"], raw_rows),
        write_csv(
            output_dir / "table_4_4_ablation_delta.csv",
            ["model", "dataset", "metric", "pruning_rate_percent", "removed_module", "variant_value", "proposed", "proposed_gain"],
            summary_rows,
        ),
        write_json(output_dir / "table_4_4_summary.json", summary_rows),
    ]

    avg_gain = sum(row[-1] for row in summary_rows) / len(summary_rows)
    files.append(
        write_text(
            output_dir / "summary.txt",
            [
                "Table 4-4 reproduction: lightweighting ablation study.",
                f"Ablation comparison rows: {len(summary_rows)}",
                f"Average Proposed gain over ablated variants: {avg_gain:.2f}",
            ],
        )
    )

    print("Table 4-4 experiment completed.")
    for file in files:
        print(f"  - {file.resolve()}")


if __name__ == "__main__":
    main()

