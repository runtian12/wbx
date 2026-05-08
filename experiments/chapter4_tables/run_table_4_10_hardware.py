"""Reproduce Table 4-10: performance across vehicle hardware platforms."""

from __future__ import annotations

from collections import defaultdict

from table_data import BASELINES, TABLE_4_10, ensure_output_dir, improvement_ratio, write_csv, write_json, write_text


def main() -> None:
    output_dir = ensure_output_dir("table_4_10_hardware")

    raw_rows = [[hardware, alg, acc, lat, tpt] for hardware, alg, acc, lat, tpt in TABLE_4_10]
    grouped = defaultdict(dict)
    for hardware, alg, acc, lat, tpt in TABLE_4_10:
        grouped[hardware][alg] = {"Acc": acc, "Lat": lat, "Tpt": tpt}

    summary_rows = []
    for hardware, values in grouped.items():
        for metric in ["Acc", "Lat", "Tpt"]:
            baseline_items = [(alg, values[alg][metric]) for alg in BASELINES]
            if metric == "Lat":
                best_alg, best_value = min(baseline_items, key=lambda item: item[1])
            else:
                best_alg, best_value = max(baseline_items, key=lambda item: item[1])
            proposed = values["Proposed"][metric]
            summary_rows.append(
                [hardware, metric, best_alg, best_value, proposed, round(improvement_ratio(proposed, best_value, metric), 2)]
            )

    files = [
        write_csv(output_dir / "table_4_10_raw.csv", ["hardware", "algorithm", "accuracy_percent", "latency_ms", "throughput_tokens_s"], raw_rows),
        write_csv(
            output_dir / "table_4_10_proposed_vs_best_baseline.csv",
            ["hardware", "metric", "best_baseline", "best_baseline_value", "proposed", "improvement_percent"],
            summary_rows,
        ),
        write_json(output_dir / "table_4_10_summary.json", summary_rows),
    ]

    files.append(
        write_text(
            output_dir / "summary.txt",
            [
                "Table 4-10 reproduction: hardware-platform performance.",
                f"Hardware platforms: {len(grouped)}",
                f"Compared metric rows: {len(summary_rows)}",
            ],
        )
    )

    print("Table 4-10 experiment completed.")
    for file in files:
        print(f"  - {file.resolve()}")


if __name__ == "__main__":
    main()

