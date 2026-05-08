"""Reproduce Table 4-11: ablation study across models and datasets."""

from __future__ import annotations

from collections import defaultdict

from table_data import TABLE_4_11, ensure_output_dir, write_csv, write_json, write_text


def main() -> None:
    output_dir = ensure_output_dir("table_4_11_ablation")

    raw_rows = [[model, variant, dataset, acc, lat, tpt] for model, variant, dataset, acc, lat, tpt in TABLE_4_11]
    grouped = defaultdict(dict)
    for model, variant, dataset, acc, lat, tpt in TABLE_4_11:
        grouped[(model, dataset)][variant] = {"Acc": acc, "Lat": lat, "Tpt": tpt}

    summary_rows = []
    for (model, dataset), variants in grouped.items():
        proposed = variants["Proposed"]
        for variant, values in variants.items():
            if variant == "Proposed":
                continue
            summary_rows.append(
                [
                    model,
                    dataset,
                    variant,
                    round(proposed["Acc"] - values["Acc"], 2),
                    round(values["Lat"] - proposed["Lat"], 2),
                    round(proposed["Tpt"] - values["Tpt"], 2),
                ]
            )

    files = [
        write_csv(output_dir / "table_4_11_raw.csv", ["model", "variant", "dataset", "accuracy_percent", "latency_ms", "throughput_tokens_s"], raw_rows),
        write_csv(
            output_dir / "table_4_11_ablation_delta.csv",
            ["model", "dataset", "removed_module", "accuracy_gain", "latency_reduction_ms", "throughput_gain_tokens_s"],
            summary_rows,
        ),
        write_json(output_dir / "table_4_11_summary.json", summary_rows),
    ]

    avg_acc_gain = sum(row[3] for row in summary_rows) / len(summary_rows)
    avg_latency_reduction = sum(row[4] for row in summary_rows) / len(summary_rows)
    avg_tpt_gain = sum(row[5] for row in summary_rows) / len(summary_rows)
    files.append(
        write_text(
            output_dir / "summary.txt",
            [
                "Table 4-11 reproduction: ablation study.",
                f"Average accuracy gain of Proposed over ablated variants: {avg_acc_gain:.2f}",
                f"Average latency reduction of Proposed over ablated variants: {avg_latency_reduction:.2f} ms",
                f"Average throughput gain of Proposed over ablated variants: {avg_tpt_gain:.2f} tokens/s",
            ],
        )
    )

    print("Table 4-11 experiment completed.")
    for file in files:
        print(f"  - {file.resolve()}")


if __name__ == "__main__":
    main()

