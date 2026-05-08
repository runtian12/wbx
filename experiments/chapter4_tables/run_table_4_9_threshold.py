"""Reproduce Table 4-9: sensitivity to key-token verification threshold."""

from __future__ import annotations

from table_data import TABLE_4_9, THRESHOLDS, ensure_output_dir, write_csv, write_json, write_text


def main() -> None:
    output_dir = ensure_output_dir("table_4_9_threshold")

    raw_rows = []
    trend_rows = []
    for model, dataset, metric, values in TABLE_4_9:
        raw_rows.append([model, dataset, metric] + values)
        start = values[0]
        end = values[-1]
        delta = end - start
        delta_percent = delta / start * 100
        trend_rows.append([model, dataset, metric, start, end, round(delta, 4), round(delta_percent, 2)])

    files = [
        write_csv(output_dir / "table_4_9_raw.csv", ["model", "dataset", "metric"] + THRESHOLDS, raw_rows),
        write_csv(
            output_dir / "table_4_9_threshold_trend.csv",
            ["model", "dataset", "metric", "value_at_0.50", "value_at_0.65", "absolute_change", "change_percent"],
            trend_rows,
        ),
        write_json(output_dir / "table_4_9_summary.json", trend_rows),
    ]

    acc_drop = [row[-2] for row in trend_rows if row[2] == "Acc"]
    lat_drop = [row[-2] for row in trend_rows if row[2] == "Lat"]
    tpt_gain = [row[-2] for row in trend_rows if row[2] == "Tpt"]
    files.append(
        write_text(
            output_dir / "summary.txt",
            [
                "Table 4-9 reproduction: threshold sensitivity.",
                f"Mean accuracy change from 0.50 to 0.65: {sum(acc_drop) / len(acc_drop):.2f}",
                f"Mean latency change from 0.50 to 0.65: {sum(lat_drop) / len(lat_drop):.2f} ms",
                f"Mean throughput change from 0.50 to 0.65: {sum(tpt_gain) / len(tpt_gain):.2f} tokens/s",
            ],
        )
    )

    print("Table 4-9 experiment completed.")
    for file in files:
        print(f"  - {file.resolve()}")


if __name__ == "__main__":
    main()

