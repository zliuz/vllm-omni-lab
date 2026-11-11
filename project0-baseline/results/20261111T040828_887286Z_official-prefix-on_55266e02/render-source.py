"""Render audited CSVs into reviewable tables and a standalone comparison plot."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    rounds = read_csv(out / "official-rounds.csv")
    compare = read_csv(out / "comparison.csv")
    context = json.loads((out / "context.json").read_text())
    outcome = json.loads((out / "outcome.json").read_text())
    audit = json.loads((out / "audit.json").read_text())
    assert audit["passed"]
    official_table = [
        "| 并发 | 官方输出 tok/s：中位数 [最小, 最大] | req/s | TTFT p50 / p95 / p99（ms） | TPOT p50 / p95 / p99（ms） | 显存峰值（MiB） | 失败/正式请求 |",
        "|---:|---:|---:|---|---|---:|---:|",
    ]
    comparison_table = [
        "| 并发 | 自写输出 tok/s | 官方输出 tok/s | 官方相对自写 | 自写 / 官方 TTFT p50（ms） | 自写 / 官方 TPOT p50（ms） |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    round_table = [
        "| 并发 | 轮次 | 输出 tok/s | req/s | 实际输出 token | TTFT p50 / p95 / p99（ms） | TPOT p50 / p95 / p99（ms） |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in compare:
        c = int(row["concurrency"])
        f = lambda key, row=row: float(row[key])
        latency = lambda metric, f=f: " / ".join(
            f"{f(f'official_{metric}_ms_p{p}_median'):.2f}" for p in (50, 95, 99)
        )
        official_table.append(
            f"| {c} | {f('official_output_tok_s_median'):.2f} [{f('official_output_tok_s_min'):.2f}, {f('official_output_tok_s_max'):.2f}] | "
            f"{f('official_request_per_s_median'):.3f} | {latency('ttft')} | {latency('tpot')} | {f('official_gpu_sampled_peak_mib'):.0f} | 0/1536 |"
        )
        comparison_table.append(
            f"| {c} | {f('custom_output_tok_s_median'):.2f} | {f('official_output_tok_s_median'):.2f} | {f('output_tok_s_difference_percent'):+.2f}% | "
            f"{f('custom_ttft_ms_p50_median'):.2f} / {f('official_ttft_ms_p50_median'):.2f} | "
            f"{f('custom_tpot_ms_p50_median'):.2f} / {f('official_tpot_ms_p50_median'):.2f} |"
        )
    for row in rounds:
        latency = lambda metric, row=row: " / ".join(
            f"{float(row[f'{metric}_ms_p{p}']):.2f}" for p in (50, 95, 99)
        )
        round_table.append(
            f"| {row['concurrency']} | {row['phase'].removeprefix('round')} | {float(row['output_tok_s']):.2f} | "
            f"{float(row['request_per_s']):.3f} | {row['output_tokens']} | {latency('ttft')} | {latency('tpot')} |"
        )
    with (out / "tables.md").open("x", encoding="utf-8") as stream:
        stream.write(
            "\n".join(official_table)
            + "\n\n"
            + "\n".join(comparison_table)
            + "\n\n"
            + "\n".join(round_table)
            + "\n"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for prefix, label, color, marker in (
        ("custom", "Custom client (Nov 10)", "#377eb8", "o"),
        ("official", "Official CLI (Nov 11)", "#e67300", "s"),
    ):
        xs = [int(row["concurrency"]) for row in compare]
        for axis, metric in ((axes[0], "output_tok_s"), (axes[1], "ttft_ms_p99")):
            median = [float(row[f"{prefix}_{metric}_median"]) for row in compare]
            low = [
                value - float(row[f"{prefix}_{metric}_min"])
                for value, row in zip(median, compare)
            ]
            high = [
                float(row[f"{prefix}_{metric}_max"]) - value
                for value, row in zip(median, compare)
            ]
            axis.errorbar(
                xs,
                median,
                yerr=[low, high],
                color=color,
                marker=marker,
                capsize=5,
                label=label,
                linewidth=1.5,
            )
    axes[0].set_ylabel("Output throughput (tok/s)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("TTFT p99 (ms, log scale)")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("Maximum in-flight requests")
        axis.set_xticks([1, 8, 32, 64])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(
        "Prefix caching ON: median and observed range of 3 rounds\n128 warmups per concurrency; 512 measured requests per round"
    )
    image_path = out / "official-vs-custom.png"
    assert not image_path.exists()
    fig.savefig(image_path, dpi=170)
    plt.close(fig)
    report_stats = {
        "started_local": datetime.fromisoformat(context["started_at_utc"])
        .astimezone(ZoneInfo("Asia/Shanghai"))
        .isoformat(),
        "finished_local": datetime.fromisoformat(outcome["finished_at_utc"])
        .astimezone(ZoneInfo("Asia/Shanghai"))
        .isoformat(),
        "max_absolute_throughput_difference_percent": max(
            abs(float(row["output_tok_s_difference_percent"])) for row in compare
        ),
        "formal_prefix_hit_tokens": sum(
            float(row["prefix_hit_tokens"]) for row in rounds
        ),
        "formal_prefix_query_tokens": sum(
            float(row["prefix_query_tokens"]) for row in rounds
        ),
        "max_ttft_ms": max(
            max(json.loads((out / f"c{c}-round{r}.json").read_text())["ttfts"]) * 1000
            for c in (1, 8, 32, 64)
            for r in (1, 2, 3)
        ),
        "max_latency_s": max(
            max(json.loads((out / f"c{c}-round{r}.json").read_text())["latencies"])
            for c in (1, 8, 32, 64)
            for r in (1, 2, 3)
        ),
        "formal_ttft_over_1s": sum(int(row["ttft_over_1s"]) for row in rounds),
        "output_tokens_mean_by_concurrency": {
            str(c): sum(
                int(row["output_tokens"])
                for row in rounds
                if int(row["concurrency"]) == c
            )
            / 1536
            for c in (1, 8, 32, 64)
        },
        "ttft_p99_difference_over_20_percent": [
            {
                "concurrency": int(row["concurrency"]),
                "custom_ms": float(row["custom_ttft_ms_p99_median"]),
                "official_ms": float(row["official_ttft_ms_p99_median"]),
                "difference_percent": float(row["ttft_ms_p99_difference_percent"]),
            }
            for row in compare
            if abs(float(row["ttft_ms_p99_difference_percent"])) > 20
        ],
    }
    with (out / "report-stats.json").open("x", encoding="utf-8") as stream:
        json.dump(report_stats, stream, ensure_ascii=False, indent=2)
    with (out / "render-source.py").open("x", encoding="utf-8") as stream:
        stream.write(Path(__file__).read_text())
    print((out / "tables.md").read_text())
    print(json.dumps(report_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
