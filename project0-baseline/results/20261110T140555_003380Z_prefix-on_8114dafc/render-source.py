"""Render derived tables and a standalone figure from audited observations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def fmt(stat: dict, digits: int = 2) -> str:
    return f"{stat['median']:.{digits}f}" if stat["median"] is not None else "—"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation", type=Path)
    args = parser.parse_args()
    observation = args.observation.resolve()
    data = json.loads((observation / "audit.json").read_text())
    assert data["status"] == "completed_and_audited"
    configurations = data["configurations"]
    lines = [
        "| 并发 | 输出 tok/s：中位数 [最小, 最大] | 请求吞吐（req/s） | TTFT p50（ms） | TTFT p95（ms） | TTFT p99（ms） | TPOT p50（ms） | 显存峰值（MiB） | 失败/正式请求 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in configurations:
        stats = config["stats"]
        speed = stats["output_tok_s"]
        throughput = f"{speed['median']:.2f} [{speed['min']:.2f}, {speed['max']:.2f}]"
        lines.append("| " + " | ".join([
            str(config["concurrency"]), throughput, fmt(stats["request_per_s"], 3),
            fmt(stats["ttft_ms_p50"]), fmt(stats["ttft_ms_p95"]), fmt(stats["ttft_ms_p99"]),
            fmt(stats["tpot_ms_p50"]), str(config["gpu_memory_peak_mib"]),
            f"{config['failed_requests']}/{config['measured_requests']}",
        ]) + " |")
    table = "\n".join(lines) + "\n"
    with (observation / "readme-table.md").open("x", encoding="utf-8") as stream:
        stream.write(table)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [config["concurrency"] for config in configurations]
    med = [config["stats"]["output_tok_s"]["median"] for config in configurations]
    low = [config["stats"]["output_tok_s"]["min"] for config in configurations]
    high = [config["stats"]["output_tok_s"]["max"] for config in configurations]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axes[0].errorbar(x, med, yerr=[[m-l for m, l in zip(med, low)], [h-m for h, m in zip(high, med)]], marker="o", capsize=5)
    axes[0].set_ylabel("Successful output tokens / second")
    axes[0].set_title("Median and min/max across 3 measured rounds")
    for percentile in (50, 95, 99):
        y = [config["stats"][f"ttft_ms_p{percentile}"]["median"] for config in configurations]
        axes[1].plot(x, y, marker="o", label=f"TTFT p{percentile}")
    axes[1].set_ylabel("Client TTFT (ms)")
    axes[1].set_title("Median of the per-round percentiles")
    axes[1].legend()
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(x, [str(c) for c in x])
        axis.set_xlabel("In-flight request concurrency")
        axis.grid(alpha=0.25)
    figure.suptitle("Qwen2.5-7B-Instruct | vLLM 0.32.0 | RTX 4090 GPU1 | TP=1 | Prefix cache ON")
    image = observation / "prefix-on-curves.png"
    assert not image.exists()
    figure.savefig(image, dpi=180)
    plt.close(figure)

    # Readable per-round table preserves fluctuation and all tail percentiles.
    run_dir = Path(data["run_dir"])
    with (run_dir / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    detailed = [
        "| 并发 | 轮次 | 输出 tok/s | req/s | TTFT p50 / p95 / p99（ms） | TPOT p50 / p95 / p99（ms） | 失败数 |",
        "|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in rows:
        latency = lambda name: " / ".join(f"{float(row[f'{name}_p{p}']):.2f}" if row[f"{name}_p{p}"] else "—" for p in (50, 95, 99))
        detailed.append(f"| {row['concurrency']} | {row['round']} | {float(row['output_tok_s']):.2f} | {float(row['request_per_s']):.3f} | {latency('ttft_ms')} | {latency('tpot_ms')} | {row['failed']} |")
    with (observation / "rounds-table.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(detailed) + "\n")
    print(table)
    print(json.dumps({
        "started": data["started_local"], "finished": data["finished_local"],
        "cache_hit_ratio_including_warmup": data["whole_window_cache_hit_ratio"],
        "server_and_client_counts_match": data["server_and_client_counts_match"],
        "failures": data["client_failures"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
