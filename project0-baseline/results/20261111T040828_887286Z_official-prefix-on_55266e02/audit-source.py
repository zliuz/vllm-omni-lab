"""Independently audit official raw JSON and compare with the existing baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path.cwd()
CUSTOM = ROOT / "project0-baseline/results/20261110T140555_003380Z_prefix-on_8114dafc"


def percentile(values: list[float], p: int) -> float:
    values = sorted(values)
    position = (len(values) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8), (
        actual,
        expected,
    )


def counter(path: Path, name: str) -> float:
    values = []
    for line in path.read_text().splitlines():
        if re.match(re.escape(name) + r"(?:\{|\s)", line):
            values.append(float(line.rsplit(" ", 1)[-1]))
    assert values, (path, name)
    return sum(values)


def delta(out: Path, label: str, name: str) -> float:
    return counter(out / f"{label}-after.prom", name) - counter(
        out / f"{label}-before.prom", name
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    context = json.loads((out / "context.json").read_text())
    outcome = json.loads((out / "outcome.json").read_text())
    assert len(outcome["phases"]) == 16
    assert all(row["exit_code"] == 0 for row in outcome["phases"])
    assert outcome["server_pid_still_present"]
    assert outcome["server_command_after"] == context["server_command"]
    prompt_path = out / "prompts-128-source.jsonl"
    assert (
        hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        == context["source_prompt_sha256"]
    )
    assert context["source_prompt_sha256"] == outcome["source_prompt_sha256_after"]
    source = [json.loads(line) for line in prompt_path.read_text().splitlines()]
    expanded = [
        json.loads(line)
        for line in (out / "prompts-512-in-order.jsonl").read_text().splitlines()
    ]
    assert expanded == source * 4
    samples = list(csv.DictReader((out / "gpu-samples.csv").open()))
    assert all(not row["error"] for row in samples)
    baseline = list(
        csv.DictReader((CUSTOM / "20261110T140555_257856Z_851ff01a/summary.csv").open())
    )
    formal, warm = [], []
    all_output_tokens = 0
    for spec in context["commands"]:
        label = spec["label"]
        record = json.loads((out / (label + ".json")).read_text())
        n = spec["count"]
        assert record["completed"] == n and record["failed"] == 0
        for key in (
            "input_lens",
            "output_lens",
            "ttfts",
            "itls",
            "latencies",
            "start_times",
            "queue_times",
            "generated_texts",
            "errors",
        ):
            assert len(record[key]) == n, (label, key)
        assert not any(record["errors"])
        assert all(0 < length <= 128 for length in record["output_lens"])
        assert all(
            0 <= ttft <= latency
            for ttft, latency in zip(record["ttfts"], record["latencies"])
        )
        tokens = sum(record["output_lens"])
        all_output_tokens += tokens
        assert tokens == record["total_output_tokens"]
        close(record["output_throughput"], tokens / record["duration"])
        close(record["request_throughput"], n / record["duration"])
        assert delta(out, label, "vllm:request_success_total") == n
        assert delta(out, label, "vllm:generation_tokens_total") == tokens
        assert delta(out, label, "vllm:num_preemptions_total") == 0
        metrics = {
            "ttft": record["ttfts"],
            "e2el": record["latencies"],
            "tpot": [
                (lat - first) / (length - 1)
                for lat, first, length in zip(
                    record["latencies"], record["ttfts"], record["output_lens"]
                )
                if length > 1
            ],
            "itl": [value for values in record["itls"] for value in values],
        }
        summary = {
            "concurrency": spec["concurrency"],
            "phase": spec["phase"],
            "requests": n,
            "succeeded": n,
            "failed": 0,
            "elapsed_s": record["duration"],
            "output_tokens": tokens,
            "output_tok_s": record["output_throughput"],
            "request_per_s": record["request_throughput"],
            "mean_output_tokens": tokens / n,
        }
        for metric, values in metrics.items():
            for p in (50, 95, 99):
                measured = percentile(values, p) * 1000
                close(record[f"p{p}_{metric}_ms"], measured)
                summary[f"{metric}_ms_p{p}"] = measured
        start = min(record["start_times"])
        finish = max(
            st + latency
            for st, latency in zip(record["start_times"], record["latencies"])
        )
        window = [
            row for row in samples if start <= float(row["perf_counter"]) <= finish
        ]
        assert window, label
        summary["gpu_sampled_peak_mib"] = max(
            float(row["memory_used_mib"]) for row in window
        )
        summary["gpu_samples"] = len(window)
        summary["ttft_over_1s"] = sum(value > 1 for value in record["ttfts"])
        hits = delta(out, label, "vllm:prefix_cache_hits_total")
        queries = delta(out, label, "vllm:prefix_cache_queries_total")
        summary["prefix_hit_tokens"] = hits
        summary["prefix_query_tokens"] = queries
        summary["prefix_token_hit_percent"] = 100 * hits / queries
        (warm if spec["phase"] == "warmup" else formal).append(summary)
    total_requests = sum(row["requests"] for row in formal + warm)
    assert total_requests == 6656
    for name, expected in (
        ("vllm:request_success_total", total_requests),
        ("vllm:generation_tokens_total", all_output_tokens),
        ("vllm:num_preemptions_total", 0),
    ):
        actual = counter(out / "server-after.prom", name) - counter(
            out / "server-before.prom", name
        )
        assert actual == expected, (name, actual, expected)
    comparisons = []
    for c in (1, 8, 32, 64):
        official = [row for row in formal if row["concurrency"] == c]
        custom = [row for row in baseline if int(row["concurrency"]) == c]
        assert len(official) == len(custom) == 3
        row = {"concurrency": c}
        for prefix, values in (("official", official), ("custom", custom)):
            for metric in (
                "output_tok_s",
                "request_per_s",
                "ttft_ms_p50",
                "ttft_ms_p95",
                "ttft_ms_p99",
                "tpot_ms_p50",
                "tpot_ms_p95",
                "tpot_ms_p99",
            ):
                floats = [float(value[metric]) for value in values]
                row[f"{prefix}_{metric}_median"] = statistics.median(floats)
                row[f"{prefix}_{metric}_min"] = min(floats)
                row[f"{prefix}_{metric}_max"] = max(floats)
        for metric in (
            "output_tok_s",
            "request_per_s",
            "ttft_ms_p50",
            "ttft_ms_p99",
            "tpot_ms_p50",
        ):
            row[metric + "_difference_percent"] = 100 * (
                row[f"official_{metric}_median"] / row[f"custom_{metric}_median"] - 1
            )
        row["official_gpu_sampled_peak_mib"] = max(
            value["gpu_sampled_peak_mib"] for value in official
        )
        row["throughput_within_20_percent"] = (
            abs(row["output_tok_s_difference_percent"]) <= 20
        )
        comparisons.append(row)
    write_csv(out / "official-rounds.csv", formal)
    write_csv(out / "official-warmup.csv", warm)
    write_csv(out / "comparison.csv", comparisons)
    audit = {
        "passed": True,
        "formal_requests": 6144,
        "warmup_requests": 512,
        "total_output_tokens": all_output_tokens,
        "gpu_samples": len(samples),
        "server_counter_match": True,
        "percentile_recomputed": True,
        "all_throughput_within_20_percent": all(
            row["throughput_within_20_percent"] for row in comparisons
        ),
        "timing_definitions_identical": False,
        "checks": [
            "all phases exit 0 and complete expected request counts",
            "server requests/tokens match each phase",
            "no preemptions",
            "raw arrays lengths and success/errors",
            "token cap and throughput recomputation",
            "all TTFT/TPOT/E2EL/ITL percentiles independently recomputed",
            "128 prompts repeated four times in order",
            "original server PID/command unchanged",
            "GPU sample errors absent",
        ],
    }
    with (out / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
    with (out / "audit-source.py").open("x", encoding="utf-8") as stream:
        stream.write(Path(__file__).read_text())
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    for row in comparisons:
        print(
            row["concurrency"],
            "official",
            row["official_output_tok_s_median"],
            "diff%",
            row["output_tok_s_difference_percent"],
            "TTFT",
            row["official_ttft_ms_p50_median"],
            "TPOT",
            row["official_tpot_ms_p50_median"],
        )


if __name__ == "__main__":
    main()
