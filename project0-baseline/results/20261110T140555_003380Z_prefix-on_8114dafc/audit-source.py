"""Independently audit completed benchmark records and derive report data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median, quantiles
from zoneinfo import ZoneInfo


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def json_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def counter(text: str, name: str) -> float:
    return sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines() if line.startswith(name + "{"))


def percentile(values: list[float], p: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[p - 1]


def assert_number(cell: str, expected: float | None) -> None:
    if expected is None:
        assert cell == "", (cell, expected)
    else:
        assert math.isclose(float(cell), expected, rel_tol=1e-9, abs_tol=1e-7), (cell, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation", type=Path)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    observation = args.observation.resolve()
    runs = [path.parent for path in observation.glob("*/metadata.json")]
    assert len(runs) == 1
    run = runs[0]
    context = json.loads((observation / "observation-context.json").read_text())
    events = json_rows(run / "events.jsonl")
    if events[-1]["event"] != "completed" or not (observation / "observation-outcome.json").exists():
        print(json.dumps({"status": "running", "last_event": events[-1], "completed_rows": len(csv_rows(run / "summary.csv")) if (run / "summary.csv").exists() else 0}, ensure_ascii=False, indent=2))
        return
    outcome = json.loads((observation / "observation-outcome.json").read_text())
    assert outcome["exit_code"] == 0
    assert outcome["prompts_sha256_after"] == context["prompts_sha256"]
    assert outcome["client_source_sha256_after"] == context["client_source_sha256"]
    raw = json_rows(run / "requests.jsonl")
    summary = csv_rows(run / "summary.csv")
    repeated = csv_rows(run / "repeats.csv")
    samples = csv_rows(observation / "gpu-samples.csv")
    expected_concurrency = (1, 8, 32, 64)
    assert len(raw) == 4 * (128 + 512 * 3)
    assert len(summary) == 12
    assert len(repeated) == 4 * 23
    assert sum(item["phase"] == "warmup" for item in raw) == 512
    assert sum(item["phase"] == "measured" for item in raw) == 6144
    per_round = {}
    for c in expected_concurrency:
        warmup = [item for item in raw if item["concurrency"] == c and item["phase"] == "warmup"]
        assert len(warmup) == 128
        for r in (1, 2, 3):
            records = [item for item in raw if item["concurrency"] == c and item["phase"] == "measured" and item["round"] == r]
            assert len(records) == 512
            assert [item["request_id"] for item in records] == list(range(512))
            assert [item["prompt_index"] for item in records] == [index % 128 for index in range(512)]
            per_round[(c, r)] = records
    metrics = ("ttft_ms", "tpot_ms", "e2e_ms", "output_tok_s", "chunk_itl_ms")
    for row in summary:
        c, r = int(row["concurrency"]), int(row["round"])
        records = per_round[(c, r)]
        successful = [item for item in records if item["success"]]
        elapsed = float(row["elapsed_s"])
        assert int(row["requests"]) == 512
        assert int(row["succeeded"]) == len(successful)
        assert int(row["failed"]) == 512 - len(successful)
        assert_number(row["failure_rate"], (512 - len(successful)) / 512)
        assert_number(row["request_per_s"], len(successful) / elapsed)
        total_tokens = sum(item["output_tokens"] for item in successful)
        assert int(row["output_tokens"]) == total_tokens
        assert_number(row["output_tok_s"], total_tokens / elapsed)
        values = {name: [] for name in metrics}
        for item in successful:
            start, end = item["started_at"], item["finished_at"]
            times = item["content_times"]
            tokens = item["output_tokens"]
            assert start <= times[0] <= times[-1] <= end
            assert all(a <= b for a, b in zip(times, times[1:]))
            assert 0 < tokens <= 128
            values["ttft_ms"].append((times[0] - start) * 1000)
            values["e2e_ms"].append((end - start) * 1000)
            values["output_tok_s"].append(tokens / (end - start))
            if tokens > 1:
                values["tpot_ms"].append((end - times[0]) / (tokens - 1) * 1000)
            if len(times) > 1:
                values["chunk_itl_ms"].append((times[-1] - times[0]) / (len(times) - 1) * 1000)
        for name in metrics:
            for p in (50, 95, 99):
                assert_number(row[f"{name}_p{p}"], percentile(values[name], p))

    # Cross-check multi-round CSV against the individual measured rows.
    for row in repeated:
        c, name = int(row["concurrency"]), row["metric"]
        values = [float(item[name]) for item in summary if int(item["concurrency"]) == c and item[name] != ""]
        assert_number(row["median"], median(values) if values else None)
        assert_number(row["min"], min(values) if values else None)
        assert_number(row["max"], max(values) if values else None)

    data = []
    for c in expected_concurrency:
        rows = [item for item in summary if int(item["concurrency"]) == c]
        records = [item for item in raw if item["concurrency"] == c and item["phase"] == "measured"]
        windows = [(min(item["started_at"] for item in per_round[(c, r)]), max(item["finished_at"] for item in per_round[(c, r)])) for r in (1, 2, 3)]
        observed = [sample for sample in samples if not sample["error"] and any(start <= (float(sample["sample_start_perf"]) + float(sample["sample_end_perf"])) / 2 <= end for start, end in windows)]
        assert observed
        stats = {}
        for name in summary[0]:
            if name in ("concurrency", "round"):
                continue
            values = [float(item[name]) for item in rows if item[name] != ""]
            stats[name] = {"median": median(values) if values else None, "min": min(values) if values else None, "max": max(values) if values else None}
        data.append({
            "concurrency": c, "measured_requests": len(records),
            "failed_requests": sum(not item["success"] for item in records),
            "warmup_failed_requests": sum(not item["success"] for item in raw if item["concurrency"] == c and item["phase"] == "warmup"),
            "gpu_samples": len(observed),
            "gpu_memory_peak_mib": max(int(sample["memory_used_mib"]) for sample in observed),
            "gpu_temperature_peak_c": max(int(sample["temperature_c"]) for sample in observed),
            "gpu_power_peak_w": max(float(sample["power_w"]) for sample in observed),
            "stats": stats,
            "actual_output_tokens_mean": sum(item["output_tokens"] for item in records if item["success"]) / sum(item["success"] for item in records),
        })
    before = (observation / "server-before.prom").read_text()
    after = (observation / "server-after.prom").read_text()
    queries = counter(after, "vllm:prefix_cache_queries_total") - counter(before, "vllm:prefix_cache_queries_total")
    hits = counter(after, "vllm:prefix_cache_hits_total") - counter(before, "vllm:prefix_cache_hits_total")
    server_success = counter(after, "vllm:request_success_total") - counter(before, "vllm:request_success_total")
    generation_tokens = counter(after, "vllm:generation_tokens_total") - counter(before, "vllm:generation_tokens_total")
    local_tokens = sum(item["output_tokens"] or 0 for item in raw)
    assert 'enable_prefix_caching="True"' in before and 'enable_prefix_caching="True"' in after
    result = {
        "status": "completed_and_audited", "observation_dir": str(observation), "run_dir": str(run),
        "started_local": datetime.fromisoformat(context["started_at_utc"]).astimezone(ZoneInfo("Asia/Shanghai")).isoformat(),
        "finished_local": datetime.fromisoformat(outcome["finished_at_utc"]).astimezone(ZoneInfo("Asia/Shanghai")).isoformat(),
        "raw_requests": len(raw), "measured_requests": 6144,
        "client_failures": dict(Counter(item["error"] for item in raw if not item["success"])),
        "gpu_sampler_errors": [sample["error"] for sample in samples if sample["error"]],
        "whole_window_cache_queried_tokens": queries,
        "whole_window_cache_hit_tokens": hits,
        "whole_window_cache_hit_ratio": hits / queries if queries else None,
        "server_completed_requests_delta": server_success,
        "server_generation_tokens_delta": generation_tokens,
        "client_recorded_tokens": local_tokens,
        "server_and_client_counts_match": server_success == len(raw) and generation_tokens == local_tokens,
        "configurations": data,
    }
    if args.save:
        with (observation / "audit.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
