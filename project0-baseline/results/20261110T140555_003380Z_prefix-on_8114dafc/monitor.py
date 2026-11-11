"""Execute the existing benchmark CLI with append-only external observation files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx


def now() -> str:
    return datetime.now(UTC).isoformat()


def capture(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=10).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--api-pid", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ") + "_prefix-on_" + uuid4().hex[:8]
    observation = root / "project0-baseline/results" / run_id
    observation.mkdir(parents=True, exist_ok=False)
    print(f"OBSERVATION_DIR={observation}", flush=True)
    with (observation / "monitor.py").open("x", encoding="utf-8") as stream:
        stream.write(Path(__file__).read_text(encoding="utf-8"))
    command = [
        str(root / ".venv/bin/python"), "-u", "project0-baseline/scripts/bench.py",
        "--model", "models/Qwen2.5-7B-Instruct",
        "--prompts", "benchmarks/prompts/baseline-v1.jsonl",
        "--concurrency", "1", "8", "32", "64",
        "--requests", "512", "--warmup", "128", "--rounds", "3",
        "--timeout", "120", "--temperature", "0",
        "--output-dir", str(observation),
    ]
    api_proc = Path("/proc") / str(args.api_pid)
    server_argv = api_proc.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
    safe_names = {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "VLLM_USE_V1"}
    safe_environment = {}
    for entry in api_proc.joinpath("environ").read_bytes().decode().split("\0"):
        key, _, value = entry.partition("=")
        if key in safe_names:
            safe_environment[key] = value
    prompts = root / "benchmarks/prompts/baseline-v1.jsonl"
    hash_before = hashlib.sha256(prompts.read_bytes()).hexdigest()
    before_gpu = capture(["nvidia-smi"])
    gpu_query = [
        "nvidia-smi", "--id=1",
        "--query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    headers = {}
    if key := os.getenv("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=10, trust_env=False, headers=headers) as client:
        metrics_before = client.get("/metrics")
        metrics_before.raise_for_status()
        if 'enable_prefix_caching="True"' not in metrics_before.text:
            raise RuntimeError("Expected an existing prefix-caching-enabled server")
        models = client.get("/v1/models")
        models.raise_for_status()
        model_ids = [model["id"] for model in models.json()["data"]]
        if "models/Qwen2.5-7B-Instruct" not in model_ids:
            raise RuntimeError(f"Wrong served model: {model_ids}")
        info = client.get("/server_info", params={"config_format": "json"})
        # Developer endpoint may be disabled; its status is preserved explicitly.
        server_info = {"status_code": info.status_code}
        if info.is_success:
            server_info["response"] = info.json()
        for name, content in (
            ("server-before.prom", metrics_before.text),
            ("gpu-before.txt", before_gpu),
        ):
            with (observation / name).open("x", encoding="utf-8") as stream:
                stream.write(content)
        context = {
            "started_at_utc": now(), "benchmark_command": command,
            "benchmark_command_shell": shlex.join(command),
            "server_command": server_argv, "server_environment_allowlist": safe_environment,
            "server_info": server_info, "models": models.json(),
            "client_python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in (
                "vllm", "vllm-omni", "torch", "transformers", "httpx",
            )},
            "prompts_sha256": hash_before,
            "client_source_sha256": hashlib.sha256((root / "project0-baseline/scripts/bench.py").read_bytes()).hexdigest(),
            "gpu_sampling_interval_s": 1.0, "gpu_sampling_command": gpu_query,
            "gpu_sample_clock": "time.perf_counter; same host clock as raw RequestResult timestamps",
            "cache_counters_window": "whole benchmark including warmup; existing service not restarted or reset",
        }
        with (observation / "observation-context.json").open("x", encoding="utf-8") as stream:
            json.dump(context, stream, ensure_ascii=False, indent=2)

        stop = threading.Event()

        def sample_gpu() -> None:
            with (observation / "gpu-samples.csv").open("x", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["utc", "sample_start_perf", "sample_end_perf", "gpu_index", "memory_used_mib", "memory_free_mib", "gpu_util_percent", "temperature_c", "power_w", "error"])
                while not stop.is_set():
                    start = perf_counter()
                    try:
                        values = next(csv.reader([capture(gpu_query)]))
                        writer.writerow([now(), start, perf_counter(), *[value.strip() for value in values], ""])
                    except Exception as exc:
                        writer.writerow([now(), start, perf_counter(), "", "", "", "", "", "", f"{type(exc).__name__}: {exc}"])
                    stream.flush()
                    stop.wait(max(0.01, 1.0 - (perf_counter() - start)))

        thread = threading.Thread(target=sample_gpu, daemon=True)
        thread.start()
        exit_code = 1
        try:
            with (observation / "bench-console.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                assert process.stdout is not None
                print(f"BENCH_PID={process.pid}", flush=True)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                exit_code = process.wait()
        finally:
            stop.set()
            thread.join(timeout=15)
            after = client.get("/metrics")
            after.raise_for_status()
            with (observation / "server-after.prom").open("x", encoding="utf-8") as stream:
                stream.write(after.text)
            with (observation / "gpu-after.txt").open("x", encoding="utf-8") as stream:
                stream.write(capture(["nvidia-smi"]))
            outcome = {
                "finished_at_utc": now(), "exit_code": exit_code,
                "prompts_sha256_after": hashlib.sha256(prompts.read_bytes()).hexdigest(),
                "client_source_sha256_after": hashlib.sha256((root / "project0-baseline/scripts/bench.py").read_bytes()).hexdigest(),
            }
            with (observation / "observation-outcome.json").open("x", encoding="utf-8") as stream:
                json.dump(outcome, stream, ensure_ascii=False, indent=2)
            print(f"BENCH_EXIT_CODE={exit_code}", flush=True)
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
