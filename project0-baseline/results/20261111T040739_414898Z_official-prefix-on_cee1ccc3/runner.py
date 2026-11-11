"""Run unchanged vLLM CLI against the existing server; keep all evidence."""
from __future__ import annotations

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


def dump(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def capture(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=15).stdout


def main() -> int:
    root = Path.cwd().resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / "project0-baseline/results" / f"{stamp}_official-prefix-on_{uuid4().hex[:8]}"
    out.mkdir(exist_ok=False)
    print(f"OBSERVATION_DIR={out}", flush=True)
    (out / "runner.py").write_text(Path(__file__).read_text(), encoding="utf-8")
    source = root / "benchmarks/prompts/baseline-v1.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    assert len(rows) == 128 and all(row["output_tokens"] == 128 for row in rows)
    expanded = out / "prompts-512-in-order.jsonl"
    with expanded.open("x", encoding="utf-8") as stream:
        for i in range(512):
            stream.write(json.dumps(rows[i % 128], ensure_ascii=False) + "\n")
    (out / "prompts-128-source.jsonl").write_bytes(source.read_bytes())
    installed = root / ".venv/lib/python3.12/site-packages/vllm/benchmarks"
    sources = {}
    for rel, name in (("serve.py", "official-serve-source.py"),
                      ("datasets/datasets.py", "official-datasets-source.py"),
                      ("lib/endpoint_request_func.py", "official-endpoint-source.py")):
        content = (installed / rel).read_bytes()
        (out / name).write_bytes(content)
        sources[rel] = hashlib.sha256(content).hexdigest()
    (out / "official-help.txt").write_text(Path("/tmp/vllm-official-bench-help-20261111.txt").read_text())
    api_pid = 439160
    proc = Path(f"/proc/{api_pid}")
    argv = proc.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
    safe_environment = {}
    for entry in proc.joinpath("environ").read_bytes().decode().split("\0"):
        key, _, value = entry.partition("=")
        if key in {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "VLLM_USE_V1"}:
            safe_environment[key] = value
    env = os.environ.copy()
    env.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    common = [str(root / ".venv/bin/vllm"), "bench", "serve",
              "--model", "models/Qwen2.5-7B-Instruct",
              "--tokenizer", "models/Qwen2.5-7B-Instruct",
              "--backend", "openai-chat", "--base-url", "http://127.0.0.1:8000",
              "--endpoint", "/v1/chat/completions", "--dataset-name", "custom",
              "--disable-shuffle", "--no-oversample", "--skip-chat-template",
              "--custom-output-len", "128", "--temperature", "0",
              "--request-rate", "inf", "--num-warmups", "0",
              "--ready-check-timeout-sec", "0", "--seed", "42",
              "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99",
              "--save-result", "--save-detailed", "--disable-tqdm", "--result-dir", str(out)]
    commands = []
    for concurrency in (1, 8, 32, 64):
        for phase, count, dataset in [("warmup", 128, out / "prompts-128-source.jsonl"),
                                       ("round1", 512, expanded), ("round2", 512, expanded),
                                       ("round3", 512, expanded)]:
            label = f"c{concurrency}-{phase}"
            command = common + ["--dataset-path", str(dataset), "--num-prompts", str(count),
                                "--max-concurrency", str(concurrency), "--result-filename", label + ".json"]
            commands.append(dict(label=label, concurrency=concurrency, phase=phase,
                                 count=count, command=command, command_shell=shlex.join(command)))
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=10, trust_env=False) as client:
        def metrics(name: str) -> str:
            response = client.get("/metrics")
            response.raise_for_status()
            with (out / name).open("x", encoding="utf-8") as stream:
                stream.write(response.text)
            return response.text

        before = metrics("server-before.prom")
        assert 'enable_prefix_caching="True"' in before
        assert 'vllm:num_requests_running{' in before
        models = client.get("/v1/models").json()
        assert any(item["id"] == "models/Qwen2.5-7B-Instruct" for item in models["data"])
        (out / "gpu-before.txt").write_text(capture(["nvidia-smi"]))
        dump(out / "context.json", dict(started_at_utc=now(), python=platform.python_version(),
             packages={name: importlib.metadata.version(name) for name in
                       ("vllm", "vllm-omni", "torch", "transformers", "httpx", "aiohttp", "numpy")},
             server_command=argv, server_pid=api_pid, server_environment=safe_environment,
             model=models, source_prompt_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             expanded_prompt_sha256=hashlib.sha256(expanded.read_bytes()).hexdigest(),
             official_source_sha256=sources, commands=commands,
             client_environment_overrides={key: env[key] for key in
                 ("NO_PROXY", "no_proxy", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONUNBUFFERED")},
             notes=["Warmup is a separate official CLI invocation traversing all 128 prompts.",
                    "Official native timeout is 6 hours; custom uses 120 seconds.",
                    "Server and cache are not restarted/reset between invocations."]))
        stop = threading.Event()

        def sample_gpu() -> None:
            command = ["nvidia-smi", "--id=1", "--query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"]
            with (out / "gpu-samples.csv").open("x", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["utc", "perf_counter", "gpu", "memory_used_mib", "memory_free_mib", "utilization_percent", "temperature_c", "power_w", "error"])
                while not stop.is_set():
                    stamp, tick = now(), perf_counter()
                    try:
                        writer.writerow([stamp, tick, *[x.strip() for x in capture(command).strip().split(",")], ""])
                    except Exception as exc:
                        writer.writerow([stamp, tick, "", "", "", "", "", "", str(exc)])
                    stream.flush()
                    stop.wait(1)

        thread = threading.Thread(target=sample_gpu, daemon=True)
        thread.start()
        outcomes = []
        process = None
        try:
            with (out / "events.jsonl").open("x", encoding="utf-8") as events:
                def event(value: dict) -> None:
                    value.update(utc=now(), perf_counter=perf_counter())
                    events.write(json.dumps(value, ensure_ascii=False) + "\n")
                    events.flush()
                    print(json.dumps(value, ensure_ascii=False), flush=True)

                for spec in commands:
                    label = spec["label"]
                    metrics(label + "-before.prom")
                    start = perf_counter()
                    event(dict(event="phase_start", label=label, count=spec["count"]))
                    with (out / (label + ".log")).open("x", encoding="utf-8") as log:
                        process = subprocess.Popen(spec["command"], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
                        while True:
                            try:
                                process.wait(timeout=30)
                                break
                            except subprocess.TimeoutExpired:
                                event(dict(event="progress", label=label, elapsed_s=perf_counter() - start))
                        code = process.returncode
                    metrics(label + "-after.prom")
                    result_path = out / (label + ".json")
                    result = json.loads(result_path.read_text()) if result_path.exists() else {}
                    outcome = dict(label=label, exit_code=code, completed=result.get("completed"),
                                   failed=result.get("failed"), duration=result.get("duration"),
                                   output_throughput=result.get("output_throughput"))
                    outcomes.append(outcome)
                    event(dict(event="phase_end", **outcome))
                    if code or result.get("completed") != spec["count"] or result.get("failed") != 0:
                        raise RuntimeError(f"Official benchmark phase failed: {outcome}; see {label}.log")
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            stop.set()
            thread.join(timeout=20)
            metrics("server-after.prom")
            (out / "gpu-after.txt").write_text(capture(["nvidia-smi"]))
            dump(out / "outcome.json", dict(finished_at_utc=now(), phases=outcomes,
                 source_prompt_sha256_after=hashlib.sha256(source.read_bytes()).hexdigest(),
                 server_pid_still_present=proc.exists(),
                 server_command_after=proc.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0") if proc.exists() else None))
    print(f"COMPLETE={out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
