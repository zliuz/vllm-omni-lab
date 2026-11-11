"""只测试已完成的框架；HTTP 使用内存 transport，核心算法使用假结果。"""

import asyncio
import csv
import importlib.util
import json
import sys
from collections.abc import Coroutine
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench.py"
SPEC = importlib.util.spec_from_file_location("project0_bench", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


@pytest.fixture
def config(tmp_path: Path) -> bench.BenchConfig:
    """生成只有一条 prompt 的临时实验配置，避免使用真实结果目录。"""
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        '{"prompt": "hello", "output_tokens": 8}\n',
        encoding="utf-8",
    )
    return bench.parse_args(
        [
            "--model",
            "test-model",
            "--prompts",
            str(prompt_file),
            "--output-dir",
            str(tmp_path / "results"),
            "--concurrency",
            "1",
            "2",
            "--requests",
            "4",
        ]
    )


@pytest.mark.parametrize(
    "line",
    [
        "not-json",
        "[]",
        "null",
        "{}",
        '{"prompt": "", "output_tokens": 8}',
        '{"prompt": "x", "output_tokens": true}',
        '{"prompt": "x", "output_tokens": 0}',
        '{"prompt": "x", "output_tokens": 1.5}',
    ],
)
def test_invalid_prompts_report_line_number(tmp_path: Path, line: str) -> None:
    """坏输入应在请求前失败，并能定位到原文件行号。"""
    path = tmp_path / "bad.jsonl"
    path.write_text("\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":2:"):
        bench.load_prompts(path)


def test_empty_prompts_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="为空"):
        bench.load_prompts(path)


def test_default_baseline_dataset_is_balanced_and_loadable() -> None:
    """检查实际负载：128 条不同任务，八类交错排列，统一输出上限且可直接读取。"""
    settings = bench.parse_args(["--model", "test-model", "--dry-run"])
    assert settings.prompts.name == "baseline-v1.jsonl"
    rows = [
        json.loads(line)
        for line in settings.prompts.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 128
    assert len({row["id"] for row in rows}) == 128
    assert len({row["prompt"] for row in rows}) == 128
    categories = {row["category"] for row in rows}
    assert len(categories) == 8
    assert all(
        sum(row["category"] == category for row in rows) == 16
        for category in categories
    )
    for offset in range(0, len(rows), 8):
        assert {row["category"] for row in rows[offset : offset + 8]} == categories
    assert {row["output_tokens"] for row in rows} == {128}
    prompts = bench.load_prompts(settings.prompts)
    assert len(prompts) == 128
    assert [prompt.text for prompt in prompts] == [row["prompt"] for row in rows]
    assert settings.requests >= len(prompts)  # 默认正式轮次至少覆盖全部任务一次。


def test_legacy_smoke_dataset_is_still_available() -> None:
    """默认集更新后，原来的八条冒烟输入仍可显式选择，避免破坏旧实验复现。"""
    path = SCRIPT.parents[2] / "benchmarks/prompts/smoke.jsonl"
    prompts = bench.load_prompts(path)
    assert len(prompts) == 8
    assert prompts[0].text == "用三句话解释什么是 continuous batching。"


@pytest.mark.parametrize(
    "args",
    [
        ["--concurrency", "0"],
        ["--concurrency", "1", "1"],
        ["--requests", "1"],
        ["--warmup", "9"],
        ["--rounds", "2"],
        ["--timeout", "nan"],
        ["--temperature", "inf"],
        ["--base-url", "http://localhost:8000/v1"],
        ["--base-url", "http://user:password@localhost:8000"],
    ],
)
def test_invalid_cli_rejected(args: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        bench.parse_args(["--model", "test-model", *args])
    assert exc.value.code == 2


def test_dry_run_neither_connects_nor_writes(
    config: bench.BenchConfig,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """未完成核心代码时仍能校验；dry-run 的成功不能冒充实测成功。"""

    def forbidden_client(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run 不应创建 HTTP client")

    monkeypatch.setattr(bench.httpx, "AsyncClient", forbidden_client)
    code = bench.main(
        [
            "--model",
            config.model,
            "--prompts",
            str(config.prompts),
            "--output-dir",
            str(config.output_dir),
            "--dry-run",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert code == 0 and output["prompt_count"] == 1
    assert "未运行" in output["status"]
    assert not config.output_dir.exists()


def test_run_directories_preserve_previous_results(config: bench.BenchConfig) -> None:
    """连续实验应进入不同目录，旧元数据保持不变。"""
    first = bench.create_run_dir(config)
    previous = (first / "metadata.json").read_bytes()
    second = bench.create_run_dir(config)
    assert first != second
    assert (first / "metadata.json").read_bytes() == previous
    metadata = json.loads(previous)
    assert len(metadata["prompts_sha256"]) == 64


def mock_server(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    """将框架的 HTTP 访问改为内存模型列表响应，不启动真实服务。"""
    original_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": model}]})

    def client_factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return original_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(bench.httpx, "AsyncClient", client_factory)


def test_orchestration_separates_warmup_and_persists_rounds(
    config: bench.BenchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用假请求验证每档单独预热、3 个正式轮及原始记录保存，非性能实验。"""
    mock_server(monkeypatch, config.model)
    calls = []
    summarized = []

    async def fake_round(
        client: httpx.AsyncClient,
        settings: bench.BenchConfig,
        prompts: list[bench.Prompt],
        concurrency: int,
        count: int,
    ) -> bench.BatchResult:
        calls.append((concurrency, count))
        return bench.BatchResult(
            [
                bench.RequestResult(request_id=i, prompt_index=0, success=True)
                for i in range(count)
            ],
            elapsed_s=1.0,
        )

    def fake_summary(
        batch: bench.BatchResult,
        concurrency: int,
        round_index: int,
    ) -> dict[str, bench.MetricValue]:
        summarized.append((concurrency, round_index, len(batch.requests)))
        row = dict.fromkeys(bench.SUMMARY_FIELDS)
        row.update(
            concurrency=concurrency, round=round_index, requests=len(batch.requests)
        )
        return row

    monkeypatch.setattr(bench, "run_round", fake_round)
    monkeypatch.setattr(bench, "summarize_round", fake_summary)
    monkeypatch.setattr(bench, "summarize_repeats", lambda rows: [])
    run_dir = asyncio.run(
        bench.run_benchmark(config, bench.load_prompts(config.prompts))
    )
    assert calls == [(c, n) for c in (1, 2) for n in (10, 4, 4, 4)]
    assert summarized == [(c, r, 4) for c in (1, 2) for r in (1, 2, 3)]
    raw = [
        json.loads(line)
        for line in (run_dir / "requests.jsonl").read_text().splitlines()
    ]
    assert sum(item["phase"] == "warmup" for item in raw) == 20
    assert sum(item["phase"] == "measured" for item in raw) == 24
    assert len((run_dir / "summary.csv").read_text().splitlines()) == 7
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "completed"


def test_server_model_mismatch_records_failure(
    config: bench.BenchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """服务模型名错误应保留失败证据，不生成任何性能指标。"""
    mock_server(monkeypatch, "another-model")
    with pytest.raises(ValueError, match="模型名不匹配"):
        asyncio.run(bench.run_benchmark(config, bench.load_prompts(config.prompts)))
    (run_dir,) = config.output_dir.iterdir()
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "aborted"
    assert events[-1]["error_type"] == "ValueError"
    assert not (run_dir / "summary.csv").exists()
    assert not (run_dir / "requests.jsonl").exists()


def test_raw_results_survive_summary_failure(
    config: bench.BenchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """统计练习未完成时，已经收到的原始请求记录仍可用于排查。"""
    mock_server(monkeypatch, config.model)

    async def fake_round(
        client: httpx.AsyncClient,
        settings: bench.BenchConfig,
        prompts: list[bench.Prompt],
        concurrency: int,
        count: int,
    ) -> bench.BatchResult:
        return bench.BatchResult(
            [
                bench.RequestResult(
                    request_id=i,
                    prompt_index=0,
                    success=False,
                    error="synthetic request failure",
                )
                for i in range(count)
            ],
            elapsed_s=1.0,
        )

    def broken_summary(*args: object) -> dict[str, bench.MetricValue]:
        raise ValueError("synthetic summary failure")

    monkeypatch.setattr(bench, "run_round", fake_round)
    monkeypatch.setattr(bench, "summarize_round", broken_summary)
    settings = replace(config, concurrency=(1,))
    with pytest.raises(ValueError, match="synthetic summary failure"):
        asyncio.run(bench.run_benchmark(settings, bench.load_prompts(config.prompts)))
    (run_dir,) = config.output_dir.iterdir()
    assert len((run_dir / "requests.jsonl").read_text().splitlines()) == 14
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "aborted"


def test_run_round_limits_concurrency_and_keeps_failed_requests(
    config: bench.BenchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证并发上限、prompt 循环、结果顺序和单请求失败不影响整轮执行。"""
    active = 0
    peak_active = 0
    seen = []

    async def fake_send(
        client: httpx.AsyncClient,
        config: bench.BenchConfig,
        prompt: bench.Prompt,
        request_id: int,
        prompt_index: int,
    ) -> bench.RequestResult:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        seen.append((request_id, prompt_index, prompt.text))
        try:
            await asyncio.sleep(0)  # 主动让出执行机会，制造多个在途请求。
            return bench.RequestResult(
                request_id=request_id,
                prompt_index=prompt_index,
                success=request_id != 3,
                error="synthetic HTTP failure" if request_id == 3 else None,
            )
        finally:
            active -= 1

    async def scenario() -> bench.BatchResult:
        async with httpx.AsyncClient() as client:
            return await bench.run_round(
                client,
                config,
                [bench.Prompt("A", 8), bench.Prompt("B", 8)],
                2,
                7,
            )

    monkeypatch.setattr(bench, "send_request", fake_send)
    # 使用人工时钟验证整轮耗时差值，不把测试耗时当成性能实验。
    times = iter([10.0, 12.5])
    monkeypatch.setattr(bench, "perf_counter", lambda: next(times))
    batch = asyncio.run(scenario())
    assert peak_active == 2 and active == 0
    assert sorted(seen) == [(i, i % 2, "AB"[i % 2]) for i in range(7)]
    assert [result.request_id for result in batch.requests] == list(range(7))
    assert sum(not result.success for result in batch.requests) == 1
    assert batch.elapsed_s == 2.5


@pytest.mark.parametrize("cancel_round", [False, True])
def test_run_round_drains_tasks_before_propagating_failure(
    config: bench.BenchConfig,
    monkeypatch: pytest.MonkeyPatch,
    cancel_round: bool,
) -> None:
    """异常和外部取消均需等待异步清理，并处理等待 semaphore 的任务。"""
    real_create_task = asyncio.create_task
    children = []
    started = set()
    cleaned = set()
    failure = RuntimeError("synthetic unexpected failure")

    def record_task(
        coro: Coroutine[object, object, bench.RequestResult],
    ) -> asyncio.Task[bench.RequestResult]:
        task = real_create_task(coro)
        children.append(task)
        return task

    async def scenario() -> None:
        two_started = asyncio.Event()

        async def fake_send(
            client: httpx.AsyncClient,
            config: bench.BenchConfig,
            prompt: bench.Prompt,
            request_id: int,
            prompt_index: int,
        ) -> bench.RequestResult:
            started.add(request_id)
            if len(started) >= 2:
                two_started.set()
            try:
                if not cancel_round and request_id == 0:
                    await two_started.wait()
                    raise failure
                await asyncio.Event().wait()  # 等待被取消，而非自行结束。
                raise AssertionError("不可达")
            finally:
                # 模拟响应流关闭时的异步清理，验证 run_round 确实等待了它。
                await asyncio.sleep(0)
                cleaned.add(request_id)

        monkeypatch.setattr(bench, "send_request", fake_send)
        async with httpx.AsyncClient() as client, asyncio.timeout(2.0):
            if cancel_round:
                # 用原函数创建父任务，避免将其误记作 run_round 的子任务。
                parent = real_create_task(
                    bench.run_round(
                        client,
                        config,
                        [bench.Prompt("A", 8)],
                        2,
                        6,
                    )
                )
                await two_started.wait()
                parent.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await parent
            else:
                with pytest.raises(RuntimeError) as exc:
                    await bench.run_round(client, config, [bench.Prompt("A", 8)], 2, 6)
                assert exc.value is failure  # 原异常对象没有被清理异常替换。

            # 在退出事件循环之前检查，不能借 asyncio.run 的收尾掩盖任务泄漏。
            assert len(children) == 6
            assert all(task.done() for task in children)
            assert cleaned == started
            assert len(started) < 6  # 也包含尚未取得并发槽位的任务。
            if cancel_round:
                assert all(task.cancelled() for task in children)

    monkeypatch.setattr(bench.asyncio, "create_task", record_task)
    asyncio.run(scenario())


@pytest.fixture
def successful_request() -> bench.RequestResult:
    """手算样例：E2E=2s，TTFT=0.5s，4 token，两个 chunk 间隔均为 0.5s。"""
    return bench.RequestResult(
        request_id=0,
        prompt_index=0,
        success=True,
        started_at=10.0,
        finished_at=12.0,
        content_times=[10.5, 11.0, 11.5],
        output_tokens=4,
    )


def test_request_metrics_matches_hand_calculation(
    successful_request: bench.RequestResult,
) -> None:
    """用固定期望验证公式和秒/毫秒换算，而不是重复实现计算过程。"""
    original = asdict(successful_request)
    assert bench.request_metrics(successful_request) == pytest.approx(
        {
            "ttft_ms": 500.0,
            "e2e_ms": 2000.0,
            "tpot_ms": 500.0,
            "output_tok_s": 2.0,
            "chunk_itl_ms": 500.0,
        }
    )
    assert asdict(successful_request) == original


def test_failed_request_metrics_are_empty(
    successful_request: bench.RequestResult,
) -> None:
    """即使失败前已收到部分文本，也不能混入成功请求的指标分布。"""
    result = replace(successful_request, success=False, error="synthetic timeout")
    assert bench.request_metrics(result) == dict.fromkeys(bench.METRICS)


@pytest.mark.parametrize(
    "tokens,speed",
    [
        (None, None),
        (0, 0.0),
        (1, 0.5),
        (-1, None),
        (True, None),
        (1.5, None),
    ],
)
def test_request_metrics_handles_missing_and_small_token_counts(
    successful_request: bench.RequestResult,
    tokens: float | None,
    speed: float | None,
) -> None:
    """0/1 token 没有 TPOT；缺失/无效 usage 不抹掉已观测到的时间指标。"""
    result = replace(successful_request, output_tokens=tokens)
    metrics = bench.request_metrics(result)
    assert metrics["tpot_ms"] is None
    assert metrics["output_tok_s"] == speed
    assert metrics["ttft_ms"] == 500.0
    assert metrics["e2e_ms"] == 2000.0
    assert metrics["chunk_itl_ms"] == 500.0


def test_one_text_event_can_contain_multiple_tokens(
    successful_request: bench.RequestResult,
) -> None:
    """只有一个文本事件时，实际 4 token 仍可用于 TPOT，但 chunk ITL 无定义。"""
    metrics = bench.request_metrics(replace(successful_request, content_times=[10.5]))
    assert metrics["tpot_ms"] == 500.0
    assert metrics["output_tok_s"] == 2.0
    assert metrics["chunk_itl_ms"] is None


def test_no_text_events_preserves_e2e_and_usage(
    successful_request: bench.RequestResult,
) -> None:
    metrics = bench.request_metrics(replace(successful_request, content_times=[]))
    assert metrics == {
        "ttft_ms": None,
        "tpot_ms": None,
        "chunk_itl_ms": None,
        "e2e_ms": 2000.0,
        "output_tok_s": 2.0,
    }


@pytest.mark.parametrize(
    "field,value,ttft",
    [
        ("started_at", None, None),
        ("started_at", float("nan"), None),
        ("finished_at", None, 500.0),
        ("finished_at", float("inf"), 500.0),
    ],
)
def test_missing_boundary_only_invalidates_dependent_metrics(
    successful_request: bench.RequestResult,
    field: str,
    value: float | None,
    ttft: float | None,
) -> None:
    metrics = bench.request_metrics(replace(successful_request, **{field: value}))
    assert metrics["ttft_ms"] == ttft
    assert metrics["chunk_itl_ms"] == 500.0
    assert metrics["e2e_ms"] is None
    assert metrics["tpot_ms"] is None
    assert metrics["output_tok_s"] is None


@pytest.mark.parametrize(
    "times",
    [
        [10.5, 10.4],
        [9.9, 11.0],
        [10.5, 12.1],
        [10.5, float("nan")],
        [float("inf")],
    ],
)
def test_invalid_text_timeline_is_not_repaired_into_valid_latency(
    successful_request: bench.RequestResult,
    times: list[float],
) -> None:
    metrics = bench.request_metrics(replace(successful_request, content_times=times))
    assert metrics["ttft_ms"] is None
    assert metrics["tpot_ms"] is None
    assert metrics["chunk_itl_ms"] is None
    assert metrics["e2e_ms"] == 2000.0
    assert metrics["output_tok_s"] == 2.0


def test_zero_elapsed_time_does_not_divide_by_zero(
    successful_request: bench.RequestResult,
) -> None:
    result = replace(successful_request, finished_at=10.0, content_times=[10.0, 10.0])
    assert bench.request_metrics(result) == {
        "ttft_ms": 0.0,
        "e2e_ms": 0.0,
        "tpot_ms": 0.0,
        "chunk_itl_ms": 0.0,
        "output_tok_s": None,
    }


def test_reversed_start_and_finish_have_no_valid_metrics(
    successful_request: bench.RequestResult,
) -> None:
    result = replace(successful_request, finished_at=9.0)
    assert bench.request_metrics(result) == dict.fromkeys(bench.METRICS)


def test_summarize_round_matches_hand_calculated_percentiles_and_throughput(
    successful_request: bench.RequestResult,
) -> None:
    """两条成功、一条部分生成后失败：明确检验插值及整轮吞吐的分母。"""
    first = replace(
        successful_request,
        started_at=0.0,
        finished_at=2.0,
        content_times=[0.5, 1.0],
        output_tokens=3,
    )
    second = replace(
        successful_request,
        request_id=1,
        started_at=0.0,
        finished_at=4.0,
        content_times=[1.5, 2.5, 3.0],
        output_tokens=5,
    )
    failed = replace(first, request_id=2, success=False, output_tokens=100)
    # 故意打乱输入顺序，失败请求的 100 token 不得计入成功吞吐。
    batch = bench.BatchResult([second, failed, first], elapsed_s=5.0)
    original = asdict(batch)
    row = bench.summarize_round(batch, concurrency=2, round_index=3)
    assert list(row) == list(bench.SUMMARY_FIELDS)
    assert row == pytest.approx(
        {
            "concurrency": 2,
            "round": 3,
            "requests": 3,
            "succeeded": 2,
            "failed": 1,
            "failure_rate": 1 / 3,
            "elapsed_s": 5.0,
            "output_tokens": 8,
            "output_tok_s": 1.6,
            "request_per_s": 0.4,
            "ttft_ms_p50": 1000.0,
            "ttft_ms_p95": 1450.0,
            "ttft_ms_p99": 1490.0,
            "tpot_ms_p50": 687.5,
            "tpot_ms_p95": 743.75,
            "tpot_ms_p99": 748.75,
            "e2e_ms_p50": 3000.0,
            "e2e_ms_p95": 3900.0,
            "e2e_ms_p99": 3980.0,
            "output_tok_s_p50": 1.375,
            "output_tok_s_p95": 1.4875,
            "output_tok_s_p99": 1.4975,
            "chunk_itl_ms_p50": 625.0,
            "chunk_itl_ms_p95": 737.5,
            "chunk_itl_ms_p99": 747.5,
        }
    )
    assert asdict(batch) == original


def test_summarize_round_single_sample_has_identical_percentiles(
    successful_request: bench.RequestResult,
) -> None:
    row = bench.summarize_round(bench.BatchResult([successful_request], 4.0), 1, 1)
    assert row["failure_rate"] == 0.0
    assert row["request_per_s"] == 0.25
    assert row["output_tok_s"] == 1.0  # 单请求速度为 2，不能当成整轮吞吐。
    for percentile in (50, 95, 99):
        assert row[f"ttft_ms_p{percentile}"] == 500.0
        assert row[f"tpot_ms_p{percentile}"] == 500.0
        assert row[f"e2e_ms_p{percentile}"] == 2000.0
        assert row[f"output_tok_s_p{percentile}"] == 2.0
        assert row[f"chunk_itl_ms_p{percentile}"] == 500.0


def test_summarize_round_filters_missing_metrics_independently(
    successful_request: bench.RequestResult,
) -> None:
    one_token = replace(
        successful_request,
        request_id=1,
        started_at=100.0,
        finished_at=101.0,
        content_times=[100.25],
        output_tokens=1,
    )
    row = bench.summarize_round(
        bench.BatchResult([successful_request, one_token], 4.0), 2, 1
    )
    assert row["ttft_ms_p50"] == 375.0  # 两条请求都有 TTFT。
    assert row["e2e_ms_p50"] == 1500.0
    for percentile in (50, 95, 99):
        # 第二条没有 TPOT/chunk ITL，不得用 0 填充并拉低分位数。
        assert row[f"tpot_ms_p{percentile}"] == 500.0
        assert row[f"chunk_itl_ms_p{percentile}"] == 500.0


@pytest.mark.parametrize("tokens", [None, -1, True, 1.5])
def test_summarize_round_does_not_publish_partial_token_throughput(
    successful_request: bench.RequestResult,
    tokens: float | None,
) -> None:
    missing_usage = replace(successful_request, request_id=1, output_tokens=tokens)
    row = bench.summarize_round(
        bench.BatchResult([successful_request, missing_usage], 5.0),
        2,
        1,
    )
    assert row["requests"] == 2 and row["succeeded"] == 2
    assert row["output_tokens"] is None
    assert row["output_tok_s"] is None
    assert row["request_per_s"] == 0.4
    assert row["ttft_ms_p50"] == 500.0
    assert row["output_tok_s_p50"] == 2.0  # 此分布中只剩已知 usage 的那一条。


def test_summarize_round_all_failed_has_zero_success_throughput(
    successful_request: bench.RequestResult,
) -> None:
    failed = replace(successful_request, success=False, output_tokens=100)
    row = bench.summarize_round(bench.BatchResult([failed, failed], 2.0), 2, 1)
    assert row["requests"] == 2 and row["failed"] == 2 and row["succeeded"] == 0
    assert row["failure_rate"] == 1.0
    assert row["output_tokens"] == 0
    assert row["output_tok_s"] == 0.0
    assert row["request_per_s"] == 0.0
    assert all(
        row[f"{name}_p{p}"] is None for name in bench.METRICS for p in (50, 95, 99)
    )


def test_summarize_round_empty_batch_has_no_measured_rates() -> None:
    row = bench.summarize_round(bench.BatchResult([], 2.0), 1, 1)
    assert list(row) == list(bench.SUMMARY_FIELDS)
    assert row["requests"] == row["succeeded"] == row["failed"] == 0
    assert row["failure_rate"] is None
    assert row["output_tokens"] is None
    assert row["output_tok_s"] is None
    assert row["request_per_s"] is None


@pytest.mark.parametrize("elapsed_s", [0.0, -1.0, float("nan"), float("inf")])
def test_summarize_round_invalid_duration_does_not_create_rates(
    successful_request: bench.RequestResult,
    elapsed_s: float,
) -> None:
    row = bench.summarize_round(
        bench.BatchResult([successful_request], elapsed_s), 1, 1
    )
    assert row["elapsed_s"] == (0.0 if elapsed_s == 0 else None)
    assert row["request_per_s"] is None and row["output_tok_s"] is None
    assert row["output_tokens"] == 4
    assert row["failure_rate"] == 0.0
    assert row["ttft_ms_p50"] == 500.0


@pytest.mark.parametrize("concurrency,round_index", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_summarize_round_rejects_invalid_round_labels(
    concurrency: int,
    round_index: int,
) -> None:
    with pytest.raises(ValueError):
        bench.summarize_round(bench.BatchResult([], 1.0), concurrency, round_index)


def repeat_rows(
    concurrency: int,
    values: list[bench.MetricValue],
) -> list[dict[str, bench.MetricValue]]:
    """构造正式轮次汇总，其他指标故意缺列，以检查缺失值不会被当成 0。"""
    return [
        {"concurrency": concurrency, "round": i, "output_tok_s": value}
        for i, value in enumerate(values, 1)
    ]


def test_summarize_repeats_groups_and_sorts_without_mutating_input() -> None:
    """三轮和四轮分别手算奇/偶数样本中位数；打乱并发和轮次顺序。"""
    rows = repeat_rows(8, [100.0, 300.0, 200.0]) + repeat_rows(
        1, [10.0, 30.0, 20.0, 40.0]
    )
    rows.reverse()
    original = [row.copy() for row in rows]
    summaries = bench.summarize_repeats(rows)
    expected_metrics = [
        name for name in bench.SUMMARY_FIELDS if name not in ("concurrency", "round")
    ]
    assert [(row["concurrency"], row["metric"]) for row in summaries] == [
        (concurrency, name) for concurrency in (1, 8) for name in expected_metrics
    ]
    by_metric = {(row["concurrency"], row["metric"]): row for row in summaries}
    assert by_metric[(8, "output_tok_s")] == {
        "concurrency": 8,
        "metric": "output_tok_s",
        "median": 200.0,
        "min": 100.0,
        "max": 300.0,
    }
    assert by_metric[(1, "output_tok_s")] == {
        "concurrency": 1,
        "metric": "output_tok_s",
        "median": 25.0,
        "min": 10.0,
        "max": 40.0,
    }
    assert rows == original
    assert all(
        list(row) == ["concurrency", "metric", "median", "min", "max"]
        for row in summaries
    )


def test_summarize_repeats_preserves_zero_and_ignores_missing_values() -> None:
    summaries = bench.summarize_repeats(repeat_rows(1, [0.0, None, 10.0]))
    by_metric = {row["metric"]: row for row in summaries}
    assert by_metric["output_tok_s"] == {
        "concurrency": 1,
        "metric": "output_tok_s",
        "median": 5.0,
        "min": 0.0,
        "max": 10.0,
    }
    assert by_metric["ttft_ms_p99"] == {
        "concurrency": 1,
        "metric": "ttft_ms_p99",
        "median": None,
        "min": None,
        "max": None,
    }


@pytest.mark.parametrize(
    "invalid", [None, "missing", True, -1.0, float("nan"), float("inf")]
)
def test_summarize_repeats_all_invalid_values_remain_empty(
    invalid: bench.MetricValue,
) -> None:
    summaries = bench.summarize_repeats(repeat_rows(1, [invalid, invalid, invalid]))
    assert all(
        row["median"] is None and row["min"] is None and row["max"] is None
        for row in summaries
    )


def test_summarize_repeats_retains_failed_rounds() -> None:
    """全失败轮次的失败率 1、成功吞吐 0 仍然是真实记录，不能被丢弃。"""
    rows = repeat_rows(1, [0.0, 0.0, 0.0])
    for row in rows:
        row["failure_rate"] = 1.0
        row["succeeded"] = 0
    by_metric = {row["metric"]: row for row in bench.summarize_repeats(rows)}
    assert by_metric["failure_rate"]["median"] == 1.0
    assert by_metric["output_tok_s"]["median"] == 0.0
    assert by_metric["succeeded"]["max"] == 0


def test_summarize_repeats_empty_input_has_no_groups() -> None:
    assert bench.summarize_repeats([]) == []


def test_summarize_repeats_requires_three_rounds_in_every_group() -> None:
    rows = repeat_rows(1, [1.0, 2.0, 3.0]) + repeat_rows(8, [1.0, 2.0])
    with pytest.raises(ValueError, match="并发 8.*至少需要 3 轮"):
        bench.summarize_repeats(rows)


def test_summarize_repeats_rejects_duplicate_rounds() -> None:
    rows = repeat_rows(1, [1.0, 2.0, 3.0])
    rows[2]["round"] = 1
    with pytest.raises(ValueError, match="重复"):
        bench.summarize_repeats(rows)


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("concurrency", None),
        ("concurrency", 0),
        ("concurrency", True),
        ("concurrency", "1"),
        ("round", None),
        ("round", 0),
        ("round", True),
        ("round", "1"),
    ],
)
def test_summarize_repeats_rejects_bad_labels_and_warmup(
    field: str,
    invalid: bench.MetricValue,
) -> None:
    rows = repeat_rows(1, [1.0, 2.0, 3.0])
    rows[0][field] = invalid
    with pytest.raises(ValueError, match="正整数"):
        bench.summarize_repeats(rows)


def test_complete_benchmark_writes_repeat_csv_with_mock_sse(
    config: bench.BenchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """运行完整客户端（不替换核心函数），验证模拟 SSE 到多轮 CSV 的保存路径。"""
    original_client = httpx.AsyncClient
    posts = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": config.model}]})
        assert request.method == "POST" and request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        posts.append(payload)
        wire = (
            'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=wire
        )

    def client_factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return original_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(bench.httpx, "AsyncClient", client_factory)
    run_dir = asyncio.run(
        bench.run_benchmark(config, bench.load_prompts(config.prompts))
    )
    assert len(posts) == 44  # 每档预热 10 次、正式 3*4 次，共两档并发。
    with (run_dir / "summary.csv").open(encoding="utf-8", newline="") as stream:
        measured = list(csv.DictReader(stream))
    assert len(measured) == 6
    assert {row["requests"] for row in measured} == {"4"}
    assert {row["output_tokens"] for row in measured} == {"8"}
    with (run_dir / "repeats.csv").open(encoding="utf-8", newline="") as stream:
        repeats = list(csv.DictReader(stream))
    assert len(repeats) == 2 * (len(bench.SUMMARY_FIELDS) - 2)
    request_counts = [row for row in repeats if row["metric"] == "requests"]
    assert {row["concurrency"] for row in request_counts} == {"1", "2"}
    assert all(
        float(row[key]) == 4
        for row in request_counts
        for key in ("median", "min", "max")
    )
    # 单个文本事件没有 chunk ITL；空值应一路保存为空单元格，而非字符串 None 或 0。
    assert all(
        row["median"] == "" for row in repeats if row["metric"] == "chunk_itl_ms_p50"
    )
    events = [
        json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "completed"
