"""项目 0 在线压测客户端的学习框架。

已提供 CLI、数据校验、连接复用、预热/多轮流程和只追加的结果保存。
已实现流式请求、并发调度、单请求指标、每轮聚合和多轮统计。
先运行 ``python project0-baseline/scripts/bench.py --help`` 或 ``--dry-run``。
详细实现约定见同目录 README.md。客户端不导入 vLLM，也不加载模型。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from statistics import median
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[2]
MetricValue = float | int | str | None
METRICS = ("ttft_ms", "tpot_ms", "e2e_ms", "output_tok_s", "chunk_itl_ms")
SUMMARY_FIELDS = (
    "concurrency",
    "round",
    "requests",
    "succeeded",
    "failed",
    "failure_rate",
    "elapsed_s",
    "output_tokens",
    "output_tok_s",
    "request_per_s",
    *(f"{name}_{p}" for name in METRICS for p in ("p50", "p95", "p99")),
)


@dataclass(frozen=True)
class BenchConfig:
    """一次实验的固定配置；requests/warmup 均指每档并发、每轮的总请求数。"""

    model: str
    base_url: str
    prompts: Path
    output_dir: Path
    concurrency: tuple[int, ...]
    requests: int
    warmup: int
    rounds: int
    timeout: float
    temperature: float
    dry_run: bool


@dataclass(frozen=True)
class Prompt:
    """一条负载；text 为用户消息，output_tokens 为生成上限，不是实测长度。"""

    text: str
    output_tokens: int


@dataclass
class RequestResult:
    """单请求的原始观测，所有时点均使用 time.perf_counter()，单位为秒。

    request_id/prompt_index: 轮内请求序号及其在 prompt 集中的下标。
    started_at: 获得并发槽位之后、发送 HTTP 请求之前的时点。
    finished_at: 收齐 SSE 流的时点；失败时为捕获异常的时点。
    content_times: 每个非空文本 delta 的到达时点，不记录角色/usage/结束事件。
    output_tokens: usage.completion_tokens；缺失为 None，不能用 chunk 数代替。
    success/error: 请求是否完整成功以及失败原因；失败也要保留记录。
    output_text: 拼接后的文本，便于检查返回内容，可在正式大规模实验时裁剪。
    """

    request_id: int
    prompt_index: int
    success: bool = False
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    content_times: list[float] = field(default_factory=list)
    output_tokens: int | None = None
    output_text: str = ""


@dataclass
class BatchResult:
    """一轮原始结果；elapsed_s 为整轮墙钟耗时，含排队，不能累加请求耗时代替。"""

    requests: list[RequestResult]
    elapsed_s: float


def parse_args(argv: Sequence[str] | None = None) -> BenchConfig:
    """解析并校验 CLI。

    参数：argv 为参数列表，None 表示使用命令行。
    返回：BenchConfig。非法参数通过 argparse 报错并以状态码 2 退出。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="必须匹配 /v1/models 的 id")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000", help="服务根地址，不含 /v1"
    )
    parser.add_argument(
        "--prompts", type=Path, default=ROOT / "benchmarks/prompts/baseline-v1.jsonl"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "project0-baseline/results"
    )
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32, 64])
    parser.add_argument(
        "--requests", type=int, default=128, help="每个正式轮次的总请求数"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="每档并发先预热，至少 10 个请求"
    )
    parser.add_argument(
        "--rounds", type=int, default=3, help="每档并发正式轮数，至少 3 轮"
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="单请求超时秒数")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--dry-run", action="store_true", help="仅校验配置和 prompt，不联网、不写结果"
    )
    args = parser.parse_args(argv)
    url = urlsplit(args.base_url)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username
        or url.password
        or url.path not in ("", "/")
        or url.query
        or url.fragment
    ):
        parser.error(
            "--base-url 必须为不含认证信息、查询参数和 /v1 路径的 HTTP(S) 根地址"
        )
    if not args.model.strip():
        parser.error("--model 不能为空")
    if not args.concurrency or any(c < 1 for c in args.concurrency):
        parser.error("--concurrency 必须全部为正整数")
    if len(set(args.concurrency)) != len(args.concurrency):
        parser.error("--concurrency 不能重复")
    if args.requests < max(args.concurrency):
        parser.error("--requests 不能小于最大并发，否则无法达到该档并发")
    if args.warmup < 10 or args.rounds < 3:
        parser.error("基线要求 --warmup >= 10 且 --rounds >= 3")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout 必须为有限正数")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        parser.error("--temperature 必须是 0～2 之间的有限数")
    return BenchConfig(
        **{
            **vars(args),
            "base_url": args.base_url.rstrip("/"),
            "concurrency": tuple(args.concurrency),
        }
    )


def load_prompts(path: Path) -> list[Prompt]:
    """读取 JSONL 负载；每行包含非空 prompt 和正整数 output_tokens。

    参数：path 为 UTF-8 JSONL 文件；空行忽略。
    返回：按文件顺序排列的 Prompt 列表。
    异常：空文件/字段错误抛 ValueError（含行号）；读文件失败抛 OSError。
    """
    prompts = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                text, tokens = data["prompt"], data["output_tokens"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("prompt 必须为非空字符串")
                if type(tokens) is not int or tokens < 1:
                    raise ValueError("output_tokens 必须为正整数")
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_no}: 无效 prompt：{exc}") from exc
            prompts.append(Prompt(text=text, output_tokens=tokens))
    if not prompts:
        raise ValueError(f"prompt 集为空：{path}")
    return prompts


async def send_request(
    client: httpx.AsyncClient,
    config: BenchConfig,
    prompt: Prompt,
    request_id: int,
    prompt_index: int,
) -> RequestResult:
    """TODO 1：发送一个 OpenAI Chat Completions 流式请求并收集原始观测。

    参数：client 是共享连接池；config 提供模型/超时/采样参数；prompt 是负载；
        request_id/prompt_index 用于在原始记录中追溯请求。
    返回：RequestResult；HTTP、超时、SSE 错误转成 success=False 和 error。
        不要吞掉 NotImplementedError 或 asyncio.CancelledError。
    """

    result = RequestResult(
        request_id=request_id,
        prompt_index=prompt_index,
    )
    # 实现提示：

    # 将 Python 字典交给 httpx，它会序列化成 JSON 请求体。
    payload: dict[str, object] = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": prompt.text,
            }
        ],
        "temperature": config.temperature,
        "max_tokens": prompt.output_tokens,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }
    # 1. 用 perf_counter 记录 started_at；此函数应在取得并发槽位之后调用。
    result.started_at = perf_counter()

    # 2. client.stream("POST", "/v1/chat/completions", json=...)，复用 client。
    #    请求包含 model/messages/temperature/max_tokens/stream=True，以及
    #    stream_options={"include_usage": True}，max_tokens 来自 prompt.output_tokens。
    #    限制整个请求的耗时，后续读取响应流也要放在这个作用域内。
    try:
        async with asyncio.timeout(config.timeout):
            # 发送请求，并在作用域结束时自动关闭响应流、释放连接。
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
            ) as response:

                # 尽早发现模型名错误、服务端错误等 HTTP 层面的问题。
                response.raise_for_status()

                # 暂存当前事件的 data 字段，遇到空行再处理。
                data_lines: list[str] = []
                saw_done = False

                # 用 response.raise_for_status() 和 response.aiter_lines() 处理 SSE。
                async for line in response.aiter_lines():
                    # aiter_lines() 已去掉行尾换行符，因此空行表现为 ""。
                    if line == "":
                        if not data_lines:
                            continue

                        event_data = "\n".join(data_lines)
                        data_lines.clear()

                        if not event_data:
                            continue

                        # [DONE] 是接口的结束标记，不是 JSON。
                        if event_data == "[DONE]":
                            saw_done = True
                            result.finished_at = perf_counter()
                            break

                        # 先记下完整事件到达的时间，避免把后续 JSON 处理耗时算进去。
                        received_at = perf_counter()

                        event = json.loads(event_data)
                        if not isinstance(event, dict):
                            raise ValueError("SSE 数据必须是 JSON 对象")

                        # HTTP 状态成功之后，流中仍可能返回错误事件。
                        if "error" in event:
                            raise ValueError(f"服务端返回流式错误：{event['error']}")

                        # 读取实际输出 token 数；usage 可能不存在，也可能为 null。
                        usage = event.get("usage")
                        if usage is not None:
                            if not isinstance(usage, dict):
                                raise ValueError("usage 必须是 JSON 对象")

                            tokens = usage.get("completion_tokens")
                            if type(tokens) is not int or tokens < 0:
                                raise ValueError("completion_tokens 必须是非负整数")

                            result.output_tokens = tokens

                        # usage 事件的 choices 可以是空列表。
                        choices = event.get("choices", [])
                        if not isinstance(choices, list):
                            raise ValueError("choices 必须是列表")

                        if choices:
                            choice = choices[0]
                            if not isinstance(choice, dict):
                                raise ValueError("choice 必须是 JSON 对象")

                            delta = choice.get("delta", {})
                            if not isinstance(delta, dict):
                                raise ValueError("delta 必须是 JSON 对象")

                            content = delta.get("content")
                            if content is not None and not isinstance(content, str):
                                raise ValueError("delta.content 必须是字符串或 null")

                            # 角色声明、空文本和结束事件不应作为首个文本到达。
                            if content:
                                result.content_times.append(received_at)
                                result.output_text += content

                        continue

                    # 忽略 SSE 注释，例如 ": keep-alive"。
                    if line.startswith(":"):
                        continue

                    # 只按第一个冒号拆分，避免破坏 JSON 中的冒号。
                    field, _, value = line.partition(":")

                    if field == "data":
                        # SSE 规则：只移除冒号后紧跟的一个空格。
                        value = value.removeprefix(" ")

                        data_lines.append(value)

                # 网络流结束，不代表服务端正常完成了生成。
                if not saw_done:
                    raise ValueError("响应流在收到 [DONE] 前结束")

                # 文本压测需要首个有效文本，才能计算 TTFT。
                if not result.content_times:
                    raise ValueError("响应没有返回非空文本，无法计算 TTFT")

                # 必须使用服务端实际计数，不能用生成上限或 chunk 数代替。
                if result.output_tokens is None:
                    raise ValueError("响应缺少 usage.completion_tokens")

                if result.output_tokens <= 0:
                    raise ValueError("已收到文本，但 completion_tokens 不大于 0")


        # success 默认是 False，只有完整处理成功后才改为 True。
        result.success = True
    except (httpx.HTTPError, TimeoutError, ValueError) as exc:
        # 保留已经收到的文本和时间戳，便于排查部分成功的请求。
        result.success = False
        result.error = f"{type(exc).__name__}: {exc}"
        result.finished_at = perf_counter()

    finally:
        # 取消或其他未捕获异常发生时，也尽可能补上结束时间。
        if result.finished_at is None:
            result.finished_at = perf_counter()

    return result



async def run_round(
    client: httpx.AsyncClient,
    config: BenchConfig,
    prompts: list[Prompt],
    concurrency: int,
    count: int,
) -> BatchResult:
    """执行固定并发的一轮请求，预热与正式轮均复用此接口。

    参数：client/config/prompts 供 send_request 使用；concurrency 是在途请求上限；
        count 是本轮总请求数，并非每个并发各发 count 次。
    返回：包含全部 count 条记录及整轮墙钟耗时的 BatchResult。

    使用 asyncio.Semaphore 限制在途请求数，用 gather 收集全部结果。
    第 i 个请求选 prompts[i % len(prompts)]；每轮从同一顺序开始以固定负载。
    TTFT 从取得槽位后计时；整轮 elapsed_s 从开始调度前计至所有请求结束。
    单请求失败不能取消其余请求，取消实验时则应取消并等待所有在途任务退出。
    返回 success=False 属于已记录的请求失败；意外异常和任务取消在清理后继续抛出。
    """
    # 防止没有并发槽位、没有请求或没有可用 prompt。
    if concurrency < 1:
        raise ValueError("concurrency 必须大于 0")
    if count < 1:
        raise ValueError("count 必须大于 0")
    if not prompts:
        raise ValueError("prompts 不能为空")

    # 所有请求共享同一个信号量，最多同时执行 concurrency 个请求。
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(request_id: int) -> RequestResult:
        """为一个请求选择 prompt，取得并发槽位后执行并返回结果。"""
        prompt_index = request_id % len(prompts)
        prompt = prompts[prompt_index]

        # 没有空闲槽位时在这里等待；退出作用域时自动释放槽位。
        async with semaphore:
            return await send_request(
                client=client,
                config=config,
                prompt=prompt,
                request_id=request_id,
                prompt_index=prompt_index,
            )

    # 从创建任务之前开始计时，包含任务调度、等待槽位和请求执行。
    round_started_at = perf_counter()

    # 为本轮的 count 个请求创建任务。
    tasks: list[asyncio.Task[RequestResult]] = [
        asyncio.create_task(run_one(request_id))
        for request_id in range(count)
    ]

    try:
        # success=False 仍是正常返回值，不会中断其他请求。
        results = await asyncio.gather(*tasks)
    except BaseException:
        # gather 遇到子任务异常时，其他任务可能仍在执行或等待 semaphore。
        # 此处也处理 CancelledError：它不属于 Exception，而属于 BaseException。
        for task in tasks:
            # 已经在取消中的任务可能正在异步释放连接，避免重复 cancel 打断清理。
            if not task.done() and not task.cancelling():
                task.cancel()

        # cancel() 只是发出取消请求；还必须等待子任务退出、释放槽位和响应流。
        # return_exceptions=True 仅用于收尾，避免清理时的异常盖过最初的异常。
        await asyncio.gather(*tasks, return_exceptions=True)
        raise  # 清理后继续传播原异常或取消，不返回伪造的完整轮次结果。

    elapsed_s = perf_counter() - round_started_at

    return BatchResult(
        requests=results,
        elapsed_s=elapsed_s,
    )


def request_metrics(result: RequestResult) -> dict[str, float | None]:
    """将单请求原始观测转换为指标，返回键名见 METRICS。

    参数：result 为一条 RequestResult；原始时点单位为秒。
    返回：ttft_ms/tpot_ms/e2e_ms/output_tok_s/chunk_itl_ms 的字典。
        时延单位为毫秒，输出速度单位为 token/秒；不修改原始结果。
        失败请求全部留空；分母为零或数据缺失时相应值为 None。

    约定：TTFT=首个有效文本时点-started_at；E2E=finished_at-started_at；
    TPOT=(E2E-TTFT)/(实际输出 token 数-1)，输出不超过 1 token 时无定义；
    每请求 output_tok_s=实际输出 token 数/E2E（秒）。
    chunk_itl_ms 为相邻文本 delta 间隔的均值；它不是精确的逐 token ITL。
    非有限、倒序或越界的文本时点不能用于 TTFT/TPOT/chunk ITL；
    E2E 和输出速度仍可由独立有效的数据计算。有效的零时延保留为 0。
    """
    # 先为全部指标保留空值，之后只填写具有有效观测依据的指标。
    # None 在 CSV 中为空单元格，不会被误认为“测得时延为 0”。
    metrics: dict[str, float | None] = dict.fromkeys(METRICS)
    if not result.success:
        return metrics

    # 缺失、NaN 或无穷大的起止时间不能参与相减，但不影响独立的 chunk 观测。
    started_at = result.started_at
    finished_at = result.finished_at
    if started_at is not None and not math.isfinite(started_at):
        started_at = None
    if finished_at is not None and not math.isfinite(finished_at):
        finished_at = None

    # E2E：整个请求的耗时。中间计算始终使用秒，写入时延字段时才乘 1000。
    e2e_s = None
    if started_at is not None and finished_at is not None and finished_at >= started_at:
        e2e_s = finished_at - started_at
        metrics["e2e_ms"] = e2e_s * 1000

    # 文本事件应按到达顺序记录，且落在已知的请求起止时点之间。
    # 不排序、不截断负数：这样不会把采集错误掩盖成有效的正时延。
    times = result.content_times
    valid_content_times = bool(times) and all(math.isfinite(t) for t in times)
    if valid_content_times:
        valid_content_times = all(
            later >= earlier for earlier, later in pairwise(times)
        )
        if started_at is not None and times[0] < started_at:
            valid_content_times = False
        if finished_at is not None and times[-1] > finished_at:
            valid_content_times = False

    # TTFT：从发送前到首个非空文本事件，不将角色声明等事件算作文本。
    ttft_s = None
    if valid_content_times and started_at is not None:
        ttft_s = times[0] - started_at
        metrics["ttft_ms"] = ttft_s * 1000

    # chunk ITL：n 个文本事件有 n-1 个间隔；单个事件无法计算间隔。
    # 即使 token usage 缺失，这些客户端观测到的间隔仍然有效。
    if valid_content_times and len(times) > 1:
        gaps_s = [later - earlier for earlier, later in pairwise(times)]
        metrics["chunk_itl_ms"] = math.fsum(gaps_s) / len(gaps_s) * 1000

    # token 数只能来自实际 usage。bool 虽是 int 的子类，也不应当作计数接受。
    tokens = result.output_tokens
    if type(tokens) is not int or tokens < 0:
        return metrics

    # 单请求输出速度：分母用本请求 E2E 的秒数，而不是毫秒，也不是整轮耗时。
    # 真正测得 0 token 且耗时为正时，速度为 0；缺失 token 计数则已在上面返回。
    if e2e_s is not None and e2e_s > 0:
        metrics["output_tok_s"] = tokens / e2e_s

    # TPOT：首个 token 之后的平均每 token 耗时，分母为实际 token 数减 1。
    # 使用本项目约定的 E2E-TTFT，包含末尾 usage/DONE 的等待，不等于 chunk ITL。
    if e2e_s is not None and ttft_s is not None and tokens > 1:
        metrics["tpot_ms"] = (e2e_s - ttft_s) / (tokens - 1) * 1000

    return metrics


def summarize_round(
    batch: BatchResult,
    concurrency: int,
    round_index: int,
) -> dict[str, MetricValue]:
    """聚合一个正式轮次，输出一行 summary.csv 数据，不修改 batch。

    参数：batch 为原始轮次；concurrency 为并发档位；round_index 从 1 开始。
    返回：字段与 SUMMARY_FIELDS 完全一致的字典。

    调用 request_metrics，只对成功请求的有效指标计算 p50/p95/p99。
    分位数采用线性插值：将 n 个值排序，在下标 (n-1)*p 处取值；
    空样本留空，单样本的三个分位数均为该值，不需要增加第三方依赖。
    failure_rate=失败数/总数；request_per_s=成功数/整轮 elapsed_s；
    output_tok_s=成功请求的实际输出 token 总数/整轮 elapsed_s。
    全失败时仍输出失败率；时延分位数留空。若成功请求缺 token usage，
    token 总数及吞吐应留空，不能用部分 token 数冒充完整吞吐。
    空轮次的失败率和吞吐留空；非正/非有限耗时不用于计算吞吐。
    异常：concurrency 或 round_index 小于 1 时抛 ValueError。
    """
    if concurrency < 1:
        raise ValueError("concurrency 必须大于 0")
    if round_index < 1:
        raise ValueError("round_index 必须大于 0；预热不进入正式统计")

    # 先创建全部 CSV 字段，确保空样本和异常数据也不会改变输出结构。
    row: dict[str, MetricValue] = dict.fromkeys(SUMMARY_FIELDS)
    successful = [result for result in batch.requests if result.success]
    request_count = len(batch.requests)
    success_count = len(successful)
    failure_count = request_count - success_count
    elapsed_s: float | None = batch.elapsed_s
    if not math.isfinite(elapsed_s) or elapsed_s < 0:
        elapsed_s = None

    row.update({
        "concurrency": concurrency,
        "round": round_index,
        "requests": request_count,
        "succeeded": success_count,
        "failed": failure_count,
        "elapsed_s": elapsed_s,
    })
    if request_count == 0:
        # 没有发送请求，不应将“没有测量”表示成 0 失败率或 0 吞吐。
        return row
    row["failure_rate"] = failure_count / request_count  # 0～1 的比例，不是百分数。

    # 任意成功请求缺失/带有无效的 usage，完整 token 总数就未知。
    # 失败请求即使生成了部分文本，也不计入成功请求的输出 token 总数。
    valid_token_counts = all(
        type(result.output_tokens) is int and result.output_tokens >= 0
        for result in successful
    )
    total_tokens: int | None = None
    if valid_token_counts:
        # 全失败时 successful 为空，sum 得到 0，表示本轮确实没有成功输出。
        # None 过滤用于类型收窄；上面的完整性检查已保证这里不会遗漏任何有效计数。
        total_tokens = sum(
            result.output_tokens for result in successful if result.output_tokens is not None
        )
        row["output_tokens"] = total_tokens

    # 整轮吞吐的分母是墙钟耗时，不能累加请求耗时或取每请求速度的均值。
    # 零耗时可以保留在 elapsed_s 中，但不能作除数。
    if elapsed_s is not None and elapsed_s > 0:
        row["request_per_s"] = success_count / elapsed_s
        if total_tokens is not None:
            row["output_tok_s"] = total_tokens / elapsed_s

    # 每个请求只计算一次指标；各指标独立过滤缺失值，样本量可能不同。
    # 例如 1-token 请求没有 TPOT，但其 TTFT 和 E2E 仍可参与统计。
    per_request = [request_metrics(result) for result in successful]
    for name in METRICS:
        values: list[float] = []
        for metrics in per_request:
            value = metrics[name]
            if value is not None and math.isfinite(value) and value >= 0:
                values.append(value)
        values.sort()
        if not values:
            continue  # 保留 None，CSV 中显示为空，不将缺失值补成 0。
        for percentile in (50, 95, 99):
            # 分数下标落在两个样本之间时，按距离线性插值。
            # 例：[100, 300] 的 p95 下标为 0.95，结果为 100 + 0.95*200 = 290。
            position = (len(values) - 1) * percentile / 100
            lower = math.floor(position)
            upper = math.ceil(position)
            weight = position - lower
            row[f"{name}_p{percentile}"] = (
                values[lower] + (values[upper] - values[lower]) * weight
            )

    return row


def summarize_repeats(
    rows: list[dict[str, MetricValue]],
) -> list[dict[str, MetricValue]]:
    """按并发分组，统计至少 3 个正式轮次的中位数及波动，不修改输入。

    参数：rows 为所有正式轮次的 summary 行，不包含预热。
    返回：长表，每行键为 concurrency/metric/median/min/max，写入 repeats.csv。
    遍历 SUMMARY_FIELDS 中除 concurrency/round 外的数值列；缺失值不当成 0。
    每档并发按不同的正整数 round 编号计数，至少 3 轮；重复编号或预热编号 0
    抛 ValueError。各指标独立过滤无效值，因此实际有效样本可能少于轮数。
    指标全缺失时仍保留一行空统计；空 rows 返回空列表。
    输出按并发数升序、SUMMARY_FIELDS 中的指标顺序排列。
    每轮 p99 的中位数不等于混合全部请求后计算的 p99，报告时需注明口径。
    """
    groups: dict[int, list[dict[str, MetricValue]]] = {}
    seen_rounds: dict[int, set[int]] = {}
    for row in rows:
        concurrency = row.get("concurrency")
        round_index = row.get("round")
        # rows 来自 summarize_round 的内存结果，不自动把 CSV 字符串转成数字。
        # 拒绝布尔值、非法标签和 round=0，避免预热混入或不同配置被错误分组。
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("每轮 concurrency 必须为正整数")
        if type(round_index) is not int or round_index < 1:
            raise ValueError("每轮 round 必须为正整数，不能包含预热轮次")
        rounds = seen_rounds.setdefault(concurrency, set())
        if round_index in rounds:
            raise ValueError(f"并发 {concurrency} 的 round={round_index} 重复")
        rounds.add(round_index)
        groups.setdefault(concurrency, []).append(row)

    summaries: list[dict[str, MetricValue]] = []
    for concurrency in sorted(groups):
        group = groups[concurrency]
        if len(group) < 3:
            raise ValueError(f"并发 {concurrency} 只有 {len(group)} 个正式轮次，至少需要 3 轮")

        # 标识列不做统计；请求数、失败率、吞吐和各分位数均逐列汇总。
        # 每轮权重相同，不按请求数加权，也不将所有轮次的原始请求重新混合。
        for name in SUMMARY_FIELDS:
            if name in ("concurrency", "round"):
                continue
            values: list[float] = []
            for row in group:
                value = row.get(name)
                # None/缺列/字符串/布尔值/非有限数/负数均视为无效观测。
                # 有效的 0 必须保留，例如全失败轮次的成功吞吐就是 0。
                if (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value >= 0):
                    values.append(value)

            summaries.append({
                "concurrency": concurrency,
                "metric": name,
                # median 在偶数样本时取中间两项的均值；min/max 表示观测范围。
                # 没有有效值就保留 None，CSV 中为空，不编造为 0。
                "median": median(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            })

    return summaries


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    """将 row 追加为一行 UTF-8 JSON；不覆盖已有记录，写入异常向上传递。"""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def append_csv(path: Path, row: dict[str, MetricValue], fields: Sequence[str]) -> None:
    """按 fields 追加一行 CSV；新文件写表头，None 写为空单元格。"""
    needs_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def create_run_dir(config: BenchConfig) -> Path:
    """创建唯一结果子目录并保存不可变 metadata.json，返回目录路径。

    参数：config 为本次配置。目录名使用 UTC 时间和随机后缀，不覆盖历史。
    元数据记录 prompt SHA256、客户端版本及计时口径；服务端环境需另填实验报告。
    """
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8]
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "prompts_sha256": hashlib.sha256(config.prompts.read_bytes()).hexdigest(),
        "client_python": sys.version,
        "client_httpx": version("httpx"),
        "measurement": "closed-loop; TTFT excludes slot wait; E2E includes stream end",
        "server_environment": "未自动采集；请在实验报告记录版本、驱动、完整启动命令及显存峰值",
    }
    with (run_dir / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, default=str)
    return run_dir


async def run_benchmark(config: BenchConfig, prompts: list[Prompt]) -> Path:
    """运行完整实验流程，返回保存目录；异常记录到 events.jsonl 后继续抛出。

    参数：config 为已校验配置，prompts 为 load_prompts 的非空返回值。
    每档并发独立预热，再执行正式轮；每轮原始结果先落盘，再计算汇总。
    API key 仅从 VLLM_API_KEY 读取，不写入配置或结果；禁用环境代理以直连服务。
    """
    run_dir = create_run_dir(config)
    print(f"结果目录：{run_dir}")
    events = run_dir / "events.jsonl"
    append_jsonl(events, {"event": "started"})
    headers = {}
    if api_key := os.getenv("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    rows = []
    try:
        async with httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            headers=headers,
            limits=httpx.Limits(
                max_connections=max(config.concurrency),
                max_keepalive_connections=max(config.concurrency),
            ),
            trust_env=False,
        ) as client:
            response = await client.get("/v1/models")
            response.raise_for_status()
            model_ids = [item["id"] for item in response.json()["data"]]
            if config.model not in model_ids:
                raise ValueError(f"模型名不匹配；服务当前提供：{model_ids}")
            for concurrency in config.concurrency:
                for round_index in range(config.rounds + 1):
                    phase = "warmup" if round_index == 0 else "measured"
                    count = config.warmup if phase == "warmup" else config.requests
                    print(
                        f"concurrency={concurrency} phase={phase} round={round_index}"
                    )
                    batch = await run_round(client, config, prompts, concurrency, count)
                    # 保存预热数据便于排查，但正式统计不会包含预热记录。
                    for result in batch.requests:
                        append_jsonl(
                            run_dir / "requests.jsonl",
                            {
                                "phase": phase,
                                "concurrency": concurrency,
                                "round": round_index,
                                **asdict(result),
                            },
                        )
                    append_jsonl(
                        events,
                        {
                            "event": "batch_completed",
                            "phase": phase,
                            "concurrency": concurrency,
                            "round": round_index,
                            "elapsed_s": batch.elapsed_s,
                        },
                    )
                    if phase == "measured":
                        row = summarize_round(batch, concurrency, round_index)
                        append_csv(run_dir / "summary.csv", row, SUMMARY_FIELDS)
                        rows.append(row)
        for row in summarize_repeats(rows):
            append_csv(
                run_dir / "repeats.csv",
                row,
                ("concurrency", "metric", "median", "min", "max"),
            )
    except BaseException as exc:
        # 保留失败/中断证据，并让程序以失败状态退出；不生成零值替代指标。
        append_jsonl(
            events,
            {"event": "aborted", "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    append_jsonl(events, {"event": "completed"})
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口；返回 0 表示校验/实验成功，1 表示失败，130 表示用户中断。"""
    config = parse_args(argv)
    try:
        prompts = load_prompts(config.prompts)
        if config.dry_run:
            print(
                json.dumps(
                    {
                        "config": asdict(config),
                        "prompt_count": len(prompts),
                        "status": "配置校验通过；未运行请求或指标统计",
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
            return 0
        asyncio.run(run_benchmark(config, prompts))
    except (NotImplementedError, ValueError, OSError, httpx.HTTPError) as exc:
        print(f"压测未完成：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("压测已中断，请查看结果目录中的 events.jsonl。", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
