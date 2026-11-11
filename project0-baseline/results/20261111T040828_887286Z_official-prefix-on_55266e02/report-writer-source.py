"""Write a new immutable experiment report after the full audit succeeds."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> None:
    out = Path(sys.argv[1]).resolve()
    root = Path.cwd()
    stats = json.loads((out / "report-stats.json").read_text())
    audit = json.loads((out / "audit.json").read_text())
    context = json.loads((out / "context.json").read_text())
    assert audit["passed"]
    tables = (out / "tables.md").read_text().strip().split("\n\n")
    official_table, comparison_table, round_table = tables
    rel = "../results/" + out.name
    pct = 100 * stats["formal_prefix_hit_tokens"] / stats["formal_prefix_query_tokens"]
    conclusion = (
        "四档输出吞吐的三轮中位数全部落在自写脚本的 ±20% 范围内，符合项目 0 的量级交叉验证要求。"
        if audit["all_throughput_within_20_percent"]
        else "部分档位输出吞吐超出自写脚本的 ±20% 范围，不能判定全部通过交叉验证，需检查差异。"
    )
    with (out / "official-rounds.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    anomalies = [row for row in rows if int(row["ttft_over_1s"]) > 0]
    anomaly_text = (
        "本轮官方正式请求没有 TTFT 超过 1 秒的记录。"
        if not anomalies
        else "本轮官方也观察到 TTFT 超过 1 秒的请求："
        + "、".join(
            f"并发 {row['concurrency']} {row['phase']}：{row['ttft_over_1s']} 次"
            for row in anomalies
        )
        + "。原因未定位，全部保留。"
    )
    tail_differences = stats["ttft_p99_difference_over_20_percent"]
    tail_text = (
        "各档 TTFT p99 的三轮中位数差异也在 ±20% 内；单轮波动仍需另看。"
        if not tail_differences
        else "TTFT p99 的三轮中位数并非全部在 ±20% 内："
        + "；".join(
            f"并发 {row['concurrency']}，自写 {row['custom_ms']:.2f} ms，官方 {row['official_ms']:.2f} ms，差异 {row['difference_percent']:+.2f}%"
            for row in tail_differences
        )
        + "。因此通过的是输出吞吐的量级检查，不能声称所有延迟分位数均通过 ±20% 对齐。尾延迟差异尚未通过交错试验定位。"
    )
    with (out / "comparison.csv").open() as stream:
        compare = list(csv.DictReader(stream))
    for row in compare:
        if abs(float(row["ttft_ms_p50_difference_percent"])) > 20:
            tail_text += (
                f" 此外，并发 {row['concurrency']} 的 TTFT p50 为自写 "
                f"{float(row['custom_ttft_ms_p50_median']):.2f} ms、官方 "
                f"{float(row['official_ttft_ms_p50_median']):.2f} ms，差异 "
                f"{float(row['ttft_ms_p50_difference_percent']):+.2f}%。"
            )
    text = f"""# 2026-11-11：官方 vLLM 工具与自写客户端交叉验证（前缀缓存开启）

## 元数据与范围

{conclusion} 最大绝对差异为 **{stats["max_absolute_throughput_difference_percent"]:.2f}%**。
这是同一运行中服务、相同负载的量级核对；两次实验跨日顺序执行，没有进行交错配对试验。

| 项目 | 实测/记录值 |
|---|---|
| 官方测量时间 | {stats["started_local"]} — {stats["finished_local"]}，Asia/Shanghai |
| 对照 | 2026-11-10 自写 `project0-baseline/scripts/bench.py`，保留原三轮数据 |
| 工具变量 | 自写 httpx 客户端 → 安装版本自带的原生 `vllm bench serve`（aiohttp） |
| 固定条件 | 物理 GPU1、TP=1、模型、服务 PID、启动参数、输入顺序、生成参数、并发档位 |
| 官方版本 | vLLM {context["packages"]["vllm"]}；本轮未修改安装包/上游代码 |
| 预热与正式 | 每档先遍历 128 条预热，再测 3 × 512；4 档共 512 次预热 + 6144 次正式 |
| 成功/失败 | 16 次 CLI 调用全部退出 0；6656 个请求全部成功，无服务端抢占增长 |
| 缓存 | 实验前后指标均为 enable_prefix_caching=True；未清空/重启服务 |
| 关闭缓存组 | 未运行；本报告不能给出缓存开关的因果收益 |

## 环境与服务

- 2 × RTX 4090 24564 MiB，无 NVLink；本轮仅物理 GPU1，GPU0 是其他用户的 SGLang。
- 驱动 580.178.04，驱动支持 CUDA 13.0；PyTorch 2.13.0+cu130，uv Python 3.12.16。
- vLLM / vLLM-Omni 0.32.0，transformers {context["packages"]["transformers"]}，aiohttp {context["packages"]["aiohttp"]}。
- 官方 custom 数据加载补装 pandas {context["packages"]["pandas"]} 与 pytz {context["packages"]["pytz"]}，已写入开发依赖与环境文档；没有升级推理框架。
- 模型 `models/Qwen2.5-7B-Instruct`，模型配置 dtype=bfloat16。配置、generation_config、tokenizer_config 的 SHA256 与上一轮相同。
- API PID 439160，EngineCore PID 439648；实验前后仍为同一 API 进程与完整启动参数。
- KV 指标仍为 103840 token、6490 个 16-token block；chunked prefill 的有效状态未从配置接口直接验证。

实际服务命令（命令中没有显式 prefix cache 开关；开启状态由实时指标确认）：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python .venv/bin/vllm serve models/Qwen2.5-7B-Instruct \\
  --port 8000 --tensor-parallel-size 1 --max-model-len 8192 \\
  --gpu-memory-utilization 0.90 --max-num-seqs 64
```

完整客户端命令和环境白名单见 [context.json]({rel}/context.json)。
输入加载检查和模型配置哈希见 [alignment-check.json]({rel}/alignment-check.json)。
安装包源码快照与 SHA256 已保存；本机没有 external/vllm checkout，本轮以实际安装版本代码为准。

## 负载对齐

- 原始数据为 `benchmarks/prompts/baseline-v1.jsonl`，128 条、8 类任务；SHA256 为 `a624abd107e8bb3abf3737d4dc33bab4443c678292c721beca42c02435b9c009`。
- 每轮将原始输入循环 4 次成为 512 条；派生 JSONL 只改变条目数量，prompt 文本、顺序、输出上限均保持原样。
- 本地默认聊天模板长度为 59—379 token、中位数 78.5；服务端应用模板一次。
- 官方指定 `--disable-shuffle --no-oversample --skip-chat-template`，避免默认乱序、随机过采样和客户端预套聊天模板。
- temperature=0，生成上限 128，EOS 可提前结束；没有指定 ignore_eos。其余采样参数继承同一个服务的模型默认配置。
- 官方通过 max_completion_tokens=128，自写通过 max_tokens=128 设置相同生成上限。
- 每档用独立的官方 CLI 遍历 128 条作为预热，再调用 3 次正式 CLI；`--num-warmups 0`，避免官方内置预热只重复第一条输入。
- `--request-rate inf --max-concurrency C`；无探针、无额外就绪生成请求，在途数由信号量限制，排队时间不进入单请求 TTFT/E2E。
- 原生 CLI 在每次调用前加载本地 tokenizer；这段启动时间不进入其 benchmark duration。自写脚本的正式统计同样以整轮调度与请求完成为界。

## 官方三轮汇总

下表为每轮指标的三轮中位数，方括号为三轮最小/最大值；不是合并请求后的分位数。
GPU1 每秒采样并以原始 start_times/latencies 筛选正式阶段；显存是整卡已用量，可能漏掉瞬时尖峰。

{official_table}

## 与自写脚本对照

差异 = (官方三轮中位数 / 自写三轮中位数 − 1) × 100%。两侧均使用实际输出 token 统计整轮吞吐。

{comparison_table}

![官方与自写客户端：三轮中位数和观察范围]({rel}/official-vs-custom.png)

误差线是三轮观察范围，不是置信区间。TTFT p99 使用对数轴，包含上一轮自写并发 64 第二轮的异常值，没有删除异常轮次。

## 官方每轮明细

{round_table}

## 分析与限制

1. **吞吐交叉验证**：{conclusion} 这支持自写脚本的 token 计数、并发和整轮计时没有出现数量级错误；不等于证明所有异常路径都正确。
2. **单请求指标与尾延迟**：{tail_text} 官方 TTFT 取首个带 choices 的事件，可能是空文本 role 事件；自写取首个非空文本 delta。官方 E2E 到最后一个 choices 事件，自写到 DONE。两者 TPOT 公式相同，但端点时刻略有差别。这些口径差异不构成已证实的尾延迟差异原因。
3. **ITL 不能直接对等**：官方汇集所有请求的事件间隔后计算分位数；自写 chunk_itl_ms 是每请求文本事件间隔均值，再对请求取分位数。事件也不一定对应单个 token。完整官方 ITL、E2E 分位数保留在 CSV/JSON 中。
4. **超时配置差异**：本版本官方源码原生总超时为 6 小时，CLI 未提供相同的 120 秒选项；自写为 120 秒。本轮最大正式请求延迟 {stats["max_latency_s"]:.3f} 秒，未触及任一上限，未修改官方源码来强行统一。
5. **波动仍需单独看**：自写并发 64 第 2 轮吞吐 2222.19 tok/s、TTFT p99 7362.91 ms，34 次 TTFT 超过 1 秒，同期出现 GPU 短暂空闲；原因未定位。{anomaly_text} 两次测试不是同时进行，不能把波动差异直接归因客户端。
6. **缓存收益未验证**：两轮都是热缓存状态；官方正式阶段 token 加权缓存命中率 {pct:.2f}%（{int(stats["formal_prefix_hit_tokens"])}/{int(stats["formal_prefix_query_tokens"])}），不是请求命中率，也不能据此推导关闭缓存的速度。
7. **显存差异**：官方开始前 GPU1 已用 21169 MiB；此前自写低并发阶段为 21129 MiB、并发 64 达到 21169 MiB。不能把这 40 MiB 的已有状态差异归因官方客户端。
8. **适用范围**：这是共享服务器、短至中等输入、输出上限 128、单卡 TP=1 的交叉检查，没有锁频或交错随机化；不能推广到长输入、多卡、其他模型或冷缓存场景。

## 失败记录与数据核对

- 首次尝试因缺少 pandas，在读取数据集时退出 1；服务端请求/token 计数均无增长。原始失败目录保留为 `20261111T040739_414898Z_official-prefix-on_cee1ccc3`，未覆盖。
- 补装依赖后使用全新目录完整运行，未复用失败产物冒充结果。
- 独立复算确认：6656 个请求与 {audit["total_output_tokens"]} 个实际输出 token 的客户端总数和服务端计数一致；每个阶段也逐项对应。
- 从原始数组重算 TTFT、TPOT、E2E、ITL 的 p50/p95/p99，以及 tok/s 和 req/s；核对全部通过。
- 输入原始顺序与四次循环、输出 cap、错误字段、退出码、服务 PID/命令、无抢占和 GPU 采样错误检查通过。
- [核对结果]({rel}/audit.json)、[每轮 CSV]({rel}/official-rounds.csv)、[预热 CSV]({rel}/official-warmup.csv)、[对照 CSV]({rel}/comparison.csv)、[原始运行事件]({rel}/events.jsonl)。
- 所有正式 JSON 保留逐请求 start_times、TTFT、latency、实际 token 数、ITL、文本与错误字段；本轮没有改动自写客户端。

## 复现方式

从仓库根目录运行，先确认当前服务版本、启动参数和前缀缓存状态与上述一致，且 GPU1 没有其他实验流量。
以下命令复用本次已校验的 512 条派生输入，为新实验创建独立输出目录：

```bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
OUT="project0-baseline/results/$(date -u +%Y%m%dT%H%M%S)_official-recheck"
mkdir "$OUT"
DATA512="project0-baseline/results/{out.name}/prompts-512-in-order.jsonl"
common=(.venv/bin/vllm bench serve
  --model models/Qwen2.5-7B-Instruct --tokenizer models/Qwen2.5-7B-Instruct
  --backend openai-chat --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions
  --dataset-name custom --disable-shuffle --no-oversample --skip-chat-template
  --custom-output-len 128 --temperature 0 --request-rate inf --seed 42
  --num-warmups 0 --ready-check-timeout-sec 0 --disable-tqdm
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99
  --save-result --save-detailed --result-dir "$OUT")
for c in 1 8 32 64; do
  "${{common[@]}}" --max-concurrency "$c" --num-prompts 128 \\
    --dataset-path benchmarks/prompts/baseline-v1.jsonl --result-filename "c$c-warmup.json"
  for r in 1 2 3; do
    "${{common[@]}}" --max-concurrency "$c" --num-prompts 512 \\
      --dataset-path "$DATA512" --result-filename "c$c-round$r.json"
  done
done
```

上述 CLI 参数组合均已在本轮实际执行；监控器记录的完整原始 argv 见 context.json。
独立观察脚本保存在本次目录 runner.py；它含本轮 PID，属于执行记录，换服务后应重新发现 PID。
"""
    path = (
        root / "project0-baseline/experiments/20261111-official-vs-custom-55266e02.md"
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    with (out / "report-writer-source.py").open("x", encoding="utf-8") as stream:
        stream.write(Path(__file__).read_text())
    print(path)


if __name__ == "__main__":
    main()
