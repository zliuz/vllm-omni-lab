# benchmarks

共享压测工具、prompt 集与结果约定。项目 0 起逐步补齐。

## 目录

```text
benchmarks/
├── prompts/
│   ├── smoke.jsonl          # 8 条冒烟 prompt，保留旧输入
│   ├── baseline-v1.jsonl    # 128 条默认混合任务，输出上限统一
│   └── README.md            # 内容、长度统计与前缀缓存实验约定
├── results/             # 小体积结果（CSV/JSON/PNG），大日志不入库
└── README.md
```

## prompt 集

两个输入集均每行一个 JSON，客户端要求以下字段：

```json
{"prompt": "...", "output_tokens": 128}
```

`smoke.jsonl` 用于快速冒烟；客户端默认使用 `baseline-v1.jsonl`，覆盖 8 类任务，
每类 16 条。正式实验应固定同一份 prompt 集，不要中途更换。
数据设计、实际 token 长度和缓存开关建议见 [prompts/README.md](prompts/README.md)。

> `vllm bench serve` 的自定义数据集参数在不同版本中可能变化（`--dataset-name custom`、`--dataset-path` 等）。使用前以 `vllm bench serve --help` 为准。

## 工具

| 工具 | 用途 | 备注 |
|---|---|---|
| `vllm bench serve` | 官方在线压测（TTFT/TPOT/吞吐） | 优先作为交叉验证 |
| `vllm bench latency` / `throughput` | 离线延迟/吞吐 | 测单请求极限 |
| `project0-baseline/scripts/bench.py` | 自写压测客户端 | 项目 0 产出，理解指标定义 |

## 结果约定

- 命名：`<模型简称>-<并发>-<关键配置>-<YYYYMMDD>.csv`
- 每次结果旁放一份 `.md`，写清命令与环境（或直接链接到实验记录）
- 图中必须标注：并发、TP/DP、版本、是否开启 prefix caching / chunked prefill
- 报告 p50/p95/p99，不只报平均值

## 基准纪律

1. 同一张表内，除目标变量外所有配置一致
2. warmup ≥ 10 次，正式 ≥ 3 轮
3. 记录显存峰值与是否发生 OOM
4. 不把不同 prompt 集、不同 `max-model-len` 的数字直接对比
