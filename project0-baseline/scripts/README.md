# 项目 0 脚本说明

下面的命令均从仓库根目录执行，先激活现有环境：

```bash
source .venv/bin/activate
```

| 脚本 | 功能 | 输入 | 输出与资源使用 |
|---|---|---|---|
| `smoke_test.py` | 直接调用 `vllm.LLM` 验证离线推理环境 | 可选模型 ID 或本地路径；默认 `models/Qwen2.5-0.5B-Instruct` | 控制台日志和 `OUTPUT: ...`；会加载模型并占用 GPU，无需 HTTP 服务 |
| `serve.sh` | 启动 OpenAI 兼容 HTTP 服务 | 位置参数或 `MODEL`，以及下表中的环境变量 | 前台服务日志；持续占用 GPU，默认端口 8000 |
| `bench.py` | 在线流式压测客户端 | CLI 配置和 JSONL prompt 集 | 生成原始数据、CSV 和实验元数据；客户端本身不加载模型 |

根目录的环境、源码下载和模型下载脚本见 [../../scripts/README.md](../../scripts/README.md)。

## `smoke_test.py`：离线冒烟测试

先执行 `nvidia-smi` 确认空闲 GPU，再运行：

```bash
CUDA_VISIBLE_DEVICES=1 python project0-baseline/scripts/smoke_test.py
```

也可在脚本后传入模型路径。当前固定配置为上下文 1024、显存比例 0.35、
temperature=0、最多生成 32 token。它使用多进程入口保护，必须作为文件运行。
这不是在线压测，也不测并发。脚本内部用法中的 GPU0 只是示例，本机优先用 GPU1。

## `serve.sh`：启动在线服务

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
MODEL=models/Qwen2.5-7B-Instruct \
bash project0-baseline/scripts/serve.sh
```

| 输入 | 默认值 | 含义 |
|---|---|---|
| 第一个位置参数 / `MODEL` | `models/Qwen2.5-7B-Instruct` | 模型路径或 ID，位置参数优先 |
| `CUDA_VISIBLE_DEVICES` | 未设置 | 由 CUDA 读取，用于选择物理 GPU；脚本仅打印当前值 |
| `PORT` | `8000` | HTTP 端口 |
| `TP` | `1` | 张量并行数 |
| `MAX_MODEL_LEN` | `8192` | 最大上下文长度 |
| `GPU_MEM_UTIL` | `0.90` | 本实例的整卡显存预算比例，不是剩余显存比例 |
| `MAX_NUM_SEQS` | `64` | 服务端最大序列数，与客户端并发分别设置 |
| `EXTRA_ARGS` | 空 | 按空白拆分的额外 CLI 参数，不支持带空格的复杂引用 |

本地模型路径相对当前工作目录解析。传模型 ID 时，vLLM 可能联网查缓存或下载；
下载脚本放在 `models/` 的文件不会仅凭同名 ID 自动被使用。
`HF_HUB_OFFLINE=1` 约束 Hugging Face 文件访问，不保证其他组件都不联网。

服务就绪后，在另一个终端检查 API 模型名：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8000/v1/models
```

压测的 `--model` 必须与返回的 `data[].id` 一致。本地路径启动时，通常也是该路径；
设置了 `--served-model-name` 时以 API 返回值为准。

## `bench.py`：学习框架与实现顺序

核心练习按 **TODO 1～5** 组织，每个函数都有参数、返回值、异常或实现提示。
目前 5 个核心练习均已有实现，并包含中文注释和模拟数据测试。
已完成开启前缀缓存的真实压测、显存采样及官方工具交叉验证，结果见 [项目总结](../README.md)。
关闭缓存组尚未运行；吞吐同量级不代表全部延迟分位数完全对齐。
`--help` 和 `--dry-run` 已可用，dry-run 不联网、不写结果、不调用 TODO。

```bash
python project0-baseline/scripts/bench.py --help
python project0-baseline/scripts/bench.py \
  --model models/Qwen2.5-7B-Instruct --dry-run
```

| 顺序 | 函数 | 核心逻辑 | 建议验证 |
|---|---|---|---|
| TODO 1 | `send_request` | HTTP 流式请求、SSE 解析、文本到达时间、token usage、失败记录 | 单请求；角色事件、空 choices、usage、结束事件、断流、HTTP 错误、超时 |
| TODO 2 | `run_round` | 并发槽位、任务调度、整轮计时和取消清理 | 记录实际在途请求数，确保不超过并发；单个请求失败不取消其他请求 |
| TODO 3（已实现） | `request_metrics` | 单请求 TTFT、TPOT、E2E、输出速度及 chunk 间隔 | 已用人工时点验证公式、0/1 token、缺失/无效时间戳及失败请求 |
| TODO 4（已实现） | `summarize_round` | 每轮分位数、吞吐、失败率 | 已覆盖全成功、部分失败、全失败、缺 usage；输出字段匹配 `SUMMARY_FIELDS` |
| TODO 5（已实现） | `summarize_repeats` | 每个并发档位的多轮中位数、最小值和最大值 | 至少三轮、并发分组、重复轮次、缺失值与 CSV 保存 |

调用关系：

```text
main → parse_args / load_prompts
     → dry-run：只打印配置
     → run_benchmark：创建结果目录，检查 /v1/models，复用一个 HTTP client
         → 每档并发：run_round(预热) → run_round(正式，至少 3 轮)
             → send_request（受并发槽位约束）
         → 保存每轮原始记录 → summarize_round → request_metrics
         → summarize_repeats → 保存多轮统计
```

### 负载与参数

JSONL 每行形如：

```json
{"prompt": "解释 continuous batching。", "output_tokens": 128}
```

`output_tokens` 是请求的生成上限，模型提前 EOS 时实际长度可能更短。
框架固定顺序循环使用 prompt；不要在 A/B 实验中途更换数据。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--model` | 必填 | API 中提供的模型名，不会被客户端下载 |
| `--base-url` | `http://127.0.0.1:8000` | 不带 `/v1` 的服务根地址 |
| `--prompts` | `benchmarks/prompts/baseline-v1.jsonl` | 128 条、8 类任务；默认路径按仓库位置定位，显式相对路径按当前目录解析 |
| `--output-dir` | `project0-baseline/results` | 每次实验创建唯一子目录，不覆盖历史 |
| `--concurrency` | `1 8 32 64` | 要测试的并发档位 |
| `--requests` | `128` | 每个正式轮次的总请求数，至少等于最大并发 |
| `--warmup` | `10` | 每档并发先预热的总请求数，至少 10；不进入正式统计 |
| `--rounds` | `3` | 每档并发的正式轮数，至少 3 |
| `--timeout` | `120` | 整个单请求的超时秒数，核心实现需使用 `asyncio.timeout` |
| `--temperature` | `0.0` | 所有请求共享的采样参数 |
| `--dry-run` | 关闭 | 只校验配置与负载 |

框架预热只保证请求数不少于 10；若要让预热覆盖全部并发槽位，可设 `--warmup 64`。
若比较 128 条新输入的热缓存收益，推荐两组都设 `--warmup 128`，先遍历完整输入集。
首份基线建议关闭前缀缓存；数据说明和已核对的服务启动参数见
[固定输入集与缓存策略](../../benchmarks/prompts/README.md)。旧冒烟输入仍可通过
`--prompts benchmarks/prompts/smoke.jsonl` 显式选择。此次仅改变默认输入集，
未自动修改并发、请求数、预热或服务端配置。
客户端固定 `trust_env=False`，避免环境代理改变本地压测路径。
鉴权仅通过 `VLLM_API_KEY` 环境变量传入，不保存到元数据；错误信息也不要包含认证头。

### 指标口径

| 指标 | 本框架的约定 |
|---|---|
| TTFT | 获得并发槽位并准备发送请求，到首个非空文本 delta；毫秒 |
| E2E | 从发送前到完整流结束，包含最终 usage / 结束事件开销；毫秒 |
| TPOT | `(E2E - TTFT) / (实际输出 token 数 - 1)`；不足 2 token 时留空 |
| 每请求输出 tok/s | 实际输出 token 数 / 本请求 E2E 秒数 |
| 整轮输出 tok/s | 成功请求实际输出 token 总数 / 整轮墙钟秒数 |
| 请求吞吐 | 成功请求数 / 整轮墙钟秒数 |
| 失败率 | 失败请求数 / 本轮总请求数 |
| `chunk_itl_ms` | 相邻非空文本 delta 到达间隔的均值；明确标记为 chunk 观测 |

SSE 事件、网络分块和 token 不是一一对应关系。**不能将 chunk 数当 token 数，
也不能把 chunk 间隔冒充精确的逐 token ITL。** 使用服务端最终
`usage.completion_tokens` 作为 token 计数；缺失时记录原因，不能用 `max_tokens` 代替。
只对成功请求的有效指标计算分位数；None 输出为空单元格，失败数据仍保留。
分位数固定使用线性插值：排序后在下标 `(n-1)*p` 处取值，`p` 为 0.50/0.95/0.99。
一个有效样本的三个分位数相同，没有有效样本则留空；不将缺失值补成 0。
失败率是 0～1 的比例。成功请求中任意一条缺少有效 token usage，整轮 token 总数和
输出 tok/s 留空，请求吞吐仍可计算。全失败但耗时有效时，成功吞吐为 0；
空轮次或无效耗时不产生吞吐数值。

这是固定在途并发、请求完成后补下一个请求的闭环负载，不是固定到达率测试。
客户端等待并发槽位的时间不计入 TTFT，但服务端排队和网络等待计入。
短输出时，结束事件的额外开销可能影响 TPOT；与官方工具对比时检查计时口径。
多轮统计是“每轮 p99 的中位数”等，不能标作所有请求合并后的 p99。
每档并发至少需要 3 个不同的正式轮次，拒绝重复 round 编号和预热编号 0。
各轮等权统计，`min`/`max` 是各轮有效值的观测范围。每项指标独立忽略缺失或
无效值，有效 0 保留；全缺失时 median/min/max 留空。有效样本可能少于轮数，
可查 `summary.csv` 确认每项指标实际有几轮数据。

### 运行方式与产物

以下为正式实验命令；当前客户端尚未运行真实基线压测：

```bash
python project0-baseline/scripts/bench.py \
  --model models/Qwen2.5-7B-Instruct \
  --concurrency 1 8 32 64 --requests 512 --warmup 128 --rounds 3
```

每次在结果目录中创建独立的 UTC 时间戳子目录：

| 文件 | 内容 |
|---|---|
| `metadata.json` | CLI 配置、prompt SHA256、客户端 Python/httpx 版本、计时口径 |
| `events.jsonl` | 开始、每轮耗时、成功结束或异常中断；仅追加 |
| `requests.jsonl` | 每轮原始观测、输出文本和失败原因；预热有独立 phase 标记 |
| `summary.csv` | 每个正式轮次一行，含 p50/p95/p99、吞吐和失败率 |
| `repeats.csv` | 每档并发各项指标的多轮 median/min/max |

中断时保留已写入的数据；检查 `events.jsonl` 是否有 `completed`，不能将中断实验
当作完整基线。metadata 记录的是客户端环境，不代表服务端环境。

GPU 显存峰值采样、绘图和官方交叉验证尚未接入框架，仍需：

1. 在实验期间采样**物理 GPU1** 的显存，用独立结果文件记录采样间隔和峰值；不采样就写“未验证”。
2. 按 [实验模板](../../docs/experiments/TEMPLATE.md) 记录服务端版本/commit、驱动/CUDA、
   完整启动命令、max-model-len、采样参数、prefix caching / chunked prefill 状态及失败配置。
3. 用现有 matplotlib 生成并发与吞吐/TTFT 曲线，标明模型与环境。
4. 先看当前安装版本的 `vllm bench serve --help`，统一 prompt、token 计数、预热、
   并发与计时口径后交叉验证；差异超过项目要求的 ±20% 时排查原因。
   重复小 prompt 集可能受 prefix cache 影响，需要固定并记录缓存策略。

### 本地框架检查

```bash
python -m pytest project0-baseline/tests -q
ruff check project0-baseline/scripts/bench.py project0-baseline/tests
```

现有测试验证输入校验、dry-run、流程编排、指标公式、统计及结果保存，使用模拟数据，
不会连接 GPU 服务，也不代替真实服务的端到端验证。
