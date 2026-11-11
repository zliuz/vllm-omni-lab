# 固定压测输入集

| 文件 | 条数 | 用途 | 输出上限 |
|---|---:|---|---|
| `smoke.jsonl` | 8 | 保留的快速冒烟集，也用于旧命令复现 | 96～192 token |
| `baseline-v1.jsonl` | 128 | 项目 0 默认的多任务混合负载 | 全部为 128 token |

两个文件都使用 UTF-8 JSONL，每行一条独立请求。baseline-v1 额外包含 `id` 和
`category` 供人工检查；客户端发送的是 `prompt`，不会把 ID 拼到文本中。
客户端只读取 `prompt` 与 `output_tokens`，旧格式仍然兼容。

## baseline-v1 的内容与边界

| category | 内容 | 数量 |
|---|---|---:|
| `explanation` | 技术概念、指标与工程权衡解释 | 16 |
| `coding` | Python / SQL 核心实现和输入校验 | 16 |
| `review` | 代码审查、异步清理与实验报告审查 | 16 |
| `extraction` | 工单、日志、库存、发布记录的结构化抽取 | 16 |
| `summarization` | 通知、会议、交接与多段材料摘要 | 16 |
| `reasoning` | 算术、约束、调度与证据一致性检查 | 16 |
| `rewriting` | 中英翻译、语气改写与技术表达 | 16 |
| `grounded_qa` | 仅依据给定材料回答，区分证据与未知 | 16 |

内容为本仓库撰写的合成任务，不是公开数据集的摘抄；材料中的事件、业务记录和
数字均用于练习，不代表本机实验结果。任务包含中文、英文、代码和 JSON 数据，
不通过重复同一句话、替换编号或添加随机前缀凑数量。

8 个类别交错排列，每连续 8 行各含一个类别。默认每轮 128 请求覆盖全文件一次；
`--requests 512` 时每条重复 4 次。正式对照中固定顺序与文件内容，不中途改题。

输出上限统一有助于控制负载，但不是固定实际生成长度：模型可能提前 EOS，也可能
到达上限被截断。实际 token 数必须读取 `usage.completion_tokens`。
本集用于接口性能观察，不以答案完整性或质量得分作为验收指标。

2026-11-10 使用本地 Qwen2.5-7B-Instruct tokenizer 的默认聊天模板，离线测得：

- 输入 token 数：最小 59，中位数 78.5，最大 379。
- 输入加输出上限的最大值：507 token。
- SHA256：`a624abd107e8bb3abf3737d4dc33bab4443c678292c721beca42c02435b9c009`。

这是**短至中等输入的混合负载**，不代表 1k/4k/8k 长上下文或真实用户流量。
token 数只描述该本地 tokenizer 与模板的计算结果，服务端若换模板，以实际 usage
为准；它不是模型吞吐、显存峰值或延迟实验。后续长上下文测试应单独建输入集，
按输入长度分档，避免将不同负载的结果混在同一条优化结论中。

开始正式实验后保持 v1 内容固定；更改任务时新建 v2，保留旧文件和旧结果。
客户端还会把所用文件的 SHA256 写入每次实验的 `metadata.json`。

## prefix caching 的影响与建议

前缀缓存复用已处理的输入前缀对应的 KV，减少 prefill 重算；它不缓存完整答案，
也不直接省去后续每个输出 token 的解码计算。命中时 TTFT、E2E 和总吞吐可能改善，
具体幅度取决于输入长度、重复程度、缓存容量及调度，不能在未实测时给出收益数字。
TPOT/ITL 仍可能受排队与资源竞争的间接影响，不能据此断言开关后必定相同。
原理见 [vLLM 官方说明](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/#limits)。

**项目 0 的第一份基线建议关闭 prefix caching。** 这样可以先观察完整输入处理与
解码的并发曲线。然后将“开启前缀缓存”作为单独的 A/B 实验，其他配置保持一致。
这不表示生产服务应该永久关闭缓存：共享文档、多轮会话和重复问题本来就是合理的
缓存收益场景，应按业务实际情况再测试。

增加到 128 条并不能保证没有命中：不同题可能共享聊天模板前缀，正式轮次还会再次
使用同一文件。尤其开启缓存时，后续轮次不能自然视作冷缓存测试。

本机 vLLM 0.32.0 的 `vllm serve --help=all` 已确认支持以下两个选项：

- `--no-enable-prefix-caching`：关闭。
- `--enable-prefix-caching`：开启。

以下是**下一次启动**时关闭缓存的示例；先正常停止自己的旧服务，并确认残留引擎
和显存占用已经释放，不要在占用同一卡时直接再启动一个实例：

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
MODEL=models/Qwen2.5-7B-Instruct \
EXTRA_ARGS="--no-enable-prefix-caching" \
bash project0-baseline/scripts/serve.sh
```

启动后检查当前运行状态，不仅依赖脚本中的默认值：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8000/metrics | rg '^vllm:cache_config_info'
```

关闭组应显示 `enable_prefix_caching="False"`。开启组换成明确的
`EXTRA_ARGS="--enable-prefix-caching"`，并分别记录完整启动命令和结果目录。
本次仅增加数据和文档，没有重启服务，也没有执行缓存开关 A/B。

若比较热缓存收益，建议两组都用 `--warmup 128`，使每档并发在正式测量前至少
遍历整份输入一次；开启组仍可能发生淘汰，不能直接假定达到 100% 命中。
当前版本 `/metrics` 的 prefix cache queries/hits 单位是 **token 数**，不是请求数。
用正式测量窗口前后的计数器差值计算 `Δhits / Δqueries`；分母为 0 时应标为无定义，
不能拿服务启动以来的累计比值冒充某一轮命中率。

先只校验推荐配置，不发送生成请求：

```bash
python project0-baseline/scripts/bench.py \
  --model models/Qwen2.5-7B-Instruct \
  --prompts benchmarks/prompts/baseline-v1.jsonl \
  --concurrency 1 8 32 64 --requests 512 --warmup 128 --rounds 3 \
  --dry-run
```

正式实验去掉 `--dry-run`，另需记录显存峰值、环境和失败配置。
新负载不能直接与旧 smoke 集数字比较并声称性能提升；缓存 A/B 必须使用同一负载。
