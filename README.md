# vllm-omni-lab

vLLM / vLLM-Omni 项目驱动学习仓库。

以 5 个递进项目学习 vLLM 与 vLLM-Omni：**部署基线 → 推理网关 → 实时语音助手 → 二次开发 → 上游贡献**。每个阶段都有可运行、可量化、可展示的交付物。

## 学习状态

| 项目 | 内容 | 周期 | 状态 |
|---|---|---|---|
| [0](project0-baseline/) | 环境 + 基线压测 | 1 周 | ⬜ 未开始 |
| [1](project1-gateway/) | 推理网关 + 调度实验 | 2 周 | ⬜ 未开始 |
| [2](project2-omni-voice/) | vLLM-Omni 实时语音助手 | 3 周 | ⬜ 未开始 |
| [3](project3-perf/) | 小特性 / 性能实验 | 2 周 | ⬜ 未开始 |
| [4](project4-upstream/) | 上游贡献 + 总结 | 2 周 | ⬜ 未开始 |

图例：⬜ 未开始 · 🟨 进行中 · ✅ 已验收 · ⏸ 暂停

## 硬件与约束

本机配置：

- **2 × RTX 4090 24GB**（共 48GB，无 NVLink / P2P，多卡走 PCIe）
- 256GB 内存 · 192 核 CPU · 1.8T 磁盘

由此得到的三条硬结论：

1. **模型选择**：文本用 4B/8B 级；Omni 用 Qwen2.5-Omni-3B（单卡）/ 7B（双卡）。32B 以上默认不考虑。
2. **并行策略**：7B 级模型优先 **2 个 TP=1 实例（数据并行）**，不要默认 TP=2 更快。项目 1 会用数据验证这一点。
3. **显存**：单卡可用约 21.6GB（`--gpu-memory-utilization 0.90`）。OOM 处理顺序：降 `--max-model-len` → 换更小模型 → FP8/AWQ 量化（4090 支持 FP8）。

## 学习循环

每个项目都走同一条循环，不允许跳步：

```
跑通官方 example → 改参数做对照实验 → 带着具体问题读源码
→ 写实验报告（图表 + 结论）→ 做一个最小代码改动 → 提 PR / 沉淀文档
```

原则：

- **小模型优先**：先用 3B/4B 跑通全链路，再换大模型
- **先 API 后源码**：能用 `vllm serve` 验证的行为，先不做源码级假设
- **每个实验必须有基线**：固定 prompt 集、warmup、多轮重复
- **源码按需读**：每次只回答 2–3 个具体问题，记录在 [docs/source-map.md](docs/source-map.md)

## 目录结构

```text
vLLM-base/
├── README.md                  # 本文件：总览、状态、快速开始
├── AGENTS.md                  # AI 代理与贡献者的工作规范（唯一规范来源）
├── CLAUDE.md                  # Claude Code 入口，引用 AGENTS.md
├── requirements-dev.txt       # 本地工具链（不含 vllm 本身）
├── Makefile                   # 常用命令入口
├── docs/
│   ├── environment.md         # 本机环境清单（运行 check_env 后填写）
│   ├── source-map.md          # vLLM / vLLM-Omni 源码地图
│   └── experiments/
│       └── TEMPLATE.md        # 实验记录模板
├── benchmarks/                # 共享压测工具与 prompt 集
│   ├── prompts/smoke.jsonl
│   └── results/
├── configs/                   # 可复用启动配置
│   ├── vllm/
│   └── gateway/
├── external/                  # 上游源码克隆位置（只读，已 gitignore）
├── scripts/                   # 环境检查 / 克隆 / 模型下载
├── project0-baseline/         # 项目 0：环境 + 基线压测
├── project1-gateway/          # 项目 1：推理网关 + 调度实验
├── project2-omni-voice/       # 项目 2：Omni 实时语音助手
├── project3-perf/             # 项目 3：小特性 / 性能实验
└── project4-upstream/         # 项目 4：上游贡献与总结
```

## 快速开始

```bash
# 0. 检查本机环境（只读，不修改系统）
bash scripts/check_env.sh
# 将输出填入 docs/environment.md

# 1. 创建 Python 虚拟环境并安装开发依赖
bash scripts/setup_env.sh

# 2. 克隆上游源码到 external/（只读参考，不提交）
bash scripts/clone_upstream.sh

# 3. 下载基线模型（默认 Qwen2.5-7B-Instruct；可传多个 model id / 本地路径）
bash scripts/download_models.sh Qwen/Qwen2.5-7B-Instruct

# 4. 启动第一个服务（详见 project0-baseline/README.md）
bash project0-baseline/scripts/serve.sh
```

也可以使用 `make check-env` / `make setup` / `make clone` / `make download`。

> 模型仓库名与 vLLM-Omni 支持的模型会随版本变化，**以仓库 README 的 Supported Models 和实际下载结果为准**。下载脚本中的默认值仅作示例。

## 实验与记录规范

所有性能数字必须可复现，记录到 `docs/experiments/` 或各项目的 `experiments/`：

- 固定 prompt 集与采样参数；A/B 实验一次只改一个变量
- 每档配置 warmup ≥ 10 次，正式 ≥ 3 轮，报告中位数与 p95/p99
- 必须记录：vLLM / vLLM-Omni 版本或 commit、驱动 / CUDA、完整启动命令、模型 id、并发数、`max-model-len`
- 未实际运行的结果写「未验证」，不得估算后当作实测

模板见 [docs/experiments/TEMPLATE.md](docs/experiments/TEMPLATE.md)。

统一指标：`TTFT`、`TPOT`（或 ITL）、输出 `tok/s`、请求吞吐、显存峰值、p50/p95/p99。

## 12 周节奏（按每周约 10 小时，可伸缩）

| 周 | 内容 | 里程碑 |
|---|---|---|
| 1 | 环境、项目 0 | 基线压测表 |
| 2–3 | 项目 1（网关 + 实验） | 一键启动 + 实验报告 |
| 4 | 缓冲 / 补源码（scheduler、KV cache） | 能画请求生命周期图 |
| 5–7 | 项目 2（Omni 语音助手） | 实时 demo + 延迟瀑布图 |
| 8 | 缓冲 / 量化与显存实验 | 3B vs 7B 对比 |
| 9–10 | 项目 3（小特性 / 路由 / 投机解码） | 代码 + 测试 + 数据 |
| 11–12 | 项目 4（PR + 文档） | 提交 PR、系列文章 |

每周只能投入约 5 小时时，按 20–24 周拉长，项目顺序不变。

## 文档索引

- [AGENTS.md](AGENTS.md)：代理/贡献者规范、常用命令、Definition of Done、禁止事项
- [docs/environment.md](docs/environment.md)：本机驱动、CUDA、Python、版本锁定
- [docs/source-map.md](docs/source-map.md)：vLLM / vLLM-Omni 源码入口
- [docs/experiments/TEMPLATE.md](docs/experiments/TEMPLATE.md)：实验记录模板
- [benchmarks/README.md](benchmarks/README.md)：压测工具与 prompt 集说明

## 注意事项

- 上游模型权重、`external/` 源码、虚拟环境均不进入版本库
- `external/` 只读；需要改 vLLM / vLLM-Omni 时，在项目目录内做 patch 或 fork，并记录基线 commit
- 本仓库中的路径基于 vLLM 一个较新的版本；克隆后先用 `find` 校准 [docs/source-map.md](docs/source-map.md) 中的路径
