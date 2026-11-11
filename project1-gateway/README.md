# 项目 1：推理网关 + 调度实验（约 2 周）

## 目标

写一个可用的 LLM 推理网关，并用它完成 **4090 专属的部署实验**——尤其是「TP=2 单实例 vs 2×TP=1 数据并行」这个必须用数据回答的问题。

## 架构

```text
客户端 (OpenAI 协议)
        │
        ▼
┌─────────────────────────┐
│  gateway :9000          │
│  - SSE 流式透传          │
│  - 路由 / 重试 / 超时     │
│  - token 计量            │
│  - Prometheus 指标       │
└───────────┬─────────────┘
            │
   ┌────────┴─────────┐
   ▼                  ▼
vLLM :8000        vLLM :8001
(GPU0, TP=1)      (GPU1, TP=1)
```

同一套网关也要能配置成「一个 TP=2 实例」作为对照组。

## 功能清单

**MVP（必须先完成）**

- [ ] `POST /v1/chat/completions` 透传，支持 `stream=true`
- [ ] 两个上游：轮询 / 最少在途请求
- [ ] 连接超时、读超时、失败重试
- [ ] 每请求 token 用量统计（从 usage 字段与流式末尾 chunk 提取）
- [ ] `/metrics`：请求数、错误数、在途数、TTFT 直方图

**进阶（时间允许再做）**

- [ ] 按 API key / 租户限流
- [ ] 上游健康检查与摘除、恢复
- [ ] 请求排队上限与快速失败
- [ ] Docker Compose 一键启动（网关 + 2 后端）
- [ ] 简单的 Web 压测面板

> 本机当前**未安装 Docker / nvidia-container-toolkit**。先用 `venv + tmux`（或 systemd）跑通网关与两个后端；Compose 作为后续可选增强。Docker 的作用、安装步骤与 Compose 示例见 [docs/docker.md](../docs/docker.md)。

## 核心实验（本项目的重点）

### 实验 A：TP=2 单实例 vs 2×TP=1（DP）

固定总并发与总显存占用，比较：

| 部署方式 | 总吞吐 | TTFT p50/p99 | TPOT p50/p99 | 显存/卡 |
|---|---:|---:|---:|---:|
| 1 实例 TP=2 | | | | |
| 2 实例各 TP=1 | | | | |

预期 DP 更优（4090 无 NVLink），但必须用数据证明，并解释通信开销体现在哪里。

### 实验 B：调度参数

- `--enable-prefix-caching` 开/关（使用重复系统提示词的负载）
- `--enable-chunked-prefill` 开/关
- `--max-num-batched-tokens` 扫描
- `--max-num-seqs` 扫描

实验 C（可选）：LoRA 多适配器服务。

## 目录

```text
project1-gateway/
├── app/
│   ├── main.py          # FastAPI 入口
│   ├── config.py        # 配置加载
│   ├── router.py        # 路由策略
│   ├── upstream.py      # 上游客户端 / 流式透传
│   ├── metrics.py       # Prometheus
│   └── usage.py         # token 计量
├── deploy/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   └── prometheus.yml
├── experiments/         # 实验记录
├── tests/
└── requirements.txt
```

配置参考 [`configs/gateway/gateway.yaml.example`](../configs/gateway/gateway.yaml.example)。

## 交付物

- 可一键启动的网关仓库（Compose）
- 实验 A/B 的报告：表 + 图 + 结论
- 网关单测（路由、重试、usage 解析）

## 验收标准

- [ ] 流式响应逐 token 透传，不被缓冲
- [ ] 杀掉一个后端后，请求能重试到另一个后端并最终成功
- [ ] `/metrics` 可被 Prometheus 抓取
- [ ] TP=2 vs DP 的结论可复现，并给出机制解释
- [ ] 报告写清所有版本与命令

## 提示

- 流式透传不要用 `httpx.post` 一次性读完，用 `client.stream(...)`
- 不要在网关里累积完整响应再转发，会吃掉流式收益
- 重试只对幂等失败（连接失败、5xx）生效，流式响应已开始时不能重试
