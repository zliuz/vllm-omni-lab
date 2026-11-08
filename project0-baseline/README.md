# 项目 0：环境 + 基线压测（约 1 周）

## 目标

在本机跑通第一个 vLLM 服务，并建立**可信、可复现**的性能基线。后续所有优化都以这里的数据为对照。

## 前置

1. `bash scripts/check_env.sh`，把结果填入 [`docs/environment.md`](../docs/environment.md)
2. `bash scripts/setup_env.sh` 创建 `.venv`
3. `bash scripts/clone_upstream.sh` 克隆上游源码
4. 固定 vLLM 版本并记录（release tag）

## 步骤

### 1. 安装 vLLM

优先安装与 CUDA 匹配的 wheel：

```bash
source .venv/bin/activate
uv pip install vllm==<固定版本>
```

wheel 不兼容时再考虑源码编译。无论哪种方式，把版本和安装命令写进 `docs/environment.md`。

### 2. 下载基线模型

```bash
bash scripts/download_models.sh Qwen/Qwen2.5-7B-Instruct
```

7B bf16 约 15.5GB，单卡 24GB 可运行（上下文需控制在 8k 左右）。

### 3. 启动服务

```bash
bash project0-baseline/scripts/serve.sh
# 或显式指定
MODEL=Qwen/Qwen2.5-7B-Instruct PORT=8000 GPU_MEM_UTIL=0.90 bash project0-baseline/scripts/serve.sh
```

验证：

```bash
curl -s http://127.0.0.1:8000/v1/models | head
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"你好"}]}'
```

### 4. 写自己的压测客户端

在 `scripts/bench.py` 实现（不要直接抄官方脚本）：

- asyncio + httpx 并发发送流式请求
- 统计 TTFT、TPOT/ITL、每请求输出 tok/s、总吞吐
- 输出 p50/p95/p99 到 CSV
- 记录失败率

### 5. 与官方工具交叉验证

```bash
vllm bench serve --help   # 先确认当前版本的参数名
vllm bench serve --model <model> --backend openai-chat --base-url http://127.0.0.1:8000 ...
```

自写脚本与官方结果的量级应一致（±20% 内）；差异大时先检查 warmup、tokenizer 计数、是否流式。

### 6. 跑基线实验

并发档位：1 / 8 / 32 / 64。每档 warmup ≥ 10 次，正式 ≥ 3 轮。

| 并发 | 输出 tok/s | 请求吞吐 | TTFT p50 | TTFT p99 | TPOT p50 | 显存峰值 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | |
| 8 | | | | | | |
| 32 | | | | | | |
| 64 | | | | | | |

### 7. 解释曲线

写一段结论，回答：

- 吞吐随并发如何变化？在哪个点趋于饱和？
- TTFT 与 TPOT 分别在什么并发下开始恶化？
- 这与 continuous batching / 显存 / KV cache 的关系是什么？

## 交付物

- `scripts/bench.py`（带用法说明）
- `results/` 下的 CSV + 图
- `experiments/YYYYMMDD-baseline.md`（模板见 `docs/experiments/TEMPLATE.md`）
- `docs/environment.md` 填写完整

## 验收标准

- [ ] 服务稳定运行，OpenAI 兼容接口可调用
- [ ] 自写压测与官方 `vllm bench serve` 同量级
- [ ] 有完整的并发 × 吞吐/TTFT 表和图
- [ ] 能用自己的话解释 TTFT、TPOT、continuous batching

## 常见问题

| 现象 | 优先检查 |
|---|---|
| CUDA OOM | 降 `--max-model-len`、降 `--max-num-seqs`、确认没有残留进程占卡 |
| 启动卡在编译 / 加载 | 首次启动会做 CUDA graph capture，属正常；看日志确认进度 |
| 下载失败 | `export HF_ENDPOINT=https://hf-mirror.com` 或改用 ModelScope |
| 吞吐异常低 | 检查是否每次请求都重新建连接、是否真的流式、是否被 CPU 侧 tokenizer 限制 |
