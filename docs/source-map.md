# 源码地图

> 路径会随 vLLM / vLLM-Omni 版本移动。**克隆后先用 `find` 校准**，再把实际路径更新到本文件。
>
> 阅读原则：每次只回答 2–3 个具体问题，写在「问题记录」里；不要通读整个仓库。

## vLLM

| 主题 | 入口（示例路径） | 要回答的问题 |
|---|---|---|
| OpenAI 接口 / 请求入口 | `vllm/entrypoints/openai/api_server.py`、`serving_engine.py` | 一个 HTTP 请求在哪里变成引擎请求？ |
| 引擎主循环 | `vllm/v1/engine/llm_engine.py` | `step()` 每轮做了什么？ |
| 调度器（continuous batching） | `vllm/v1/core/sched/scheduler.py` | 请求何时从 waiting 进入 running？抢占如何发生？ |
| KV cache / PagedAttention | `vllm/v1/core/kv_cache_manager.py`、`block_pool.py` | block 如何分配、回收、复用？ |
| 模型执行 / CUDA Graph | `vllm/v1/worker/gpu_model_runner.py` | 一次前向如何组织 input tensors？ |
| Attention 后端 | `vllm/attention/backends/` | 不同后端的选择条件与差异？ |
| 指标 | `vllm/v1/metrics/` | TTFT / TPOT / prefix cache 命中如何统计？ |
| 配置与参数 | `vllm/config.py`（或 `vllm/config/`） | 目标参数影响哪条路径？ |
| 官方压测 | `vllm bench serve`、`vllm bench latency`、`benchmarks/` | 官方指标定义与计算方式？ |

## vLLM-Omni

| 主题 | 入口（示例路径） | 要回答的问题 |
|---|---|---|
| 入口与 API | `vllm_omni/entrypoints/` | 音频/视频请求如何进入？ |
| 多阶段编排 | `vllm_omni/` 下的 engine / stage / pipeline | 阶段如何定义、串联、传数据？ |
| 模型实现 | 模型目录（如 Qwen2.5-Omni） | thinker / talker 如何切分？ |
| 模态编码 | 音频/视觉 encoder 路径 | 输入如何变成 token？ |
| 音频输出 | 音频 decoder / detokenizer 路径 | 音频 token 如何变波形？流式如何实现？ |

## 问题记录

> 每解决一个问题，追加一行。目标是 12 周后能画出完整的请求生命周期图。

| 日期 | 问题 | 结论 | 证据（文件:行 / 实验） |
|---|---|---|---|
| TODO | 例：并发上升后 TTFT 为何突变？ | TODO | TODO |
