# 项目 3：小型二次开发 / 性能实验（约 2 周）

## 目标

从「改脚本」过渡到「改源码」，完成第一个有测试、有数据支撑的改动。这是代码能力的主训练场。

## 三选一（建议 A 或 B）

### 选项 A：可观测性小特性（推荐）

给 vLLM 增加一个 metrics 字段，例如：

- 每请求排队时间（queue time）
- 前缀缓存命中的 token 数
- 每请求的 prefill/decode 时间拆分

要求：

- 改动最小、边界清晰，能向上游提 PR
- 附单测
- 用项目 1 的压测证明指标正确且开销可忽略

### 选项 B：前缀缓存亲和路由（偏应用）

在项目 1 的网关中，根据请求前缀 hash 做亲和路由：

- 相同系统提示词/前缀的请求尽量落到同一后端
- 量化：prefix cache 命中率、TTFT 改善、后端负载均衡代价
- 需要理解 vLLM 的 prefix caching 与 block hash 机制

### 选项 C：ngram 投机解码实验（纯配置）

- 在 4090 上开启 ngram speculative decoding
- 严谨 A/B：不同输入重复度（代码补全 / 摘要 / 闲聊）
- 回答：消费级 GPU 上是否值得开？收益边界在哪？

## 交付物

| 选项 | 交付物 |
|---|---|
| A | patch/PR + 单测 + 指标正确性验证 + 开销数据 |
| B | 网关代码 + 命中率/TTFT 对比 + 负载均衡分析 |
| C | 实验报告（含拒绝结论也可以）+ 可复现配置 |

## 验收标准

- [ ] 单变量 A/B，有基线对照
- [ ] 代码改动有测试；lint 通过
- [ ] 能指出改动位于请求生命周期的哪一步
- [ ] 失败/OOM/无收益的配置也如实记录
- [ ] 若做 A，patch 可干净地应用到固定 commit 的上游

## 目录约定

```text
project3-perf/
├── option-a-metrics/
├── option-b-prefix-routing/
├── option-c-spec-decode/
└── experiments/
```

- patch 文件命名：`<上游commit前8位>-<短描述>.patch`
- 实验记录复制 `docs/experiments/TEMPLATE.md`
- 不直接修改 `external/`；在副本或 fork 中改，再导出 patch

## 选题建议

优先选「小、可验证、能讲清楚」的改动。第一个 PR 不碰调度器核心与 CUDA kernel；把 `metrics`、`usage`、`日志`、`测试` 作为入口，理解请求生命周期后再深入。
