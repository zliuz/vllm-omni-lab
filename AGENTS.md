# AGENTS.md — 本仓库的工作规范

本文件是 AI 编码代理与人类贡献者在本仓库工作的**唯一规范来源**。开始任何任务前先读完本文件；与临时直觉冲突时，以本文件为准。

## 1. 项目定位

这是一个**学习型仓库**：通过 5 个递进项目掌握 vLLM 与 vLLM-Omni，目标包括部署应用、性能优化/二次开发、上游开源贡献。仓库同时保存学习者的脚本、实验数据与文档。

学习者的基础：Python / PyTorch 可用，读过推理框架论文，但工程代码能力较弱。因此：

- 优先给出**最小可运行**的实现，再逐步重构
- 解释关键改动「为什么」，而不只是「是什么」
- 不要一次引入过多抽象或框架

## 2. 硬约束（不得假设）

1. **硬件**：2 × RTX 4090 24GB，共 48GB，**无 NVLink/P2P**，多卡通信走 PCIe；256GB RAM。
   - 不要默认 TP=2 比 2×TP=1（DP）快；涉及多卡时必须实测对比
   - 7B 级 bf16 模型优先单卡或 DP
2. **显存**：单卡可用约 21.6GB。OOM 处理顺序：
   1. 降低 `--max-model-len`
   2. 降低并发 / `--max-num-seqs`
   3. 换更小模型（3B/4B）
   4. FP8/AWQ 量化（4090 为 sm89，支持 FP8）
3. **版本锁定**：vLLM 固定 release tag；vLLM-Omni 固定 commit，并以它 `pyproject.toml` 中锁定的 vLLM 版本为准。升级版本必须记录原因与前后结果。
4. **模型来源**：权重不进入仓库；国内网络优先 HF 镜像（`HF_ENDPOINT=https://hf-mirror.com`）或 ModelScope。
5. **不确定性**：模型支持列表、CLI 参数、源码路径随版本变化。执行前以 `external/` 中实际代码与 README 为准，不要凭记忆断言。

## 3. 目录职责

| 路径 | 职责 | 规则 |
|---|---|---|
| `external/vllm/`、`external/vllm-omni/` | 上游源码 | **只读**；不修改、不提交、不格式化 |
| `project0-baseline/` | 项目 0：环境与基线 | 脚本 + 结果 |
| `project1-gateway/` | 项目 1：推理网关 | 网关代码、部署、实验 |
| `project2-omni-voice/` | 项目 2：Omni 语音助手 | 客户端、服务脚本、实验 |
| `project3-perf/` | 项目 3：二次开发 | 每个选项独立子目录 |
| `project4-upstream/` | 项目 4：上游贡献 | PR 笔记与 patch |
| `benchmarks/` | 共享压测工具与 prompt 集 | 不放一次性脚本 |
| `configs/` | 可复用启动配置 | 不含密钥；本地覆盖用 `*.local.*` |
| `docs/` | 环境、源码地图、实验记录 | 结论必须能追溯到实验 |
| `scripts/` | 环境 / 克隆 / 下载 | 必须可重复执行、失败即退出 |
| `*/results/`、`*/experiments/` | 实验数据 | **只追加**，不覆盖历史 |

## 4. 常用命令

```bash
bash scripts/check_env.sh          # 只读检查环境，写入 docs/environment.md
bash scripts/setup_env.sh          # 创建 .venv 并安装开发依赖
bash scripts/clone_upstream.sh     # 克隆 vLLM / vLLM-Omni 到 external/
bash scripts/download_models.sh <model-id> [...]

# 也可以
make check-env / make setup / make clone / make download
```

服务与压测命令随项目 README 走；不要在此处臆造不存在的脚本。

## 5. 编码规范

- Python 3.11/3.12；函数与公共接口加类型标注
- 使用 `ruff` 检查（配置在 `requirements-dev.txt` 之后可补充）
- 每个新功能附 `pytest`；测试放在对应项目的 `tests/`
- 新依赖先说明理由，并写入对应项目的 `requirements.txt` 或根 `requirements-dev.txt`
- 脚本统一 `#!/usr/bin/env bash` + `set -euo pipefail`
- 不写死绝对路径；用脚本自身的 `ROOT=$(cd "$(dirname "$0")/.." && pwd)`
- 不在仓库内保存密钥、token、Cookie

## 6. 实验规范（重要）

任何性能结论都必须可复现：

1. **单变量**：A/B 实验一次只改一个配置
2. **基线**：每档配置先 warmup ≥ 10 次，正式 ≥ 3 轮，报告多轮中位数与波动
3. **记录元数据**：vLLM / vLLM-Omni 版本或 commit、GPU 驱动 / CUDA、完整启动命令、模型 id、prompt 集、并发、`max-model-len`、采样参数
4. **指标**：TTFT、TPOT/ITL、输出 tok/s、请求吞吐、显存峰值、p50/p95/p99
5. **真实性**：未运行的结果写「未验证」；禁止估算、推测或复用旧数据冒充新实验
6. **失败也记录**：OOM、崩溃、超时的配置要写进报告，不能只保留成功案例

实验模板：`docs/experiments/TEMPLATE.md`。

## 7. Definition of Done（每个项目结束前检查）

- [ ] 代码可从干净环境按 README 一条命令跑通
- [ ] 关键路径有测试或可复现的验证命令
- [ ] 实验报告包含基线、结果表/图、结论、限制与未完成项
- [ ] 更新根 `README.md` 的状态表
- [ ] 更新 `docs/environment.md` 中受影响的版本信息
- [ ] 没有把权重、日志、`.venv`、`external/` 提交进仓库
- [ ] 所有命令与路径在仓库当前版本下实际验证过

## 8. 上游贡献流程（项目 4）

1. 选 `good first issue`、文档错误、测试补齐或小 bug；**第一个 PR 不碰调度器与 CUDA kernel**
2. 在 `external/` 中复现，记录复现步骤与基线
3. 最小修复 + 单测；本地跑通 lint / 相关测试
4. PR 描述包含：问题、复现、改动、验证命令、影响面
5. 回应 review 时优先补测试与文档，不做无关重构
6. 在 `project4-upstream/notes/` 记录 PR 链接、状态、经验

## 9. 禁止事项

- ❌ 修改或提交 `external/` 中的上游源码
- ❌ 删除 / 覆盖 `results/`、`experiments/` 中的历史数据
- ❌ 提交模型权重、数据集、大日志、虚拟环境
- ❌ 执行破坏性 git 操作（`reset --hard`、`clean -fd`、`push --force`）或删除用户文件，除非用户明确要求
- ❌ 在未实测时声称性能结论或「已修复」
- ❌ 擅自升级 vLLM / vLLM-Omni 版本而不记录

## 10. 交付与沟通

- 每个任务结束时说明：改了什么、怎么验证、还剩什么
- 遇到与用户目标冲突或信息不足时先提问，不做大假设
- 报告结果时如实说明失败、跳过和未验证项
