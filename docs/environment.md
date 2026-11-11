# 本机环境清单

> **每次升级驱动、CUDA、vLLM 或 vLLM-Omni 后都要更新本文件**，否则实验数据无法复现。
> 当前数据由 `bash scripts/check_env.sh` 于 2026-11-08 采集。

## 采集时间

2026-11-08

## 硬件

| 项目 | 值 | 采集命令 |
|---|---|---|
| GPU 型号 / 显存 | **2 × NVIDIA GeForce RTX 4090，24564 MiB/卡** | `nvidia-smi --query-gpu=name,memory.total --format=csv` |
| GPU 拓扑 | **NODE（PCIe，经同一 NUMA 节点的 PCIe Host Bridge；无 NVLink）** | `nvidia-smi topo -m` |
| 驱动版本 | **580.178.04** | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| CUDA 版本（驱动支持） | **13.0** | `nvidia-smi` 右上角 |
| CPU / 内存 | 192 核 / 251 GiB（可用约 243 GiB） | `nproc`、`free -g` |
| 磁盘可用 | 1.6T（根分区 1.8T，已用 3%） | `df -h /home/zliuz` |
| 操作系统 | Ubuntu 24.04.5 LTS (Noble) | `lsb_release -a` |

**拓扑解读**：两张 4090 之间是 PCIe `NODE` 连接，没有 NVLink/P2P。多卡并行的通信成本不可忽略，7B 级模型应优先实测「2×TP=1（数据并行）」而非默认 TP=2。

**GPU 共享**：2026-11-08 安装时，GPU0 被其他用户的 SGLang 进程占用约 16GB（`sglang::scheduler`），GPU1 空闲。运行前先执行 `nvidia-smi`；单卡任务优先 `CUDA_VISIBLE_DEVICES=1`，并与占卡用户协调。

## 软件

| 项目 | 值 | 说明 |
|---|---|---|
| Python（系统） | **3.12.3**（`/usr/bin/python3`），**缺 `python3.12-dev` / `Python.h`** | 不要用它重建 `.venv` |
| Python（venv） | **3.12.16**（uv-managed，含 `Python.h`） | `.venv/bin/python` |
| uv | **0.13.7**（`/home/zliuz/.local/bin/uv`） | `~/.local/bin` 已在 PATH |
| 本地虚拟环境 | **`.venv/`**（Python 3.12.16，280 个包） | `source .venv/bin/activate` |
| git | 2.43.0 | |
| gcc | 13.3.0 | Triton JIT 编译依赖 |
| nvcc / CUDA Toolkit | **未安装** | 不需要（wheel 自带运行时）；源码编译时才需要 |
| PyTorch | **2.13.0+cu130** | 随 vLLM 安装 |
| vLLM | **0.32.0** | PyPI wheel（cp38-abi3, manylinux_2_28_x86_64） |
| vLLM-Omni | **0.32.0** | PyPI wheel（py3-none-any） |
| FlashInfer | **0.7.0.post1**（cubin + jit-cache cu130，含 sm89） | 由 `vllm download-kernels` 安装 |
| Docker | **未安装** | 项目 1 先以 venv + tmux 运行；Compose 作为可选增强 |
| nvidia-container-runtime | **未找到** | 同上 |

固定版本清单见仓库根目录 `requirements-vllm.txt`。安装命令：

```bash
source .venv/bin/activate
uv pip install -r requirements-vllm.txt
vllm download-kernels   # 需要 GitHub 代理
```

## 网络与模型源

| 项目 | 值 |
|---|---|
| Hugging Face 直连 | **不可用（超时）** |
| HF 镜像 `hf-mirror.com` | **可用** |
| `HF_ENDPOINT` | 下载时设置为 `https://hf-mirror.com` |
| GitHub 直连 | **不可用（超时，raw.githubusercontent.com / Releases 均失败）** |
| GitHub 本地代理 | **`http://127.0.0.1:27897` 可用** |
| ModelScope | **可用**（2026-11-09 实测，无需代理） |

下载模型（脚本默认优先 ModelScope，其次 HF）：

```bash
bash scripts/download_models.sh Qwen/Qwen2.5-7B-Instruct

# 指定下载源
MODEL_SOURCE=modelscope bash scripts/download_models.sh Qwen/Qwen2.5-7B-Instruct
MODEL_SOURCE=hf HF_ENDPOINT=https://hf-mirror.com bash scripts/download_models.sh Qwen/Qwen2.5-7B-Instruct
```

`HF_ENDPOINT` 是 `huggingface_hub` 官方支持的端点覆盖变量（默认 `https://huggingface.co`）；
`https://hf-mirror.com` 是第三方镜像，2026-11-09 实测直连可用。ModelScope 下载不使用该变量。

访问 GitHub（克隆源码、下载 FlashInfer kernels）：

```bash
export HTTP_PROXY=http://127.0.0.1:27897
export HTTPS_PROXY=http://127.0.0.1:27897
```

## 版本锁定记录

| 日期 | 组件 | 旧版本 | 新版本 | 原因 | 结果/影响 |
|---|---|---|---|---|---|
| 2026-11-08 | 初始环境 | — | 驱动 580.178.04 / CUDA 13.0 / Python 3.12.3 | 基线 | — |
| 2026-11-08 | 工具链 | — | uv 0.13.7 + 开发依赖 | 项目 0 准备 | pytest / ruff / FastAPI 可用 |
| 2026-11-08 | Python venv | 3.12.3（系统，缺头文件） | **3.12.16（uv-managed）** | Triton JIT 需要 `Python.h` | 冒烟测试通过 |
| 2026-11-08 | vLLM | — | **0.32.0** | 与 vLLM-Omni 0.32.0 配对；cu130 wheel | 成功 |
| 2026-11-08 | vLLM-Omni | — | **0.32.0** | 与 vLLM 0.32.0 配对 | 成功 |
| 2026-11-08 | FlashInfer | — | **0.7.0.post1 + jit-cache cu130** | 预编译 kernel | 导入警告消失 |
| 2026-11-09 | 模型下载工具 | — | **modelscope 1.40.2** | 国内网络优先，HF 直连超时 | 单文件下载验证通过 |
| 2026-11-11 | 官方压测输入加载依赖 | 未安装 | **pandas 2.3.3 + pytz 2026.6** | vLLM 0.32.0 的 custom JSONL 加载器需要 pandas；已加入 requirements-dev.txt | 安装成功；vLLM / vLLM-Omni 仍为 0.32.0 |

## 已知问题与待办

- **GPU0 可能被其他用户占用**：运行前 `nvidia-smi`；冒烟测试使用 `CUDA_VISIBLE_DEVICES=1`
- **系统 Python 缺 `Python.h`**：不要用 `/usr/bin/python3` 重建 `.venv`；`scripts/setup_env.sh` 已改用 uv-managed Python
- **GitHub 直连超时**：克隆源码、下载 kernel 时设置本地代理
- **Hugging Face 直连超时**：设置 `HF_ENDPOINT=https://hf-mirror.com`
- **vLLM 不能用 stdin 脚本**：`python - <<'PY'` 会因 spawn 子进程无法导入 `<stdin>` 失败；必须写成文件运行（见 `project0-baseline/scripts/smoke_test.py`）
- **vLLM / vLLM-Omni 必须成对升级**：当前固定 0.32.0
- **Docker 未安装**：项目 1 先用 venv + tmux
- **CUDA 13.0 是驱动支持版本**，不代表安装了 CUDA Toolkit；wheel 自带运行时
- vLLM-Omni 的模型支持列表与启动方式以 `external/vllm-omni` 的 README / examples 为准
