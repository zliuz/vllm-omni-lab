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

## 软件

| 项目 | 值 | 说明 |
|---|---|---|
| Python | **3.12.3**（`/usr/bin/python3`） | vLLM 通常支持 3.9–3.12，安装前以目标版本 `pyproject.toml` 为准 |
| uv | **未安装** | 需先安装：`curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| git | 2.43.0 | |
| nvcc / CUDA Toolkit | **未安装** | 只跑 wheel 不需要；源码编译时需要与 wheel 匹配的 CUDA Toolkit |
| PyTorch | **未安装** | 随 vLLM 安装 |
| vLLM | **未安装** | 安装后填写版本 + commit |
| vLLM-Omni | **未克隆 / 未安装** | 安装后填写 commit |
| Docker | **未安装** | 项目 1 的 Compose 部署需要 Docker + nvidia-container-toolkit；未安装时可先用 venv + tmux |
| nvidia-container-runtime | **未找到** | 同上 |

## 网络与模型源

| 项目 | 值 |
|---|---|
| Hugging Face 直连 / 镜像 | **未配置**（`HF_ENDPOINT` 未设置） |
| `HF_ENDPOINT` | `<unset>` |
| ModelScope 是否可用 | 待验证 |

国内网络建议在 `~/.bashrc` 或每次会话中设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或使用 ModelScope 下载（见 `scripts/download_models.sh`）。

## 版本锁定记录

| 日期 | 组件 | 旧版本 | 新版本 | 原因 | 结果/影响 |
|---|---|---|---|---|---|
| 2026-11-08 | 初始环境 | — | 驱动 580.178.04 / CUDA 13.0 / Python 3.12.3 | 基线 | vLLM 尚未安装 |

## 已知问题与待办

- **uv 未安装**：`scripts/setup_env.sh` 会退出并给出安装命令
- **Docker 未安装**：项目 1 先以「venv + 多进程/tmux」方式运行，Compose 作为可选增强
- **CUDA 13.0 是驱动支持版本**，不代表已安装 CUDA Toolkit；vLLM wheel 通常自带 CUDA 运行时
- 首次安装 vLLM 时确认 wheel 的 CUDA 版本与驱动兼容（580 驱动向下兼容 12.x）
- vLLM-Omni 的模型支持列表与启动方式以 `external/vllm-omni` 的 README / examples 为准
