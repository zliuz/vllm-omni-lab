# Docker 与 NVIDIA Container Toolkit

> 面向：本仓库后续维护者与项目 1 阶段的自己。
> **结论先行**：Docker 不是 vLLM 运行的必要条件。项目 0/2/3/4 都不需要；只有项目 1 的「一键启动网关 + 多个 vLLM 后端」会用到。当前环境**未安装**，等做到项目 1 的 Compose 验收项时再装。

## 1. 两者分别做什么

| 组件 | 作用 | 不装会怎样 |
|---|---|---|
| **Docker** | 把「Python + CUDA 用户态库 + vLLM + 网关代码」打包成镜像，隔离依赖、统一环境、编排多服务 | 只能在本机用 venv 跑；换机器/换版本要重装约 18GB 依赖 |
| **NVIDIA Container Toolkit**（提供 `nvidia-container-runtime`） | Docker 启动容器时，把宿主机的 `/dev/nvidia*` 设备与驱动库（`libcuda.so`）注入容器 | 容器里 `nvidia-smi` 报错，vLLM 看不到 GPU |

分层关系：

```text
容器内：vLLM + PyTorch + CUDA 用户态库（镜像里）
          │
runc → nvidia-container-runtime（启动时注入设备和驱动）
          │
宿主机：NVIDIA 驱动 580.178.04（不在容器里）
          │
2 × RTX 4090
```

关键点：

- **驱动永远在宿主机**，容器不安装驱动。
- Toolkit 只是把宿主驱动「挂」进容器。
- 宿主机驱动版本决定容器能用哪个 CUDA 版本。本机 580 支持 CUDA 13.0，正好匹配 vLLM 0.32 的 cu130 依赖。

## 2. 在本项目中的用途

### 项目 1：推理网关（主要用途）

```text
客户端 → gateway 容器 :9000
              ├── vLLM 后端 A 容器（GPU0）
              └── vLLM 后端 B 容器（GPU1）
```

Docker Compose 负责：

- 一条命令拉起 gateway + 2 个 vLLM 后端
- 容器间网络、端口映射
- 健康检查、崩溃重启
- 按 GPU 分配设备（`deploy.resources.reservations.devices`）

没有 Docker 也能做这个项目，只是要手动开 3 个终端或写 tmux/systemd 脚本。

### 各项目需求

| 项目 | 是否需要 Docker | 说明 |
|---|---|---|
| 项目 0 基线压测 | 否 | venv 已足够 |
| 项目 1 推理网关 | **Compose 阶段需要** | 也可先用 venv + tmux |
| 项目 2 Omni 语音助手 | 否 | 官方 nightly 镜像可省去 18GB 依赖安装 |
| 项目 3 二次开发 | 否 | venv + 源码更适合改代码 |
| 项目 4 上游贡献 | 否 | — |
| 部署验证 / 复现实验 | **建议** | Docker 的真正价值所在 |

Docker 的核心价值不是「能跑」，而是：

1. **可复现**：镜像 tag 固定住 vLLM / torch / CUDA / 依赖的整组版本
2. **可迁移**：同一镜像推到服务器，不会出现「我机器上能跑」
3. **可编排**：多服务、多 GPU 的启停和依赖关系
4. **可回滚**：升级失败直接切回旧镜像

## 3. 当前环境状态（2026-11-08）

| 项目 | 状态 |
|---|---|
| Docker | **未安装** |
| `nvidia-container-runtime` | **未找到** |
| sudo | **需要密码**（无免密 sudo），安装需要用户本人执行 |
| GPU0 | 被其他用户的 SGLang 占用约 16GB |
| GPU1 | 空闲 |
| Docker Hub 连通性 | **未验证**（本机 GitHub 直连超时，镜像拉取可能需要代理或镜像源） |
| 宿主机驱动 / CUDA | 580.178.04 / 驱动支持 CUDA 13.0 |

## 4. 安装步骤（需要 sudo）

### 4.1 安装 Docker Engine

推荐 Docker 官方源（Ubuntu 自带的 `docker.io` 版本较旧）：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
```

把当前用户加入 docker 组，避免每条命令都加 `sudo`（需要重新登录生效）：

```bash
sudo usermod -aG docker "$USER"
```

如果官方源访问不了，可先用 Ubuntu 仓库的 `docker.io docker-compose-v2` 代替，功能足够。

### 4.2 安装 NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 4.3 验证

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

预期看到两张 RTX 4090。镜像 tag 以 NGC 实际提供的为准；关键是 CUDA 版本不超过宿主驱动支持的上限。

验证清单：

- [ ] `docker info` 能看到 `Runtimes: nvidia`
- [ ] 上面的命令输出两张 4090
- [ ] 普通用户无需 `sudo` 即可执行 `docker ps`

## 5. 代理与镜像源

本机 GitHub 直连超时，Docker Hub 也未必可达。**安装前先验证拉取**：

```bash
docker pull hello-world
```

如果失败，按需配置：

1. **Docker daemon 代理**（拉镜像走代理）：

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf > /dev/null <<'EOF'
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:27897"
Environment="HTTPS_PROXY=http://127.0.0.1:27897"
Environment="NO_PROXY=localhost,127.0.0.1"
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

2. **镜像加速器**：如果所在网络提供 Docker Hub mirror，写入 `/etc/docker/daemon.json` 的 `registry-mirrors`。

3. **构建时下载依赖**：`docker build --build-arg HTTP_PROXY=... --build-arg HTTPS_PROXY=...`。

> 注意：容器内的 `127.0.0.1` 指的是容器自己，不是宿主机。宿主机的本地代理需要在 Compose 中加
> `extra_hosts: ["host.docker.internal:host-gateway"]`，然后使用 `http://host.docker.internal:27897`；
> 或者直接 `network_mode: host`（会失去端口隔离）。

## 6. Compose 示例：网关 + 两个 vLLM 后端

> 以下为示意配置，实际镜像名、tag、启动参数以 vLLM / vLLM-Omni 官方镜像文档为准。

```yaml
services:
  vllm-a:
    image: vllm/vllm-openai:v0.32.0        # 固定 tag，与 requirements-vllm.txt 对齐
    command: ["--model", "Qwen/Qwen2.5-7B-Instruct",
              "--max-model-len", "8192",
              "--gpu-memory-utilization", "0.90"]
    environment:
      - HF_ENDPOINT=https://hf-mirror.com
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/models"]
      interval: 30s
      timeout: 5s
      retries: 5

  vllm-b:
    image: vllm/vllm-openai:v0.32.0
    command: ["--model", "Qwen/Qwen2.5-7B-Instruct",
              "--max-model-len", "8192",
              "--gpu-memory-utilization", "0.90"]
    environment:
      - HF_ENDPOINT=https://hf-mirror.com
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    ports:
      - "8001:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/models"]
      interval: 30s
      timeout: 5s
      retries: 5

  gateway:
    build: ../project1-gateway
    ports:
      - "9000:9000"
    depends_on:
      - vllm-a
      - vllm-b
```

启动与验证：

```bash
docker compose up -d
docker compose ps
curl -s http://127.0.0.1:9000/v1/models
docker compose logs -f gateway
docker compose down
```

说明：

- `device_ids: ["0"]` / `["1"]` 用来把两个后端分到不同 GPU。
- 如果镜像里没有 `curl`，把 healthcheck 改成
  `["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/models')"]`。
- Compose 的 `deploy.resources.reservations.devices` 在 `docker compose`（v2）下可用，不要求 Swarm。
- 也可以使用 `runtime: nvidia` + `NVIDIA_VISIBLE_DEVICES=0`，但推荐上面的设备预留写法。

## 7. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | Toolkit 未安装 / 未配置。执行 `nvidia-ctk runtime configure` 后重启 Docker |
| `permission denied ... docker.sock` | 用户不在 `docker` 组；`sudo usermod -aG docker $USER` 后重新登录 |
| 容器内 `nvidia-smi: not found` | 镜像没装该工具；换用 `nvidia/cuda:*-base` 验证，或直接跑 Python 检查 `torch.cuda.is_available()` |
| `CUDA driver version is insufficient for CUDA runtime version` | 镜像的 CUDA 版本高于宿主驱动支持上限；换低 CUDA 的镜像或升级驱动 |
| 两个容器抢同一张卡导致 OOM | 设备预留**不做显存隔离**；用 `device_ids` 分卡，或与占卡用户协调 |
| 想让单卡跑多个隔离实例 | 4090 **不支持 MIG**；只能靠显存估算与调度错峰 |
| 镜像拉取超时 | 配置 daemon 代理 / registry mirror，见第 5 节 |
| 容器内下载 HF 模型失败 | 设置 `HF_ENDPOINT=https://hf-mirror.com`；容器内无法用宿主机的 `127.0.0.1` 代理 |
| 镜像太大（vLLM 约 10–20GB） | 确认 `/var/lib/docker` 所在分区空间；本机 1.6T 可用，问题不大 |

## 8. 替代方案

| 方案 | 适用场景 | 备注 |
|---|---|---|
| **venv + tmux/systemd** | 当前阶段、项目 0–4 | 已跑通；无额外依赖 |
| **官方镜像** | 不想自己装 18GB 依赖 | vLLM：`vllm/vllm-openai`；vLLM-Omni：`vllm/vllm-omni` 的 nightly 镜像 |
| **Apptainer/Singularity** | 共享集群不允许 Docker | 可运行同一镜像，但 Compose 不可用 |
| **Conda / uv** | 单机、单用户 | 与 Docker 不冲突，可并存 |

## 9. 什么时候安装

按项目路线：

- **项目 0**：不需要。当前 venv 已通过冒烟测试。
- **项目 1 前半段**：用 venv + tmux 跑通网关与两个后端，先验证路由、重试、流式。
- **项目 1 的 Compose 验收项**：此时安装 Docker + Toolkit，把已验证的进程编排成 Compose。
- **项目 2**：若不想维护 18GB venv，可切换官方镜像；否则继续 venv。

## 10. 参考

- Docker Engine on Ubuntu：<https://docs.docker.com/engine/install/ubuntu/>
- NVIDIA Container Toolkit：<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
- Docker Compose GPU 支持：<https://docs.docker.com/compose/how-tos/gpu-support/>
- vLLM Docker 部署：<https://docs.vllm.ai/en/latest/deployment/docker.html>

> 本机直连 GitHub / Hugging Face 超时；访问上述文档或拉取镜像前，先确认代理与镜像源可用。
