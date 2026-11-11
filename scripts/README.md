# 仓库公共脚本

本目录负责环境检查、开发环境准备、上游克隆和模型下载。
项目 0 的服务、冒烟测试及压测说明见
[project0-baseline/scripts/README.md](../project0-baseline/scripts/README.md)。
以下命令均从仓库根目录执行。

| 脚本 | 功能 | 主要输入 | 输出 / 副作用 |
|---|---|---|---|
| `check_env.sh` | 只读检查系统、硬件、Python 包及上游版本 | 当前 shell 的 PATH / Python 环境 | 仅打印到终端；**不会自动写入** `docs/environment.md` |
| `setup_env.sh` | 用 uv 创建 `.venv` 并安装开发依赖 | `requirements-dev.txt`；可选 `PYTHON_BIN` | 默认选 uv-managed Python 3.12，必要时下载安装；复用已有 `.venv`，不安装 vLLM |
| `clone_upstream.sh` | 将 vLLM / vLLM-Omni 克隆到 `external/` | 可选 `VLLM_REF` / `VLLM_OMNI_REF` | 新克隆时可 checkout 指定版本；已有仓库直接跳过，不更新、不校验 ref |
| `download_models.sh` | 下载一个或多个模型 | 位置参数为模型 ID；默认 Qwen2.5-7B-Instruct | 放到 `models/<模型名>/`；不进入 git |

## 环境检查与安装

```bash
# 已有环境时先激活，让检查脚本看到正确的 Python 包。
source .venv/bin/activate
bash scripts/check_env.sh
```

首次准备开发环境时：

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
uv pip install -r requirements-vllm.txt
```

`setup_env.sh` 要求 PATH 中已有 uv。本机系统 Python 缺开发头文件，
优先保留已有的 uv-managed Python 环境；头文件检查仅告警，不自动重建环境。
环境检查结束后手工更新 `docs/environment.md`，不要把旧数据当作本次采集结果。

## 上游源码

源码版本必须满足 AGENTS.md 中的配对约束。先确定匹配固定 vLLM 版本的
vLLM-Omni commit，再将两个 ref 通过环境变量传给克隆脚本。

```bash
HTTP_PROXY=http://127.0.0.1:27897 \
HTTPS_PROXY=http://127.0.0.1:27897 \
bash scripts/clone_upstream.sh
```

上面的无 ref 示例会克隆默认分支，**不构成版本锁定**，不能直接作为基线。
`VLLM_REF` 应为固定 release tag，`VLLM_OMNI_REF` 应为已核对兼容性的固定 commit。
脚本不负责选择或升级版本；`external/` 的源码只读，不在里面修改、格式化或提交。

## 模型下载

```bash
bash scripts/download_models.sh Qwen/Qwen2.5-0.5B-Instruct
bash scripts/download_models.sh Qwen/Qwen2.5-7B-Instruct

MODEL_SOURCE=hf HF_ENDPOINT=https://hf-mirror.com \
bash scripts/download_models.sh Qwen/Qwen2.5-7B-Instruct
```

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_DIR` | 仓库的 `models/` | 下载根目录，实际子目录取模型 ID 的 basename |
| `MODEL_SOURCE` | `auto` | 按**已安装命令**选择 modelscope → hf → huggingface-cli；下载失败不会自动换源 |
| `HF_ENDPOINT` | 未设置 | HF 使用的端点覆盖；ModelScope 不使用此变量 |

脚本自动优先使用 `.venv/bin` 中的下载工具。ModelScope 和 HF 的模型 ID
可能不同，需要自行确认。同 basename 的不同模型会使用同一目标目录，
遇到这种情况应通过 `MODEL_DIR` 分开存放。

下载到 `models/` 后，启动服务应显式传本地路径：

```bash
CUDA_VISIBLE_DEVICES=1 MODEL=models/Qwen2.5-7B-Instruct \
bash project0-baseline/scripts/serve.sh
```

Makefile 的 `check-env` / `setup` / `clone` / `download` 分别对应上述公共脚本；
`make download` 使用默认模型，传多个模型 ID 时直接调用下载脚本。
