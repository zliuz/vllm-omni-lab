#!/usr/bin/env bash
# 下载模型权重到 models/（不入库）。
# 用法：
#   bash scripts/download_models.sh
#   bash scripts/download_models.sh Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-Omni-3B
# 可用环境变量：
#   MODEL_DIR     下载目录（默认 <repo>/models）
#   MODEL_SOURCE  auto | modelscope | hf | huggingface-cli
#                 默认 auto：依次尝试 modelscope → hf → huggingface-cli
#   HF_ENDPOINT   例如 https://hf-mirror.com（仅对 hf / huggingface-cli 生效）
# 注意：ModelScope 的仓库 id 与 Hugging Face 不一定完全相同（常见 Qwen 官方模型一致）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 优先使用仓库虚拟环境中的下载工具（无需手动 activate）
if [ -d "$ROOT/.venv/bin" ]; then
  export PATH="$ROOT/.venv/bin:$PATH"
fi
DEST="${MODEL_DIR:-$ROOT/models}"
mkdir -p "$DEST"

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=(Qwen/Qwen2.5-7B-Instruct)
fi

# 选定下载器：默认国内网络优先 ModelScope
DOWNLOADER="${MODEL_SOURCE:-auto}"
if [ "$DOWNLOADER" = "auto" ]; then
  for cmd in modelscope hf huggingface-cli; do
    if command -v "$cmd" >/dev/null 2>&1; then
      DOWNLOADER="$cmd"
      break
    fi
  done
  if [ "$DOWNLOADER" = "auto" ]; then
    cat >&2 <<'EOF'
未找到下载工具。可任选其一安装（本仓库使用 uv 管理的 .venv）：
  uv pip install modelscope                  # 提供 modelscope（国内网络优先）
  uv pip install "huggingface_hub[cli]"      # 提供 hf / huggingface-cli
EOF
    exit 1
  fi
else
  case "$DOWNLOADER" in
    modelscope|hf|huggingface-cli)
      if ! command -v "$DOWNLOADER" >/dev/null 2>&1; then
        echo "错误：MODEL_SOURCE=$DOWNLOADER，但未找到该命令。" >&2
        exit 1
      fi
      ;;
    *)
      echo "错误：MODEL_SOURCE 只能是 auto|modelscope|hf|huggingface-cli（当前：$DOWNLOADER）" >&2
      exit 1
      ;;
  esac
fi

echo "下载目录: $DEST"
echo "下载工具: $DOWNLOADER"
if [ "$DOWNLOADER" = "modelscope" ]; then
  echo "ModelScope 不使用 HF_ENDPOINT；若仓库 id 与 HF 不同需手动传入 ModelScope 的 id。"
elif [ -n "${HF_ENDPOINT:-}" ]; then
  echo "HF_ENDPOINT=$HF_ENDPOINT"
else
  echo "提示：HF 直连超时，可先 export HF_ENDPOINT=https://hf-mirror.com"
fi

for model in "${MODELS[@]}"; do
  local_dir="$DEST/$(basename "$model")"
  echo "--- $model -> $local_dir"
  case "$DOWNLOADER" in
    modelscope)
      modelscope download "$model" --local-dir "$local_dir"
      ;;
    hf)
      hf download "$model" --local-dir "$local_dir"
      ;;
    huggingface-cli)
      huggingface-cli download "$model" --local-dir "$local_dir"
      ;;
  esac
done

echo "完成。模型目录：$DEST"
