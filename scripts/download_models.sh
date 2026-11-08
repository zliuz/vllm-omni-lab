#!/usr/bin/env bash
# 下载模型权重到 models/（不入库）。
# 用法：
#   bash scripts/download_models.sh
#   bash scripts/download_models.sh Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-Omni-3B
# 可用环境变量：
#   MODEL_DIR   下载目录（默认 <repo>/models）
#   HF_ENDPOINT 例如 https://hf-mirror.com
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${MODEL_DIR:-$ROOT/models}"
mkdir -p "$DEST"

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=(Qwen/Qwen2.5-7B-Instruct)
fi

echo "下载目录: $DEST"
if [ -n "${HF_ENDPOINT:-}" ]; then
  echo "HF_ENDPOINT=$HF_ENDPOINT"
else
  echo "提示：国内网络可先 export HF_ENDPOINT=https://hf-mirror.com"
fi

for model in "${MODELS[@]}"; do
  local_dir="$DEST/$(basename "$model")"
  echo "--- $model -> $local_dir"
  if command -v hf >/dev/null 2>&1; then
    hf download "$model" --local-dir "$local_dir"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$model" --local-dir "$local_dir"
  elif command -v modelscope >/dev/null 2>&1; then
    modelscope download --model "$model" --local_dir "$local_dir"
  else
    cat >&2 <<'EOF'
未找到下载工具。可任选其一安装：
  pip install "huggingface_hub[cli]"     # 提供 hf / huggingface-cli
  pip install modelscope                  # 提供 modelscope
EOF
    exit 1
  fi
done

echo "完成。模型目录：$DEST"
