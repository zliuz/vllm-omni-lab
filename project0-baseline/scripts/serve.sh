#!/usr/bin/env bash
# 启动项目 0 的基线服务。
# 用法：
#   bash project0-baseline/scripts/serve.sh [model]
# 可覆盖的环境变量：PORT TP MAX_MODEL_LEN GPU_MEM_UTIL MAX_NUM_SEQS EXTRA_ARGS
set -euo pipefail

MODEL="${1:-${MODEL:-Qwen/Qwen2.5-7B-Instruct}}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "未找到 vllm 命令，请先安装并激活虚拟环境。" >&2
  echo "安装方式见 project0-baseline/README.md" >&2
  exit 1
fi

echo "model=$MODEL port=$PORT tp=$TP max_model_len=$MAX_MODEL_LEN extra='$EXTRA_ARGS'"

# EXTRA_ARGS 需要按空格拆分，故意不加引号
# shellcheck disable=SC2086
exec vllm serve "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  $EXTRA_ARGS
