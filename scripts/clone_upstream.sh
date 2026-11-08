#!/usr/bin/env bash
# 克隆上游源码到 external/（只读参考，已被 gitignore）。
# 用法：
#   bash scripts/clone_upstream.sh
#   VLLM_REF=v0.x.y VLLM_OMNI_REF=<commit> bash scripts/clone_upstream.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/external"

clone_one() {
  local name="$1" url="$2" ref="${3:-}"
  local dir="$ROOT/external/$name"

  if [ -d "$dir/.git" ]; then
    echo "[$name] 已存在，跳过克隆：$(git -C "$dir" rev-parse --short HEAD)"
    return 0
  fi

  echo "[$name] cloning $url"
  git clone "$url" "$dir"

  if [ -n "$ref" ]; then
    git -C "$dir" checkout "$ref"
  fi
  echo "[$name] HEAD = $(git -C "$dir" rev-parse --short HEAD)"
}

clone_one vllm https://github.com/vllm-project/vllm "${VLLM_REF:-}"
clone_one vllm-omni https://github.com/vllm-project/vllm-omni "${VLLM_OMNI_REF:-}"

cat <<'EOF'

提示：
- 未设置 VLLM_REF / VLLM_OMNI_REF 时使用默认分支，基线随时可能变化。
- 请尽快固定版本，并把版本/commit 记录到 docs/environment.md。
- vLLM-Omni 的 pyproject.toml 会锁定兼容的 vLLM 版本，安装时以它为准。
EOF
