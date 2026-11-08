#!/usr/bin/env bash
# 创建 .venv 并安装开发依赖（不安装 vllm 本身）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
未找到 uv。建议安装：
  curl -LsSf https://astral.sh/uv/install.sh | sh
然后重新运行本脚本。
EOF
  exit 1
fi

if [ ! -d .venv ]; then
  echo "创建 .venv（Python: $PYTHON_BIN）"
  uv venv --python "$PYTHON_BIN" .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "安装开发依赖..."
uv pip install -r requirements-dev.txt

cat <<'EOF'

完成。后续使用前先激活环境：
  source .venv/bin/activate

vllm 请按项目/版本单独安装，并把版本记录到 docs/environment.md。
EOF
