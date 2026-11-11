#!/usr/bin/env bash
# 创建 .venv 并安装开发依赖（不安装 vllm 本身）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
未找到 uv。建议安装：
  curl -LsSf https://astral.sh/uv/install.sh | sh
然后重新运行本脚本。
EOF
  exit 1
fi

# 系统 Python 缺少开发头文件（Python.h）时，Triton / torch.compile 无法 JIT 编译。
# 默认使用 uv 托管的完整 Python（自带头文件），除非显式指定 PYTHON_BIN。
if [ -z "${PYTHON_BIN:-}" ]; then
  if ! uv python find 3.12 --managed-python >/dev/null 2>&1; then
    echo "安装 uv 托管的 Python 3.12 ..."
    uv python install 3.12
  fi
  PYTHON_BIN="$(uv python find 3.12 --managed-python)"
fi

if [ ! -d .venv ]; then
  echo "创建 .venv（Python: $PYTHON_BIN）"
  uv venv --python "$PYTHON_BIN" .venv
fi

INC="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("include"))')"
if [ ! -f "$INC/Python.h" ]; then
  echo "警告：$INC/Python.h 不存在，Triton JIT 会失败。请用 uv 托管 Python 重建 .venv。" >&2
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "安装开发依赖..."
uv pip install -r requirements-dev.txt

cat <<'EOF'

完成。后续使用前先激活环境：
  source .venv/bin/activate

vllm 请按项目/版本单独安装（推荐 requirements-vllm.txt），并把版本记录到 docs/environment.md。
EOF
