#!/usr/bin/env bash
# 只读环境检查：不安装、不修改任何配置。
# 注意：故意不使用 `set -e`，某个命令缺失时仍继续输出其余信息。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
say() { printf '\n=== %s ===\n' "$*"; }

say "OS"
grep -E '^(NAME|VERSION)=' /etc/os-release 2>/dev/null || uname -a

say "CPU / Memory"
nproc 2>/dev/null || true
free -h 2>/dev/null || true

say "Disk"
df -h "$ROOT" 2>/dev/null || true

say "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || true
  echo
  echo "--- topology ---"
  nvidia-smi topo -m 2>/dev/null || true
else
  echo "nvidia-smi not found"
fi

say "Python / uv / git"
python3 --version 2>/dev/null || true
uv --version 2>/dev/null || true
git --version 2>/dev/null || true

say "Python packages"
python3 - <<'PY' 2>/dev/null || true
import importlib.util

for name in ("torch", "vllm", "vllm_omni", "transformers", "huggingface_hub"):
    if importlib.util.find_spec(name) is None:
        print(f"{name}: not installed")
        continue
    try:
        mod = __import__(name)
        print(f"{name}: {getattr(mod, '__version__', '(no __version__)')}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: import failed ({exc})")
PY

say "Network / model source"
echo "HF_ENDPOINT=${HF_ENDPOINT:-<unset>}"
echo "HF_HOME=${HF_HOME:-<unset>}"

say "Upstream repos"
for name in vllm vllm-omni; do
  dir="$ROOT/external/$name"
  if [ -d "$dir/.git" ]; then
    printf '%-10s %s\n' "$name:" "$(git -C "$dir" rev-parse --short HEAD 2>/dev/null)"
  else
    printf '%-10s %s\n' "$name:" "not cloned"
  fi
done

say "Done"
echo "把以上关键信息填入 docs/environment.md"
