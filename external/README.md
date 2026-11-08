# external/

上游源码的克隆位置，**只读参考，不提交到本仓库**（已在 `.gitignore` 中忽略）。

```text
external/
├── vllm/          # https://github.com/vllm-project/vllm
└── vllm-omni/     # https://github.com/vllm-project/vllm-omni
```

## 克隆

```bash
bash scripts/clone_upstream.sh
```

脚本支持通过环境变量固定版本：

```bash
VLLM_REF=v0.x.y VLLM_OMNI_REF=<commit> bash scripts/clone_upstream.sh
```

**务必固定版本**：vLLM-Omni 迭代很快，默认分支随时可能变化，未固定的基线无法复现实验。

## 规则

- 不修改、不格式化、不提交此目录
- 需要临时改动时，用 `git diff` 导出 patch 到 `project3-perf/` 或 `project4-upstream/notes/`
- 源码阅读笔记写到 `docs/source-map.md` 的「问题记录」
- vLLM-Omni 的 `pyproject.toml` 会锁定兼容的 vLLM 版本，安装时以它为准
