"""vLLM 最小冒烟测试。

用法：
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python project0-baseline/scripts/smoke_test.py [model]

注意：必须作为文件运行，不能用 `python - <<'PY'`。
vLLM 默认使用 spawn 多进程，标准输入脚本没有可重新导入的文件路径。
"""

import sys

from vllm import LLM, SamplingParams


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen2.5-0.5B-Instruct"
    llm = LLM(
        model=model,
        max_model_len=1024,
        gpu_memory_utilization=0.35,
    )
    params = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = llm.generate(
        ["What is PagedAttention? Answer in one sentence."],
        params,
    )
    print("OUTPUT:", outputs[0].outputs[0].text.strip())


if __name__ == "__main__":
    main()
