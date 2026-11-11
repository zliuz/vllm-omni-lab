"""Plot all measured round variation, retaining the concurrency-64 tail spike."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
data = json.loads((out / "audit.json").read_text())
configs = data["configurations"]
x = [item["concurrency"] for item in configs]
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)

def band(metric):
    med = [item["stats"][metric]["median"] for item in configs]
    low = [item["stats"][metric]["min"] for item in configs]
    high = [item["stats"][metric]["max"] for item in configs]
    return med, [[a-b for a,b in zip(med,low)], [a-b for a,b in zip(high,med)]]

med, err = band("output_tok_s")
axes[0].errorbar(x, med, yerr=err, marker="o", capsize=5)
axes[0].set_ylabel("Successful output tokens / second")
axes[0].set_title("Throughput: median and min/max of 3 rounds")
for p in (50, 95):
    values = [item["stats"][f"ttft_ms_p{p}"]["median"] for item in configs]
    axes[1].plot(x, values, marker="o", label=f"p{p}: median of 3 rounds")
med, err = band("ttft_ms_p99")
axes[1].errorbar(x, med, yerr=err, marker="o", capsize=5, label="p99: median and min/max")
axes[1].set_yscale("log")
axes[1].set_ylabel("Client TTFT (ms, log scale)")
axes[1].set_title("Tail variation retained; no rounds discarded")
axes[1].annotate("c=64 round 2\np99 = 7362.91 ms", xy=(64, configs[-1]["stats"]["ttft_ms_p99"]["max"]), xytext=(0.40, 0.72), textcoords="axes fraction", arrowprops={"arrowstyle":"->"}, fontsize=9)
axes[1].legend(loc="upper left", fontsize=8)
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(c) for c in x])
    ax.set_xlabel("In-flight concurrency")
    ax.grid(alpha=0.25, which="both")
fig.suptitle("Qwen2.5-7B-Instruct | vLLM 0.32.0 | RTX 4090 GPU1 | TP=1 | Prefix cache ON")
path = out / "prefix-on-variation-v2.png"
assert not path.exists()
fig.savefig(path, dpi=180)
plt.close(fig)
for source, target in ((Path(__file__), "plot-variation-v2-source.py"),):
    with (out / target).open("x", encoding="utf-8") as stream:
        stream.write(source.read_text())
print(path)
