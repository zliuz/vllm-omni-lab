"""Check the installed official dataset loader without sending requests."""
import hashlib
import json
from pathlib import Path

from vllm.benchmarks.datasets.datasets import CustomDataset
from vllm.tokenizers import get_tokenizer

root = Path.cwd()
out = root / "project0-baseline/results/20261111T040828_887286Z_official-prefix-on_55266e02"
original = root / "benchmarks/prompts/baseline-v1.jsonl"
raw = [json.loads(line) for line in original.read_text().splitlines()]
tokenizer = get_tokenizer("models/Qwen2.5-7B-Instruct")
checks = []
for filename, n in (("prompts-128-source.jsonl", 128), ("prompts-512-in-order.jsonl", 512)):
    dataset = CustomDataset(dataset_path=str(out / filename), disable_shuffle=True, random_seed=42)
    samples = dataset.sample(tokenizer=tokenizer, num_requests=n, output_len=128,
                             skip_chat_template=True, no_oversample=True)
    assert len(samples) == n
    assert all(sample.prompt == raw[i % 128]["prompt"] for i, sample in enumerate(samples))
    assert all(sample.expected_output_len == 128 for sample in samples)
    assert all(sample.chat_messages is None for sample in samples)
    checks.append({"file": filename, "count": n, "raw_prompts_in_expected_order": True,
                   "output_cap_128": True, "no_client_chat_template": True})
previous = root / "project0-baseline/results/20261110T140555_003380Z_prefix-on_8114dafc/model-artifacts.json"
hashes = json.loads(previous.read_text())
for filename, value in hashes.items():
    assert hashlib.sha256((root / "models/Qwen2.5-7B-Instruct" / filename).read_bytes()).hexdigest() == value["sha256"]
evidence = {"dataset_checks": checks, "model_config_hashes_match_custom_run": hashes}
with (out / "alignment-check.json").open("x", encoding="utf-8") as stream:
    json.dump(evidence, stream, ensure_ascii=False, indent=2)
with (out / "alignment-check-source.py").open("x", encoding="utf-8") as stream:
    stream.write(Path(__file__).read_text())
print(json.dumps(evidence, ensure_ascii=False, indent=2))
