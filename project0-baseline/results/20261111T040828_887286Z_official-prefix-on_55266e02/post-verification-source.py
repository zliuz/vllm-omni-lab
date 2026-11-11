"""Verify the final service/source state and newly written documentation."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import httpx

root = Path.cwd()
out = root / "project0-baseline/results/20261111T040828_887286Z_official-prefix-on_55266e02"
context = json.loads((out / "context.json").read_text())
installed = root / ".venv/lib/python3.12/site-packages/vllm/benchmarks"
for rel, expected in context["official_source_sha256"].items():
    assert hashlib.sha256((installed / rel).read_bytes()).hexdigest() == expected
with httpx.Client(base_url="http://127.0.0.1:8000", trust_env=False, timeout=10) as client:
    health = client.get("/health")
    health.raise_for_status()
    version = client.get("/version")
    version.raise_for_status()
    metrics = client.get("/metrics")
    metrics.raise_for_status()
    assert 'enable_prefix_caching="True"' in metrics.text
    assert version.json()["version"] == "0.32.0"
argv = Path("/proc/439160/cmdline").read_bytes().decode().strip("\0").split("\0")
assert argv == context["server_command"]
docs = [root / "project0-baseline/README.md",
        root / "project0-baseline/experiments/20261111-official-vs-custom-55266e02.md"]
checked_links = 0
checked_shell = 0
for doc in docs:
    text = doc.read_text()
    for target in re.findall(r"\]\(([^\s)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        path = (doc.parent / target.split("#")[0]).resolve()
        assert path.exists(), (doc, target)
        checked_links += 1
    if "experiments" in doc.parts:
        for block in re.findall(r"```bash\n(.*?)```", text, flags=re.S):
            subprocess.run(["bash", "-n"], input=block, text=True, check=True)
            checked_shell += 1
result = {"health_status": health.status_code, "server_version": version.json(),
          "prefix_caching_still_enabled": True, "original_server_pid_and_command_unchanged": True,
          "installed_official_source_hashes_unchanged": True,
          "checked_local_links": checked_links, "bash_blocks_syntax_checked": checked_shell}
with (out / "post-verification.json").open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
with (out / "post-verification-source.py").open("x", encoding="utf-8") as stream:
    stream.write(Path(__file__).read_text())
print(json.dumps(result, ensure_ascii=False, indent=2))
