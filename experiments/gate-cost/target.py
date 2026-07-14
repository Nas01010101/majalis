"""Autoresearch target: measure majalis $/question on held-out seeds 5,6.

EDITABLE KNOBS (the only thing iterations change):
"""
GATE_K = "2"            # disagreement samples per gate decision
SKIP_DOUBT = "0.15"     # skip the sampler when max_doubt < this and not weak

# ---- harness below: do not edit -------------------------------------------
import json
import os
import pathlib
import subprocess
import sys

AGORA = pathlib.Path.home() / "Projects" / "majalis"
SEEDS = (5, 6)

for seed in SEEDS:  # never resume a cached cell — force live measurement
    (AGORA / "results" / "raw" / f"session_majalis_s{seed}_t8.jsonl").unlink(missing_ok=True)

env = dict(os.environ, MAJALIS_GATE_K=GATE_K, MAJALIS_GATE_SKIP_DOUBT=SKIP_DOUBT)
proc = subprocess.run(
    [str(AGORA / ".venv" / "bin" / "python"), "-m", "majalis.bench.session",
     "--arms", "majalis", "--seeds", ",".join(map(str, SEEDS)), "--steps", "8"],
    cwd=AGORA, env=env, capture_output=True, text=True, timeout=1700)
print(proc.stdout[-1500:], proc.stderr[-500:], file=sys.stderr)

correct = n = 0
cost = 0.0
for seed in SEEDS:
    for line in (AGORA / "results" / "raw" / f"session_majalis_s{seed}_t8.jsonl").read_text().splitlines():
        r = json.loads(line)
        n += 1
        correct += r["correct"]
        cost += r["cost_usd"] + r.get("ingest", {}).get("cost_usd", 0.0)

# Baseline config scores 31/32 on these seeds (one non-weak residual error,
# inside the calibrated 2.5% band) — the constraint is "no worse than
# baseline", not perfection. Amended before iteration 1; see program.md.
BASELINE_CORRECT = 31
metric = cost / n if (n == 32 and correct >= BASELINE_CORRECT) else 999.0
pathlib.Path(os.environ["AUTORESEARCH_RESULTS"]).write_text(json.dumps(
    {"metric": metric, "accuracy": f"{correct}/{n}", "cost_total": round(cost, 4)}))
print(f"accuracy {correct}/{n}  metric {metric}")
