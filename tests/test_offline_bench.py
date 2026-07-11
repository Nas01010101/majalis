"""Offline end-to-end gate benchmark: schema + regression tripwires.

Runs the real scripts/offline_bench.py on a 5-seed micro-slice (zero LLM
calls, <1s) and asserts the properties the submission leans on. These are
tripwires, not claims — the full 100-seed numbers live in
results/offline_gate_eval.json.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_micro(tmp_path):
    out = tmp_path / "eval.json"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "offline_bench.py"),
         "--seeds", "5000:5005", "--out", str(out)],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(out.read_text())


def test_offline_bench_schema_and_tripwires(tmp_path):
    rep = _run_micro(tmp_path)
    for key in ("gate_quality", "learned_vs_heuristic", "reliability"):
        assert key in rep
    learned = rep["learned_vs_heuristic"]["learned"]
    heuristic = rep["learned_vs_heuristic"]["heuristic"]
    assert learned["calibrated"] and heuristic["calibrated"]
    assert learned["n_questions"] == heuristic["n_questions"] > 0

    # Regression tripwires on this slice (both held at wide margins on the
    # full run: recall .862 vs .788, false-fire .009 vs .151).
    assert learned["recall_poisoned"] >= heuristic["recall_poisoned"]
    assert learned["false_fire_rate"] <= heuristic["false_fire_rate"]

    # The conformal coverage claim, empirically: error among accepted
    # stays within alpha for the learned gate.
    assert learned["accepted_error_rate"] <= learned["alpha"]

    # Reliability payload is chartable: bins with predictions + empirics.
    bins = rep["reliability"]["wrong_now_head"]
    assert bins and all({"mean_pred", "empirical", "n"} <= set(b) for b in bins)


def test_offline_bench_rejects_contaminated_seed_ranges(tmp_path):
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "offline_bench.py"),
         "--seeds", "0:5", "--out", str(tmp_path / "x.json")],
        capture_output=True, text=True, timeout=60)
    assert r.returncode != 0  # eval/calib/training seeds must be refused
