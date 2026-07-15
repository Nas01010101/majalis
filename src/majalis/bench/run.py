"""Benchmark runner.

    python -m majalis.bench.run --arms single,sc5,mad --families churn,multihop \
        --n 50 --seed 0

Writes one JSONL per (arm, family) under results/raw/ and prints a summary
table with Wilson 95% CIs, total tokens, and latency.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .arms import ARMS
from .stats import fmt_acc
from .tasks import grade, load_tasks

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


def run_arm(arm: str, family: str, n: int, seed: int) -> dict:
    tasks = load_tasks(family, n, seed)
    fn = ARMS[arm]
    raw_path = RESULTS_DIR / "raw" / f"{arm}_{family}_n{n}_s{seed}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    tokens = 0
    latency = 0.0
    cost = 0.0
    with raw_path.open("w") as fh:
        for task in tasks:
            t0 = time.monotonic()
            result = fn(task, seed=seed)
            ok = grade(task, result.answer)
            correct += ok
            tokens += result.ledger.total_tokens
            latency += time.monotonic() - t0
            cost += result.ledger.cost_usd
            gate = next((m.get("gate") for m in result.transcript
                         if isinstance(m, dict) and m.get("gate")), None)
            fh.write(json.dumps({
                "task_id": task.task_id,
                "gold": task.gold,
                "answer": result.answer,
                "correct": ok,
                "gate": gate,
                **result.ledger.as_dict(),
            }) + "\n")
    return {
        "arm": arm, "family": family, "n": n, "seed": seed,
        "correct": correct, "tokens": tokens, "latency_s": round(latency, 1),
        "cost_usd": round(cost, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="single,sc5,mad")
    ap.add_argument("--families", default="churn,multihop")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Import for side effect: registers the "majalis" arm when the society exists.
    try:
        from .. import society  # noqa: F401
    except ImportError:
        pass

    summaries = []
    for family in args.families.split(","):
        for arm in args.arms.split(","):
            if arm not in ARMS:
                print(f"skip unknown arm: {arm}")
                continue
            print(f"running {arm} on {family} (n={args.n}, seed={args.seed}) ...")
            summaries.append(run_arm(arm, family, args.n, args.seed))

    out = RESULTS_DIR / f"summary_s{args.seed}.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"\n{'arm':<14} {'family':<10} {'accuracy (Wilson 95%)':<28} "
          f"{'tokens':>9} {'lat(s)':>7} {'cost($)':>8}")
    for s in summaries:
        print(f"{s['arm']:<14} {s['family']:<10} {fmt_acc(s['correct'], s['n']):<28} "
              f"{s['tokens']:>9} {s['latency_s']:>7} {s['cost_usd']:>8}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
