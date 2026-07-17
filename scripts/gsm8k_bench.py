#!/usr/bin/env python3
"""GSM8K full-test-set head-to-head: single-agent vs the Majalis gate
adaptation (majalis-gated), reusing majalis.bench.gsm8k for both arms.

    .venv/bin/python scripts/gsm8k_bench.py --arms single,majalis-gated --n 1319

Writes one row per (arm, question) to results/raw/gsm8k_<arm>_s<seed>.jsonl
(resumable — already-answered task_ids are skipped) and an aggregate to
results/gsm8k_results.json with accuracy + Wilson 95% CI + mean calls/q +
$/q per arm.

Budget discipline (spend is REAL money):
  - a persistent state file (results/gsm8k_spend_state.json) tracks
    cumulative spend across every invocation of this script, so re-running
    in batches never loses track of total spend;
  - --budget-cap-total is a hard stop: no new question is dispatched once
    cumulative spend would exceed it (checked continuously, not just at
    the 200-question mark);
  - after the first 200 questions per arm, the projected full-run cost is
    printed and compared against --budget-cap-total; if it would exceed
    the cap the run stops there and the partial n is reported (not
    silently extrapolated).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majalis.bench import gsm8k  # noqa: E402
from majalis.bench.stats import wilson_ci  # noqa: E402
from majalis.llm import _get_client  # noqa: E402

DATA_PATH = ROOT / "data" / "gsm8k_test.jsonl"
RAW_DIR = ROOT / "results" / "raw"
SPEND_STATE_PATH = ROOT / "results" / "gsm8k_spend_state.json"
RESULTS_PATH = ROOT / "results" / "gsm8k_results.json"
CHECKPOINT_N = 200


def load_questions(n: int | None) -> list[dict]:
    rows = []
    for line in DATA_PATH.read_text().splitlines():
        if not line:
            continue
        d = json.loads(line)
        rows.append({"question": d["question"], "gold": gsm8k.extract_gold(d["answer"])})
    if n is not None:
        rows = rows[:n]
    return rows


def _load_spend_state() -> float:
    if SPEND_STATE_PATH.exists():
        return json.loads(SPEND_STATE_PATH.read_text()).get("cost_usd", 0.0)
    return 0.0


def _save_spend_state(cost: float) -> None:
    SPEND_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPEND_STATE_PATH.write_text(json.dumps({"cost_usd": round(cost, 6)}))


class BudgetTracker:
    """Thread-safe running total, backed by the persistent state file so
    separate script invocations never lose track of cumulative spend."""

    def __init__(self, cap_total: float):
        self.cap_total = cap_total
        self.lock = threading.Lock()
        self.session_start = _load_spend_state()
        self.total = self.session_start
        self.stop = False

    def add(self, cost: float) -> bool:
        """Returns True if still under budget after adding cost."""
        with self.lock:
            self.total += cost
            _save_spend_state(self.total)
            if self.total >= self.cap_total:
                self.stop = True
            return not self.stop

    def would_exceed(self) -> bool:
        with self.lock:
            return self.stop


def _existing_task_ids(raw_path: Path) -> set[int]:
    if not raw_path.exists():
        return set()
    ids = set()
    for line in raw_path.read_text().splitlines():
        if line:
            ids.add(json.loads(line)["idx"])
    return ids


def run_arm(arm: str, questions: list[dict], seed: int, workers: int,
           budget: BudgetTracker, write_lock: threading.Lock,
           n_target: int) -> Path:
    raw_path = RAW_DIR / f"gsm8k_{arm}_s{seed}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_task_ids(raw_path)
    fn = gsm8k.ARMS_GSM8K[arm]
    todo = [(i, q) for i, q in enumerate(questions[:n_target]) if i not in done]
    if not todo:
        print(f"  {arm}: all {n_target} already done (resumed)")
        return raw_path

    def _one(idx: int, q: dict) -> dict | None:
        if budget.would_exceed():
            return None
        result = fn(q["question"], seed=seed)
        ok = gsm8k.numeric_match(result.answer, q["gold"])
        under_budget = budget.add(result.ledger.cost_usd)
        row = {"idx": idx, "gold": q["gold"], "answer": result.answer,
               "correct": ok, "gate": result.gate, **result.ledger.as_dict()}
        with write_lock:
            with raw_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        if not under_budget:
            print(f"  BUDGET CAP HIT (${budget.cap_total}) — stopping new dispatch", flush=True)
        return row

    n_written = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, i, q): i for i, q in todo}
        for fut in as_completed(futs):
            row = fut.result()
            if row is not None:
                n_written += 1
                if n_written % 50 == 0:
                    print(f"  {arm}: {n_written}/{len(todo)} new "
                          f"(cumulative spend ${budget.total:.4f})", flush=True)
    print(f"  {arm}: batch done, {n_written} new rows written to {raw_path.name}")
    return raw_path


def aggregate(arms: list[str], seed: int) -> dict:
    out = {}
    for arm in arms:
        raw_path = RAW_DIR / f"gsm8k_{arm}_s{seed}.jsonl"
        if not raw_path.exists():
            continue
        rows = [json.loads(l) for l in raw_path.read_text().splitlines() if l]
        n = len(rows)
        correct = sum(r["correct"] for r in rows)
        lo, hi = wilson_ci(correct, n)
        fired = sum(1 for r in rows if r.get("gate") and r["gate"].get("fired"))
        out[arm] = {
            "n": n, "correct": correct, "accuracy": round(correct / n, 4) if n else 0.0,
            "wilson_95_ci": [round(lo, 4), round(hi, 4)],
            "mean_calls_per_q": round(sum(r["n_calls"] for r in rows) / n, 3) if n else 0.0,
            "mean_cost_usd_per_q": round(sum(r["cost_usd"] for r in rows) / n, 6) if n else 0.0,
            "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
            "gate_fire_rate": round(fired / n, 4) if n and arm != "single" else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="single,majalis-gated")
    ap.add_argument("--n", type=int, default=1319)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget-cap-total", type=float, default=12.0,
                    help="hard stop across BOTH parts A and B of the task")
    ap.add_argument("--skip-checkpoint", action="store_true",
                    help="skip the 200-question projection gate (only for reruns "
                         "past an already-approved checkpoint)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="dispatch no new calls; just (re)build results/gsm8k_results.json "
                         "from whatever raw JSONL already exists for --arms (arms may have "
                         "different n — each arm's raw file has its own row count)")
    args = ap.parse_args()

    arms = args.arms.split(",")
    for arm in arms:
        if arm not in gsm8k.ARMS_GSM8K:
            sys.exit(f"unknown arm: {arm}")

    if args.aggregate_only:
        agg = aggregate(arms, args.seed)
        RESULTS_PATH.write_text(json.dumps({"partial": False, "arms": agg}, indent=2))
        print(f"wrote {RESULTS_PATH}")
        for arm, a in agg.items():
            print(f"  {arm}: {a['correct']}/{a['n']} = {a['accuracy']:.1%} "
                  f"[{a['wilson_95_ci'][0]:.1%}, {a['wilson_95_ci'][1]:.1%}] "
                  f"calls/q={a['mean_calls_per_q']} $/q={a['mean_cost_usd_per_q']}")
        return

    questions = load_questions(None)
    print(f"loaded {len(questions)} GSM8K test questions from {DATA_PATH}")
    if len(questions) != 1319:
        print(f"WARNING: expected 1319 rows, got {len(questions)}")

    _get_client()  # pre-warm the singleton before any threads touch it
    budget = BudgetTracker(args.budget_cap_total)
    write_lock = threading.Lock()
    print(f"cumulative spend so far (all runs): ${budget.session_start:.4f} "
          f"/ cap ${args.budget_cap_total}")

    n_target = min(args.n, len(questions))
    checkpoint_n = min(CHECKPOINT_N, n_target)

    # Phase 1: checkpoint batch for every arm.
    if not args.skip_checkpoint:
        t0 = time.monotonic()
        for arm in arms:
            print(f"[checkpoint] running {arm} on first {checkpoint_n} ...")
            run_arm(arm, questions, args.seed, args.workers, budget, write_lock, checkpoint_n)
            if budget.would_exceed():
                break
        elapsed = time.monotonic() - t0
        agg = aggregate(arms, args.seed)
        checkpoint_cost = sum(a["total_cost_usd"] for a in agg.values())
        checkpoint_n_done = sum(a["n"] for a in agg.values())
        projected_full = (checkpoint_cost / checkpoint_n_done * n_target * len(arms)
                          if checkpoint_n_done else float("inf"))
        print(f"\n[checkpoint] {checkpoint_n_done} rows in {elapsed:.0f}s, "
              f"${checkpoint_cost:.4f} spent so far this run")
        print(f"[checkpoint] projected full-run cost ({n_target} q x {len(arms)} arms): "
              f"${projected_full:.2f}")
        if budget.would_exceed() or projected_full > args.budget_cap_total:
            print(f"STOPPING: projected cost exceeds cap (${args.budget_cap_total}). "
                  f"Keeping partial results at n={checkpoint_n} per arm.")
            RESULTS_PATH.write_text(json.dumps(
                {"partial": True, "n_per_arm": checkpoint_n, "arms": agg}, indent=2))
            print(f"wrote {RESULTS_PATH} (partial)")
            return

    # Phase 2: run to completion.
    for arm in arms:
        if budget.would_exceed():
            break
        print(f"running {arm} to n={n_target} ...")
        run_arm(arm, questions, args.seed, args.workers, budget, write_lock, n_target)

    agg = aggregate(arms, args.seed)
    RESULTS_PATH.write_text(json.dumps(
        {"partial": budget.would_exceed(), "n_per_arm": n_target, "arms": agg}, indent=2))
    print(f"\nwrote {RESULTS_PATH}")
    for arm, a in agg.items():
        print(f"  {arm}: {a['correct']}/{a['n']} = {a['accuracy']:.1%} "
              f"[{a['wilson_95_ci'][0]:.1%}, {a['wilson_95_ci'][1]:.1%}] "
              f"calls/q={a['mean_calls_per_q']} $/q={a['mean_cost_usd_per_q']}")
    print(f"\ntotal spend (all runs of this script): ${budget.total:.4f}")


if __name__ == "__main__":
    main()
