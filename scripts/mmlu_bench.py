#!/usr/bin/env python3
"""MMLU (multiple-choice) head-to-head: single-agent vs the Majalis gate
adaptation vs vanilla multi-agent debate (MAD).

MMLU is the non-saturated, debate-favorable benchmark the synthetic belief-board
streams lack: its reasoning-heavy subjects leave room for debate to lift accuracy,
so this measures whether the gate captures most of that lift at a fraction of the
MAD cost. Data: data/mmlu_test.jsonl (150 questions, 6 reasoning subjects).

    python scripts/mmlu_bench.py --arms single,majalis-gated,mad --n 150

Writes one row per (arm, question) to results/raw/mmlu_<arm>_s<seed>.jsonl
(resumable — already-answered idx are skipped) and an aggregate to
results/mmlu_results.json with accuracy + Wilson 95% CI + calls/q + $/q per arm.
Spend is real Qwen money: a persistent state file (results/mmlu_spend_state.json)
tracks cumulative spend and --budget-cap-total is a hard stop.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majalis.bench import mmlu  # noqa: E402
from majalis.bench.stats import wilson_ci  # noqa: E402

DATA_PATH = ROOT / "data" / "mmlu_test.jsonl"
RAW_DIR = ROOT / "results" / "raw"
SPEND_STATE_PATH = ROOT / "results" / "mmlu_spend_state.json"
RESULTS_PATH = ROOT / "results" / "mmlu_results.json"


def load_questions(n: int | None) -> list[dict]:
    rows = []
    for line in DATA_PATH.read_text().splitlines():
        if not line:
            continue
        d = json.loads(line)
        rows.append({"question": d["question"], "choices": d["choices"],
                     "gold": d["gold"], "subject": d.get("subject", "")})
    return rows[:n] if n is not None else rows


def _load_spend() -> float:
    if SPEND_STATE_PATH.exists():
        return json.loads(SPEND_STATE_PATH.read_text()).get("cost_usd", 0.0)
    return 0.0


def _save_spend(cost: float) -> None:
    SPEND_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPEND_STATE_PATH.write_text(json.dumps({"cost_usd": round(cost, 6)}))


class BudgetTracker:
    def __init__(self, cap_total: float):
        self.cap_total = cap_total
        self.lock = threading.Lock()
        self.total = _load_spend()
        self.stop = False

    def add(self, cost: float) -> bool:
        with self.lock:
            self.total += cost
            _save_spend(self.total)
            if self.total >= self.cap_total:
                self.stop = True
            return not self.stop

    def would_exceed(self) -> bool:
        with self.lock:
            return self.stop


def _existing_idx(raw_path: Path) -> set[int]:
    if not raw_path.exists():
        return set()
    return {json.loads(l)["idx"] for l in raw_path.read_text().splitlines() if l}


def run_arm(arm: str, questions: list[dict], seed: int, workers: int,
            budget: BudgetTracker, write_lock: threading.Lock, n_target: int) -> Path:
    raw_path = RAW_DIR / f"mmlu_{arm}_s{seed}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_idx(raw_path)
    fn = mmlu.ARMS_MMLU[arm]
    todo = [(i, q) for i, q in enumerate(questions[:n_target]) if i not in done]
    if not todo:
        print(f"  {arm}: all {n_target} already done (resumed)")
        return raw_path

    def _one(idx: int, q: dict) -> dict | None:
        if budget.would_exceed():
            return None
        result = fn(q["question"], q["choices"], seed=seed)
        ok = mmlu.letter_match(result.answer, q["gold"])
        under = budget.add(result.ledger.cost_usd)
        row = {"idx": idx, "subject": q["subject"], "gold": q["gold"],
               "answer": result.answer, "correct": ok, "gate": result.gate,
               **result.ledger.as_dict()}
        with write_lock:
            with raw_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        if not under:
            print(f"  BUDGET CAP HIT (${budget.cap_total}) — stopping new dispatch", flush=True)
        return row

    n_written = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, i, q): i for i, q in todo}
        for fut in as_completed(futs):
            if fut.result() is not None:
                n_written += 1
                if n_written % 25 == 0:
                    print(f"  {arm}: {n_written}/{len(todo)} new "
                          f"(cumulative ${budget.total:.4f})", flush=True)
    print(f"  {arm}: batch done, {n_written} new rows -> {raw_path.name}")
    return raw_path


def aggregate(arms: list[str], seed: int) -> dict:
    out = {}
    for arm in arms:
        raw_path = RAW_DIR / f"mmlu_{arm}_s{seed}.jsonl"
        if not raw_path.exists():
            continue
        rows = [json.loads(l) for l in raw_path.read_text().splitlines() if l]
        n = len(rows)
        if not n:
            continue
        correct = sum(r["correct"] for r in rows)
        lo, hi = wilson_ci(correct, n)
        fired = sum(1 for r in rows if r.get("gate") and r["gate"].get("fired"))
        out[arm] = {
            "n": n, "correct": correct, "accuracy": round(correct / n, 4),
            "wilson_95_ci": [round(lo, 4), round(hi, 4)],
            "mean_calls_per_q": round(sum(r["n_calls"] for r in rows) / n, 3),
            "mean_cost_usd_per_q": round(sum(r["cost_usd"] for r in rows) / n, 6),
            "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
            "gate_fire_rate": round(fired / n, 4) if arm == "majalis-gated" else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="single,majalis-gated,mad")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget-cap-total", type=float, default=8.0,
                    help="hard stop on cumulative MMLU spend across invocations")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    arms = args.arms.split(",")
    for arm in arms:
        if arm not in mmlu.ARMS_MMLU:
            sys.exit(f"unknown arm: {arm} (choose from {list(mmlu.ARMS_MMLU)})")

    if args.aggregate_only:
        agg = aggregate(arms, args.seed)
        RESULTS_PATH.write_text(json.dumps({"arms": agg}, indent=2))
        print(f"wrote {RESULTS_PATH}")
        for arm, a in agg.items():
            print(f"  {arm}: {a['correct']}/{a['n']} = {a['accuracy']:.1%} "
                  f"[{a['wilson_95_ci'][0]:.1%}, {a['wilson_95_ci'][1]:.1%}] "
                  f"calls/q={a['mean_calls_per_q']} $/q={a['mean_cost_usd_per_q']}")
        return

    questions = load_questions(args.n)
    print(f"MMLU: {len(questions)} questions, arms={arms}, cap=${args.budget_cap_total}, "
          f"cumulative so far ${_load_spend():.4f}")
    budget = BudgetTracker(args.budget_cap_total)
    write_lock = threading.Lock()
    for arm in arms:
        if budget.would_exceed():
            print(f"budget cap ${args.budget_cap_total} already reached — skipping {arm}")
            continue
        print(f"[{arm}]")
        run_arm(arm, questions, args.seed, args.workers, budget, write_lock, args.n)

    agg = aggregate(arms, args.seed)
    RESULTS_PATH.write_text(json.dumps({"arms": agg}, indent=2))
    print(f"\nwrote {RESULTS_PATH}")
    for arm, a in agg.items():
        print(f"  {arm}: {a['correct']}/{a['n']} = {a['accuracy']:.1%} "
              f"[{a['wilson_95_ci'][0]:.1%}, {a['wilson_95_ci'][1]:.1%}] "
              f"calls/q={a['mean_calls_per_q']} $/q={a['mean_cost_usd_per_q']}"
              + (f" fire={a['gate_fire_rate']:.1%}" if a['gate_fire_rate'] is not None else ""))


if __name__ == "__main__":
    main()
