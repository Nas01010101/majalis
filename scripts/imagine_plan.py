#!/usr/bin/env python3
"""Planning in imagination — maintenance scheduling under zero-latency serving.

    python scripts/imagine_plan.py --seeds 5000:5100 --budgets 1,2 \
        --out results/imagination_frontier.json

Zero LLM calls. The deployment regime that makes planning matter: questions
must be answered INSTANTLY from the board (no question-time debate — the
latency a live debate adds is exactly what an interactive deployment can't
pay), while repairs (debates) may run only in the maintenance window between
evidence batches, B per step. The policy must therefore predict, before the
questions arrive, which keys will be BOTH wrong AND asked — a forward
prediction problem only a world model can play.

A repair on key k simulates one debate: write the current ground-truth value
back with source="debate" at now+1, mirroring society.py's adjudication
write-back verbatim. Simulating debate-as-repair is licensed by measurement,
not assumption: the mined counterfactuals (results/wm_action_eval.json)
grade a real skeptic->adjudicate->repropose run at P(correct|debate)
~= 0.995-1.0 with zero harmful flips in 592 pairs.

Policies (all deployable — no oracle features except the labelled ceiling):
  none    — no repairs (floor)
  random  — B random known keys per step
  myopic  — top-B by learned p_wrong (v1 wrong_now head; state estimation)
  planned — top-B by p_wrong * (1 - h_1) * (0.1 + touch_rate): expected
            PERSISTENT askable error mass. The rollout term (1 - h_1)
            deprioritizes keys an authoritative filing is about to overwrite
            anyway (repairing those wastes budget — the world fixes them for
            free); touch_rate predicts whether anyone will ask.
  oracle  — top-B among actually-wrong keys (ceiling; uses labels)

Scoring answers the generator's own true/false claims from the board:
"true" iff the board's current value equals the claimed value. Wilson CIs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majalis.beliefs import BeliefBoard, parse_date_ord  # noqa: E402
from majalis.bench.stream import make_session  # noqa: E402
from majalis.wmfeat import parse_line  # noqa: E402
from majalis.wmnet import HazardWM, LearnedWM  # noqa: E402

POLICIES = ("none", "random", "myopic", "planned", "oracle")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def run_policy(policy: str, seed: int, budget: int, wm: LearnedWM,
               hz: HazardWM, rng: np.random.Generator) -> dict:
    board = BeliefBoard()
    truth: dict[str, tuple[str, int]] = {}  # key -> (value, filing date_ord)
    touches: dict[str, int] = {}
    n_q = 0
    correct = repairs = 0
    for ev in make_session(seed):
        if ev.kind == "evidence":
            for raw in ev.lines:
                f = parse_line(raw)
                if f is None:
                    continue
                k = BeliefBoard.make_key(f["entity"], f["attr"])
                d = parse_date_ord(f["date"])
                board.assert_fact(k, f["value"], d, source=f["source"])
                if f["source"] == "Filing":
                    old = truth.get(k)
                    if old is None or d > old[1]:
                        truth[k] = (f["value"].strip().lower(), d)
            # Maintenance window: rank known keys, repair top-B.
            known = [k for k in truth if board.current(k) is not None]
            if policy != "none" and known:
                if policy == "random":
                    order = list(rng.permutation(known))
                elif policy == "myopic":
                    order = sorted(known, key=lambda k: -wm.wrong_now(board, k))
                elif policy == "planned":
                    def ev_mass(k: str) -> float:
                        rate = touches.get(k, 0) / max(1, n_q)
                        return (wm.wrong_now(board, k)
                                * (1.0 - hz.hazard(board, k, 1))
                                * (0.1 + rate))
                    order = sorted(known, key=lambda k: -ev_mass(k))
                elif policy == "oracle":
                    order = [k for k in known
                             if board.current(k).value.strip().lower() != truth[k][0]]
                for k in order[:budget]:
                    cur = board.current(k)
                    gold_value = truth[k][0]
                    repairs += 1
                    if cur.value.strip().lower() != gold_value:
                        # society.py's adjudication write-back, verbatim.
                        board.assert_fact(k, gold_value, board._now_ord + 1,
                                          source="debate")
            continue
        # Question: answered INSTANTLY from the board — no debate allowed.
        meta = ev.task.meta
        k = BeliefBoard.make_key(meta["entity"], meta["attr"])
        touches[k] = touches.get(k, 0) + 1
        n_q += 1
        claim = ev.task.question.split('is currently ')[1].split('."')[0].strip().lower()
        cur = board.current(k)
        pred = "true" if (cur is not None
                          and cur.value.strip().lower() == claim) else "false"
        correct += pred == ev.task.gold
    return {"correct": correct, "n": n_q, "repairs": repairs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="5000:5100")
    ap.add_argument("--budgets", default="1,2")
    ap.add_argument("--out", default="results/imagination_frontier.json")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seeds.split(":"))
    assert lo >= 5000, "imagination eval lives on the held-out 5000+ band"
    budgets = [int(b) for b in args.budgets.split(",")]

    wm, hz = LearnedWM(), HazardWM()
    out: dict = {"seeds": args.seeds, "budgets": budgets, "policies": {}}
    for budget in budgets:
        for pol in POLICIES:
            rng = np.random.default_rng(0)
            c = n = r = 0
            for seed in range(lo, hi):
                res = run_policy(pol, seed, budget, wm, hz, rng)
                c += res["correct"]; n += res["n"]; r += res["repairs"]
            lo_ci, hi_ci = wilson(c, n)
            out["policies"][f"{pol}@B{budget}"] = {
                "acc": round(c / n, 4), "correct": c, "n": n,
                "wilson95": [round(lo_ci, 4), round(hi_ci, 4)],
                "repairs": r}
            print(f"B={budget} {pol:8} {c}/{n} = {c/n:.1%} "
                  f"[{lo_ci:.3f},{hi_ci:.3f}]  repairs={r}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
