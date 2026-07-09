"""Fit the ACCEPT gate on a calibration split.

    python -m agora.bench.calibrate --n 40 --seed 100

Runs the debate-free pipeline (extract -> propose -> risk features) on
calibration tasks disjoint from the eval seeds, labels harm = "the committed
answer would have been wrong", and saves (score, harm) pairs for
CalibratedGate. Eval seeds stay 0-99; calibration seeds live at 100+ so no
task ever appears in both.
"""
from __future__ import annotations

import argparse
import json

from ..beliefs import BeliefBoard, parse_date_ord
from ..config import MODEL_FAST, MODEL_STRONG
from ..llm import Ledger
from ..society import extract_facts, propose
from ..wm import AcceptGate, risk_score, sample_disagreement
from .tasks import grade, load_tasks


def collect(family: str, n: int, seed: int) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    harms: list[int] = []
    for task in load_tasks(family, n, seed):
        ledger = Ledger()
        board = BeliefBoard()
        for fact in extract_facts(task, ledger, MODEL_FAST):
            try:
                board.assert_fact(
                    BeliefBoard.make_key(str(fact["entity"]), str(fact["attribute"])),
                    str(fact["value"]), parse_date_ord(str(fact.get("date", ""))))
            except (KeyError, TypeError):
                continue
        proposal = propose(task, board, ledger, MODEL_STRONG)
        support = {k: board.doubt(k) for k in proposal.support_keys
                   if board.current(k) is not None}
        disagreement = sample_disagreement(task, board, ledger, MODEL_FAST, seed=seed)
        score = risk_score(max(support.values(), default=0.0),
                           disagreement, proposal.confidence)
        harm = 0 if grade(task, proposal.answer) else 1
        scores.append(score)
        harms.append(harm)
        print(f"{task.task_id}: score={score:.3f} harm={harm}")
    return scores, harms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="churn,multihop")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()
    assert args.seed >= 100, "calibration seeds live at 100+; eval uses 0-99"

    all_scores: list[float] = []
    all_harms: list[int] = []
    for family in args.families.split(","):
        s, h = collect(family, args.n, args.seed)
        all_scores += s
        all_harms += h

    AcceptGate.save_calibration(all_scores, all_harms)
    n = len(all_scores)
    print(f"\nsaved {n} calibration pairs "
          f"(base harm rate {sum(all_harms) / n:.1%})" if n else "no pairs")
    print(json.dumps({"n": n, "harm_rate": sum(all_harms) / max(1, n)}))


if __name__ == "__main__":
    main()
