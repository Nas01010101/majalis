"""Fit the ACCEPT gate on a calibration split.

    python -m majalis.bench.calibrate --n 40 --seed 100

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


from pathlib import Path

_PAIRS = Path(__file__).resolve().parents[3] / "data" / "calibration_pairs.jsonl"


def _done_ids() -> set[str]:
    if not _PAIRS.exists():
        return set()
    return {json.loads(line)["uid"] for line in _PAIRS.read_text().splitlines() if line}


def collect(family: str, n: int, seed: int) -> None:
    """Appends one JSONL pair per task; already-collected (family, seed, id)
    tuples are skipped so a killed run resumes for free."""
    done = _done_ids()
    _PAIRS.parent.mkdir(parents=True, exist_ok=True)
    for task in load_tasks(family, n, seed):
        uid = f"{family}:{seed}:{task.task_id}"
        if uid in done:
            continue
        ledger = Ledger()
        board = BeliefBoard()
        for fact in extract_facts(task.context, ledger, MODEL_FAST):
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
        with _PAIRS.open("a") as fh:
            fh.write(json.dumps({
                "uid": uid, "score": score, "harm": harm,
                # Raw features so the risk blend can be refit offline
                # without re-spending LLM calls.
                "max_doubt": round(max(support.values(), default=0.0), 4),
                "disagreement": round(disagreement, 4),
                "confidence": round(proposal.confidence, 4),
                "family": family,
            }) + "\n")
        print(f"{task.task_id}: score={score:.3f} harm={harm}", flush=True)


def collect_session(seed: int, n_steps: int = 8) -> None:
    """Session-mode pairs: replay a calibration stream with debates OFF and
    label harm = 'the answer committed straight from the board was wrong'.
    This matches the deployed decision exactly (same board state, same
    features, same models)."""
    from ..society import MajalisSession
    from .stream import make_session

    done = _done_ids()
    _PAIRS.parent.mkdir(parents=True, exist_ok=True)
    session = MajalisSession(seed=seed, gate_mode="never")
    for ev in make_session(seed, n_steps=n_steps):
        if ev.kind == "evidence":
            session.ingest(ev.lines)
            continue
        uid = f"stream:{seed}:{ev.task.task_id}"
        if uid in done:
            continue  # note: board state still advanced by prior ingests
        result = session.ask(ev.task)
        gate = result.transcript[0]["gate"]
        harm = 0 if grade(ev.task, result.answer) else 1
        with _PAIRS.open("a") as fh:
            fh.write(json.dumps({
                "uid": uid, "score": gate["p_wrong"], "harm": harm,
                "max_doubt": gate["max_doubt"],
                "disagreement": gate["disagreement"],
                "family": "stream",
            }) + "\n")
        print(f"{ev.task.task_id}: score={gate['p_wrong']:.3f} harm={harm}",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="churn,multihop")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--session-seeds", default="",
                    help="comma list of stream seeds (>=100); replaces families")
    args = ap.parse_args()

    if args.session_seeds:
        for seed in (int(s) for s in args.session_seeds.split(",")):
            assert seed >= 100, "calibration seeds live at 100+; eval uses 0-99"
            collect_session(seed)
    else:
        assert args.seed >= 100, "calibration seeds live at 100+; eval uses 0-99"
        for family in args.families.split(","):
            collect(family, args.n, args.seed)

    pairs = [json.loads(line) for line in _PAIRS.read_text().splitlines() if line]
    scores = [p["score"] for p in pairs]
    harms = [p["harm"] for p in pairs]
    AcceptGate.save_calibration(scores, harms)
    print(f"\nsaved {len(pairs)} calibration pairs "
          f"(base harm rate {sum(harms) / len(pairs):.1%})" if pairs else "no pairs")


if __name__ == "__main__":
    main()
