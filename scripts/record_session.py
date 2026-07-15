"""Record a full society session for the live viewer — one real run.

    python scripts/record_session.py --seed 0 --steps 8 --out results/replay_s0.json

Runs the real society (Qwen backbones, learned gate) over a generated
evidence stream and captures EVERYTHING the viewer replays: arriving
evidence lines, per-fact assert outcomes, a world-model snapshot of every
board key after each batch (offline numpy — free), and each question's
proposal / gate decision / skeptic attack / judge verdict / re-proposal,
with per-question cost. Resumable-safe: skips if the output already exists
(delete the file to re-record).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majalis.bench.stream import make_session  # noqa: E402
from majalis.bench.tasks import grade  # noqa: E402
from majalis.society import MajalisSession  # noqa: E402
from majalis.wmnet import load_wm  # noqa: E402


def board_snapshot(session: MajalisSession, wm) -> list[dict]:
    board = session.board
    snap = []
    for key in sorted(board._current):
        cur = board.current(key)
        snap.append({
            "key": key, "value": cur.value, "source": cur.source,
            "wrong_now": round(wm.wrong_now(board, key), 3),
            "superseded_next": round(wm.superseded_next(board, key), 3),
            "weak": board.weak_current(key),
            "churn": board.n_supersessions(key),
            "conflicts": board._conflicts.get(key, 0),
        })
    return snap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "replay_s0.json"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"{out} exists — delete it to re-record (each run spends ~$0.10)")
        return

    wm = load_wm()
    assert wm is not None, "learned weights required (data/wm_weights.json)"
    session = MajalisSession(seed=args.seed)
    events: list[dict] = []
    correct = questions = fired = 0

    for ev in make_session(args.seed, n_steps=args.steps):
        if ev.kind == "evidence":
            asserts: list[dict] = []
            cost0 = session.ingest_ledger.cost_usd
            session.ingest(ev.lines, trace=asserts)
            events.append({
                "type": "evidence", "lines": ev.lines, "asserts": asserts,
                "cost_usd": round(session.ingest_ledger.cost_usd - cost0, 5),
                "board": board_snapshot(session, wm),
            })
            continue
        result = session.ask(ev.task)
        trace = result.transcript[0]
        ok = bool(grade(ev.task, result.answer))
        questions += 1
        correct += ok
        fired += bool(trace["gate"]["fired"])
        events.append({
            "type": "question", "task_id": ev.task.task_id,
            "question": ev.task.question, "gold": ev.task.gold,
            "answer": result.answer, "correct": ok,
            "gate": trace["gate"], "events": trace["events"],
            "cost_usd": round(result.ledger.cost_usd, 5),
            "tokens": result.ledger.total_tokens,
            "board": board_snapshot(session, wm),
        })
        print(f"{ev.task.task_id}: {'✓' if ok else '✗'} "
              f"gate={'DEBATE' if trace['gate']['fired'] else 'commit'}", flush=True)

    replay = {
        "seed": args.seed, "steps": args.steps,
        "summary": {
            "questions": questions, "correct": correct, "gate_fired": fired,
            "ingest_cost_usd": round(session.ingest_ledger.cost_usd, 4),
            "total_cost_usd": round(
                session.ingest_ledger.cost_usd
                + sum(e.get("cost_usd", 0) for e in events if e["type"] == "question"), 4),
        },
        "events": events,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(replay, indent=1))
    print(f"\nrecorded {len(events)} events -> {out} "
          f"({correct}/{questions} correct, {fired} debates, "
          f"${replay['summary']['total_cost_usd']})")


if __name__ == "__main__":
    main()
