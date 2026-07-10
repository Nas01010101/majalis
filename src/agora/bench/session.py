"""Session benchmark runner — the deployment-shaped eval.

    python -m agora.bench.session --arms single,mad,agora --seeds 0,1,2

Every arm sees the SAME event sequence. Baselines re-read the full stream-
so-far per question (that is the honest no-memory best practice); the agora
arm ingests incrementally and answers from its persistent board.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..config import MODEL_STRONG
from ..llm import Ledger, chat
from ..society import AgoraSession
from .arms import _extract, vanilla_mad
from .stats import fmt_acc
from .stream import make_session
from .tasks import Task, grade

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"

_RULES = ("Think step by step, then give your final answer on the last "
          "line as: ANSWER: <short answer>")


def _replay_single(events, seed: int) -> list[dict]:
    lines: list[str] = []
    records = []
    for ev in events:
        if ev.kind == "evidence":
            lines += ev.lines
            continue
        ledger = Ledger()
        prompt = (f"Evidence:\n" + "\n".join(lines)
                  + f"\n\n{ev.task.question}\n\n{_RULES}")
        out = chat(MODEL_STRONG, [{"role": "user", "content": prompt}],
                   ledger=ledger, temperature=0.0, seed=seed)
        records.append({"task": ev.task, "answer": _extract(out), "ledger": ledger})
    return records


def _replay_mad(events, seed: int) -> list[dict]:
    lines: list[str] = []
    records = []
    for ev in events:
        if ev.kind == "evidence":
            lines += ev.lines
            continue
        task = Task(task_id=ev.task.task_id, family="stream",
                    context="\n".join(lines), question=ev.task.question,
                    gold=ev.task.gold)
        result = vanilla_mad(task, seed=seed)
        records.append({"task": ev.task, "answer": result.answer,
                        "ledger": result.ledger})
    return records


def _replay_agora(events, seed: int) -> list[dict]:
    session = AgoraSession(seed=seed)
    records = []
    for ev in events:
        if ev.kind == "evidence":
            session.ingest(ev.lines)
            continue
        result = session.ask(ev.task)
        records.append({"task": ev.task, "answer": result.answer,
                        "ledger": result.ledger,
                        "gate": result.transcript[0].get("gate")})
    # Perception cost is real: spread it across the session's questions.
    if records:
        records[0]["ingest_ledger"] = session.ingest_ledger
    return records


REPLAYS = {"single": _replay_single, "mad": _replay_mad, "agora": _replay_agora}


def run_session_arm(arm: str, seed: int, n_steps: int = 8) -> dict:
    events = make_session(seed, n_steps=n_steps)
    t0 = time.monotonic()
    records = REPLAYS[arm](events, seed)
    latency = time.monotonic() - t0
    raw_path = RESULTS_DIR / "raw" / f"session_{arm}_s{seed}_t{n_steps}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    correct = tokens = 0
    cost = 0.0
    with raw_path.open("w") as fh:
        for rec in records:
            ok = grade(rec["task"], rec["answer"])
            correct += ok
            tokens += rec["ledger"].total_tokens
            cost += rec["ledger"].cost_usd
            extra = {}
            if "ingest_ledger" in rec:
                tokens += rec["ingest_ledger"].total_tokens
                cost += rec["ingest_ledger"].cost_usd
                extra["ingest"] = rec["ingest_ledger"].as_dict()
            fh.write(json.dumps({
                "task_id": rec["task"].task_id, "gold": rec["task"].gold,
                "answer": rec["answer"], "correct": ok,
                "churned": rec["task"].meta.get("churned"),
                "gate": rec.get("gate"), **rec["ledger"].as_dict(), **extra,
            }) + "\n")
    return {"arm": arm, "seed": seed, "steps": n_steps, "n": len(records),
            "correct": correct, "tokens": tokens, "cost_usd": round(cost, 4),
            "latency_s": round(latency, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="single,agora")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", default="8", help="comma list of stream lengths")
    args = ap.parse_args()

    summaries = []
    for n_steps in (int(t) for t in args.steps.split(",")):
        for seed in (int(s) for s in args.seeds.split(",")):
            for arm in args.arms.split(","):
                print(f"session: {arm} seed={seed} steps={n_steps} ...", flush=True)
                summaries.append(run_session_arm(arm, seed, n_steps))

    out = RESULTS_DIR / "session_summary.json"
    existing = json.loads(out.read_text()) if out.exists() else []
    # Latest run wins per (arm, seed, steps).
    def key(s):  # older records may predate the steps field
        return (s["arm"], s["seed"], s.get("steps", 8))
    merged = {key(s): s for s in existing}
    merged.update({key(s): s for s in summaries})
    out.write_text(json.dumps(list(merged.values()), indent=2))

    by_cell: dict[tuple, dict] = {}
    for s in merged.values():
        agg = by_cell.setdefault((s["arm"], s.get("steps", 8)),
                                 {"n": 0, "correct": 0, "tokens": 0, "cost": 0.0})
        agg["n"] += s["n"]
        agg["correct"] += s["correct"]
        agg["tokens"] += s["tokens"]
        agg["cost"] += s["cost_usd"]
    print(f"\n{'arm':<8} {'steps':>5} {'accuracy (Wilson 95%)':<28} {'tok/q':>8} {'$/q':>9}")
    for (arm, steps), a in sorted(by_cell.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"{arm:<8} {steps:>5} {fmt_acc(a['correct'], a['n']):<28} "
              f"{a['tokens'] // max(1, a['n']):>8} "
              f"{a['cost'] / max(1, a['n']):>9.5f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
