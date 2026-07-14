"""Refit the conformal ACCEPT threshold on LEARNED risk scores — offline.

    python scripts/refit_gate_learned.py

Replays the calibration streams (seeds 100+), scores each stored pair with
the trained world model (head + stacker), and rewrites the learned-gate
calibration state. Real disagreement samples and harm labels come from
data/calibration_pairs.jsonl, so this costs zero LLM calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majalis.beliefs import BeliefBoard, parse_date_ord  # noqa: E402
from majalis.bench.stream import make_session  # noqa: E402
from majalis.wm import AcceptGate  # noqa: E402
from majalis.wmfeat import parse_line  # noqa: E402
from majalis.wmnet import LearnedWM  # noqa: E402


def main() -> None:
    wm = LearnedWM()
    pairs = {}
    for line in (ROOT / "data" / "calibration_pairs.jsonl").read_text().splitlines():
        if line:
            p = json.loads(line)
            if p["family"] == "stream":
                pairs[p["uid"]] = p

    rows = []
    for seed in sorted({int(uid.split(":")[1]) for uid in pairs}):
        board = BeliefBoard()
        for ev in make_session(seed):
            if ev.kind == "evidence":
                for raw in ev.lines:
                    f = parse_line(raw)
                    if f:
                        board.assert_fact(
                            BeliefBoard.make_key(f["entity"], f["attr"]),
                            f["value"], parse_date_ord(f["date"]),
                            source=f["source"])
                continue
            p = pairs.get(f"stream:{seed}:{ev.task.task_id}")
            if p is None:
                continue
            key = BeliefBoard.make_key(ev.task.meta["entity"], ev.task.meta["attr"])
            weak = board.weak_current(key)
            score = wm.commit_risk(wm.wrong_now(board, key),
                                   p["disagreement"], weak)
            rows.append({"score": round(score, 4), "harm": p["harm"],
                         "weak": bool(weak)})

    AcceptGate.save_calibration_learned(rows)
    kept = [r for r in rows if not r["weak"]]
    harm_rate = sum(r["harm"] for r in kept) / max(1, len(kept))
    print(f"saved {len(rows)} learned pairs ({len(rows) - len(kept)} weak, "
          f"hard-fire; non-weak harm rate {harm_rate:.1%})")


if __name__ == "__main__":
    main()
