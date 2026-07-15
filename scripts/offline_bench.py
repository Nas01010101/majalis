"""Offline end-to-end gate benchmark — zero LLM calls, fully deterministic.

    python scripts/offline_bench.py --seeds 5000:5100 --out results/offline_gate_eval.json

Replays held-out streams (never used in eval 0-99, calibration 100-999, or
world-model training 1000-3199), builds the board with the deterministic
extractor, reconstructs ground truth by the arrived-filings rule, and runs
the REAL AcceptGate.decide() for both modes — learned (MAJALIS_WM default) and
heuristic (MAJALIS_WM=heuristic) — on identical inputs. The disagreement
sampler is stubbed to 0.0 for both modes (the learned stacker measured its
weight at zero; the heuristic gets its skip-path value), so the comparison
is LLM-free and apples-to-apples.

Reports per mode: fire rate, poisoned-board recall, false-fire rate, score
AUROC vs board-wrong, and the conformal coverage check E[board wrong |
accepted] vs alpha. Plus 10-bin reliability curves for the dashboard.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import majalis.wm as wmod  # noqa: E402
from majalis.beliefs import BeliefBoard, parse_date_ord  # noqa: E402
from majalis.bench.stream import make_session  # noqa: E402
from majalis.bench.tasks import Task  # noqa: E402
from majalis.handoffs import Proposal  # noqa: E402
from majalis.wmfeat import parse_line, replay_stream  # noqa: E402

# Stub the sampler BEFORE any gate use: offline = no LLM calls, ever.
wmod.sample_disagreement = lambda *a, **kw: 0.0

# The heuristic risk blend needs an author confidence; the live stream
# calibration runs centred on ~0.9 — fixed here so both modes see one input.
_CONFIDENCE = 0.9


def _episodes(seeds: range):
    """Yield (board, key, board_wrong, weak) at every question point."""
    for seed in seeds:
        board = BeliefBoard()
        truth: dict[str, tuple[str, int]] = {}
        for ev in make_session(seed):
            if ev.kind == "evidence":
                for raw in ev.lines:
                    f = parse_line(raw)
                    if not f:
                        continue
                    k = BeliefBoard.make_key(f["entity"], f["attr"])
                    d = parse_date_ord(f["date"])
                    board.assert_fact(k, f["value"], d, source=f["source"])
                    if f["source"] == "Filing" and (k not in truth or d > truth[k][1]):
                        truth[k] = (f["value"].strip().lower(), d)
                continue
            k = BeliefBoard.make_key(ev.task.meta["entity"], ev.task.meta["attr"])
            cur = board.current(k)
            if cur is None or k not in truth:
                continue
            yield (board, k, cur.value != truth[k][0],
                   board.weak_current(k), ev.task)


def _auroc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUROC (Mann-Whitney), no sklearn dependency at runtime."""
    pairs = sorted(zip(scores, labels))
    ranks, i = {}, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        for t in range(i, j):
            ranks[t] = (i + j + 1) / 2  # average rank, 1-based
        i = j
    pos = sum(l for _, l in pairs)
    neg = len(pairs) - pos
    if not pos or not neg:
        return float("nan")
    r_pos = sum(ranks[t] for t, (_, l) in enumerate(pairs) if l)
    return (r_pos - pos * (pos + 1) / 2) / (pos * neg)


def _bins(scores: list[float], labels: list[int], n: int = 10) -> list[dict]:
    out = []
    for b in range(n):
        lo, hi = b / n, (b + 1) / n
        rows = [(s, l) for s, l in zip(scores, labels)
                if lo <= s < hi or (b == n - 1 and s == 1.0)]
        if rows:
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(rows),
                        "mean_pred": round(sum(s for s, _ in rows) / len(rows), 4),
                        "empirical": round(sum(l for _, l in rows) / len(rows), 4)})
    return out


def eval_mode(mode: str, seeds: range) -> tuple[dict, list[float], list[int]]:
    os.environ["MAJALIS_WM"] = mode
    gate = wmod.AcceptGate()
    assert (gate.wm is not None) == (mode == "learned"), f"mode wiring: {mode}"
    scores, labels = [], []
    fired = wrong = caught = false_fire = accepted = accepted_wrong = n = 0
    for board, key, board_wrong, weak, task in _episodes(seeds):
        proposal = Proposal(answer="", rationale="", support_keys=[key],
                            confidence=_CONFIDENCE)
        d = gate.decide(task, board, proposal, ledger=None, model="offline")
        n += 1
        scores.append(d.p_wrong)
        labels.append(int(board_wrong))
        fired += d.fire
        wrong += board_wrong
        caught += d.fire and board_wrong
        false_fire += d.fire and not board_wrong
        accepted += not d.fire
        accepted_wrong += (not d.fire) and board_wrong
    return ({
        "mode": mode, "calibrated": gate.calibrated, "n_questions": n,
        "board_wrong_rate": round(wrong / n, 4),
        "fire_rate": round(fired / n, 4),
        "recall_poisoned": round(caught / max(1, wrong), 4),
        "false_fire_rate": round(false_fire / max(1, n - wrong), 4),
        "score_auroc": round(_auroc(scores, labels), 4),
        # The conformal claim, checked empirically on held-out data:
        # error rate among accepted-without-debate vs alpha.
        "accepted_error_rate": round(accepted_wrong / max(1, accepted), 4),
        "alpha": gate.alpha,
        "coverage_holds": accepted_wrong / max(1, accepted) <= gate.alpha,
    }, scores, labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="5000:5100")
    ap.add_argument("--out", default=str(ROOT / "results" / "offline_gate_eval.json"))
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seeds.split(":"))
    assert lo >= 3200, "offline-bench seeds must avoid eval/calib/WM-training ranges"
    seeds = range(lo, hi)

    learned, l_scores, l_labels = eval_mode("learned", seeds)
    heuristic, h_scores, h_labels = eval_mode("heuristic", seeds)

    # Reliability of the wrong_now head itself on the same streams' step rows.
    os.environ["MAJALIS_WM"] = "learned"
    wm = wmod.load_wm()
    head_scores, head_labels = [], []
    for seed in seeds:
        for row in replay_stream(make_session(seed)):
            # Score straight from the exported net — the feature vector is
            # already built by replay_stream, no board object needed.
            h = (np.array(row["x"]) - wm.mu) / wm.sd
            for W, b in wm.trunk:
                h = np.maximum(0.0, W @ h + b)
            p = 1 / (1 + math.exp(-float(wm.head_wrong[0] @ h + wm.head_wrong[1])))
            head_scores.append(p)
            head_labels.append(row["wrong_now"])

    report = {
        "seeds": args.seeds,
        "n_questions": learned["n_questions"],
        "gate_quality": learned,
        "learned_vs_heuristic": {"learned": learned, "heuristic": heuristic},
        "reliability": {
            "commit_risk": _bins(l_scores, l_labels),
            "wrong_now_head": _bins(head_scores, head_labels),
            "wrong_now_auroc": round(_auroc(head_scores, head_labels), 4),
            "n_step_rows": len(head_scores),
        },
        "notes": [
            "zero LLM calls: sampler stubbed to 0.0 for BOTH modes "
            "(learned stacker measured its weight at zero; heuristic gets "
            "its skip-path value)",
            f"heuristic confidence fixed at {_CONFIDENCE}",
            "labels from the generator's arrived-filings ground truth",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    for m in (learned, heuristic):
        print(f"{m['mode']:>9}: fire {m['fire_rate']:.1%} | recall "
              f"{m['recall_poisoned']:.1%} | false-fire {m['false_fire_rate']:.1%} "
              f"| AUROC {m['score_auroc']:.3f} | accepted-err "
              f"{m['accepted_error_rate']:.1%} (alpha {m['alpha']}) "
              f"| coverage {'OK' if m['coverage_holds'] else 'VIOLATED'}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
