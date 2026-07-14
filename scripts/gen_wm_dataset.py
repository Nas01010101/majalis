"""Generate the learned-world-model training set — zero LLM calls.

    python scripts/gen_wm_dataset.py --train-seeds 1000:3000 --val-seeds 3000:3200

Per-key dynamics rows come from offline stream replays (labels from the
generator's arrived-filings ground truth). Question-level stacker rows are
rebuilt from data/calibration_pairs.jsonl (real LLM disagreement + harm
labels, seeds 100+) by replaying the same streams deterministically —
no re-spend. Seed ranges: eval 0-99 (untouchable), calibration 100-999,
world-model training 1000+.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majalis.beliefs import BeliefBoard, parse_date_ord  # noqa: E402
from majalis.bench.stream import make_session  # noqa: E402
from majalis.wmfeat import key_features, parse_line, replay_stream  # noqa: E402

OUT = ROOT / "data"


def gen_dynamics(seeds: range) -> dict[str, np.ndarray]:
    X, y_wrong, y_sup, groups = [], [], [], []
    for seed in seeds:
        for row in replay_stream(make_session(seed)):
            X.append(row["x"])
            y_wrong.append(row["wrong_now"])
            y_sup.append(row["superseded_next"])
            groups.append(seed)
    return {"X": np.array(X, dtype=np.float32),
            "y_wrong": np.array(y_wrong, dtype=np.int8),
            "y_sup": np.array(y_sup, dtype=np.int8),
            "seed": np.array(groups, dtype=np.int32)}


def gen_stacker() -> dict[str, np.ndarray]:
    """Join stored calibration pairs (real LLM runs) with offline-replayed
    board features at the same question points."""
    pairs_path = OUT / "calibration_pairs.jsonl"
    pairs = {}
    for line in pairs_path.read_text().splitlines():
        if line:
            p = json.loads(line)
            if p["family"] == "stream":
                pairs[p["uid"]] = p
    seeds = sorted({int(uid.split(":")[1]) for uid in pairs})
    X, extra, harms, groups = [], [], [], []
    matched = 0
    for seed in seeds:
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
            uid = f"stream:{seed}:{ev.task.task_id}"
            p = pairs.get(uid)
            if p is None:
                continue
            key = BeliefBoard.make_key(ev.task.meta["entity"], ev.task.meta["attr"])
            X.append(key_features(board, key))
            extra.append([p["disagreement"], float(board.weak_current(key))])
            harms.append(p["harm"])
            groups.append(seed)
            matched += 1
    print(f"stacker: matched {matched}/{len(pairs)} stream calibration pairs "
          f"across seeds {seeds}")
    return {"X": np.array(X, dtype=np.float32),
            "extra": np.array(extra, dtype=np.float32),
            "harm": np.array(harms, dtype=np.int8),
            "seed": np.array(groups, dtype=np.int32)}


def _parse_range(s: str) -> range:
    lo, hi = s.split(":")
    return range(int(lo), int(hi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", default="1000:3000")
    ap.add_argument("--val-seeds", default="3000:3200")
    args = ap.parse_args()
    assert _parse_range(args.train_seeds).start >= 1000, "WM seeds live at 1000+"

    train = gen_dynamics(_parse_range(args.train_seeds))
    val = gen_dynamics(_parse_range(args.val_seeds))
    stacker = gen_stacker()
    np.savez_compressed(OUT / "wm_dynamics.npz",
                        **{f"train_{k}": v for k, v in train.items()},
                        **{f"val_{k}": v for k, v in val.items()})
    np.savez_compressed(OUT / "wm_stacker.npz", **stacker)
    for name, d in [("train", train), ("val", val)]:
        print(f"{name}: {len(d['X'])} rows | wrong_now {d['y_wrong'].mean():.3f} "
              f"| superseded_next {d['y_sup'].mean():.3f}")


if __name__ == "__main__":
    main()
