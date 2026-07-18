"""Learned world model — feature schema + offline episode replay.

ONE feature function serves both training and inference: rows are computed
from a live BeliefBoard object, never from a parallel re-implementation, so
the learned heads see the same distribution at train and serve time.

Training labels come from the stream generator's own ground truth (filings
are authoritative; the latest-dated filing that has ARRIVED is the truth,
exactly the rule `bench.stream.make_session` grades gold with), so the
dataset costs zero LLM calls. Two decision-relevant targets (AAWM,
arXiv:2606.09032 — predict what the policy needs, not next-observation):

  wrong_now       — is the board's current value for this key incorrect
                    right now? (replaces the hand-set doubt/conflict blend)
  superseded_next — does an authoritative filing overturn this value within
                    the lookahead window? (replaces the fixed-prior Lomax
                    survival; the learned per-key decay DAGE lists as open)
"""
from __future__ import annotations

import re

from .beliefs import BeliefBoard, parse_date_ord

FEATURES = [
    "age_days",          # now - current value's date
    "exposure_days",     # now - first assertion on this key
    "n_assertions",      # history length
    "n_supersessions",   # observed churn
    "n_conflicts",       # stale echoes / same-date conflicts seen
    "churn_per_month",   # supersessions per 30 exposure-days
    "n_distinct_values", # distinct values ever asserted
    "tier_cur",          # 0 authoritative / 1 weak (current source)
    "weak_current",      # weak source displaced an authoritative value
    "frac_weak_hist",    # share of weak-source assertions in history
    "lomax_p_valid",     # the old closed form, demoted to one feature
    "doubt_heuristic",   # the old hand blend, demoted to one feature
]

# The generator emits template lines; source label = text before the colon.
_LINE_RE = re.compile(
    r"^\[(?P<date>[A-Za-z]{3} \d{4})\] (?P<source>Filing|Blog recap|Industry note|Rumor):"
    r" (?:sources describe )?(?P<entity>.+?)'s (?P<attr>[\w -]+?)"
    r" (?:is now|is|as|remains) (?P<value>.+?)\.$"
)


def parse_line(line: str) -> dict | None:
    m = _LINE_RE.match(line.strip())
    if not m:
        return None
    return {"entity": m["entity"], "attr": m["attr"], "value": m["value"],
            "date": m["date"], "source": m["source"]}


def key_features(board: BeliefBoard, key: str) -> list[float]:
    """The per-key feature vector (order = FEATURES). Inference calls this
    on the live LLM-built board; training calls it on the replayed board."""
    cur = board.current(key)
    if cur is None:
        return [0.0] * len(FEATURES)
    hist = board._history.get(key, [])
    now = board._now_ord
    exposure = max(1.0, float(now - min(b.date_ord for b in hist))) if hist else 1.0
    n_sup = board.n_supersessions(key)
    weak_hist = sum(1 for b in hist if _src_tier(b.source)) / max(1, len(hist))
    return [
        max(0.0, float(now - cur.date_ord)),
        exposure,
        float(len(hist)),
        float(n_sup),
        float(board._conflicts.get(key, 0)),
        30.0 * n_sup / exposure,
        float(len({b.value for b in hist})),
        float(_src_tier(cur.source)),
        float(board.weak_current(key)),
        weak_hist,
        board.p_valid(key),
        board.doubt(key),
    ]


def _src_tier(source: str) -> int:
    s = source.lower()
    return 0 if ("filing" in s or s == "debate") else 1


HAZARD_HORIZONS = (1, 2, 4)  # evidence batches; 2 == the legacy superseded_next window


def replay_stream(events: list, lookahead_steps: int = 2) -> list[dict]:
    """Replay one generated session offline: deterministic extraction builds
    the board; the arrived-filings rule reconstructs ground truth; emit one
    row per (known key, evidence step) with both labels.

    superseded_next looks ahead `lookahead_steps` evidence batches for an
    authoritative filing that changes the key's value. Each row also carries
    the multi-horizon hazard curve `superseded_within` {k: 0/1} for
    HAZARD_HORIZONS — the forward-dynamics targets that let the model be
    ROLLED OUT (board risk k steps ahead), not just queried one step.
    """
    board = BeliefBoard()
    truth: dict[str, tuple[str, int]] = {}  # key -> (value, filing date_ord)
    # Pre-parse per-step filings for the lookahead label.
    step_filings: list[dict[str, tuple[str, int]]] = []
    for ev in events:
        if ev.kind != "evidence":
            continue
        filings: dict[str, tuple[str, int]] = {}
        for line in ev.lines:
            f = parse_line(line)
            if f and f["source"] == "Filing":
                k = BeliefBoard.make_key(f["entity"], f["attr"])
                d = parse_date_ord(f["date"])
                if k not in filings or d > filings[k][1]:
                    filings[k] = (f["value"].strip().lower(), d)
        step_filings.append(filings)

    rows: list[dict] = []
    step = -1
    for ev in events:
        if ev.kind != "evidence":
            continue
        step += 1
        for line in ev.lines:
            f = parse_line(line)
            if f is None:
                continue
            k = BeliefBoard.make_key(f["entity"], f["attr"])
            d = parse_date_ord(f["date"])
            board.assert_fact(k, f["value"], d, source=f["source"])
            if f["source"] == "Filing":
                old = truth.get(k)
                if old is None or d > old[1]:
                    truth[k] = (f["value"].strip().lower(), d)
        for k, (gold_value, _d) in truth.items():
            cur = board.current(k)
            if cur is None:
                continue
            def overturned_within(h: int) -> int:
                return int(any(
                    k in step_filings[s] and step_filings[s][k][0] != cur.value
                    for s in range(step + 1, min(len(step_filings), step + 1 + h))))
            rows.append({
                "step": step,
                "key": k,
                "x": key_features(board, k),
                "wrong_now": int(cur.value != gold_value),
                "superseded_next": overturned_within(lookahead_steps),
                "superseded_within": {h: overturned_within(h) for h in HAZARD_HORIZONS},
            })
    return rows
