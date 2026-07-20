"""Numpy inference for the learned world model (trained in train/train_wm.py).

The deploy surface stays dependency-light: the GPU box trains with torch and
exports plain weights; this module is the entire runtime — a 2-layer trunk,
two heads, and the logistic stacker fit on real logged episodes.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from .beliefs import BeliefBoard
from .wmfeat import key_features

_WEIGHTS = Path(__file__).resolve().parents[2] / "data" / "wm_weights.json"


class LearnedWM:
    """wrong_now(board, key)  -> P(the board's current value is incorrect)
    superseded_next(board, key) -> P(an authoritative filing overturns it soon)
    commit_risk(p_wrong, disagreement, weak) -> P(committed answer wrong)"""

    def __init__(self, path: Path = _WEIGHTS):
        w = json.loads(Path(path).read_text())
        self.mu = np.array(w["mu"])
        self.sd = np.array(w["sd"])
        self.trunk = [(np.array(W), np.array(b)) for W, b in w["trunk"]]
        # torch Linear(64,1) exports weight (1,64), bias (1,) — flatten for
        # scalar heads.
        self.head_wrong = (np.array(w["head_wrong"][0]).ravel(),
                           float(np.array(w["head_wrong"][1]).ravel()[0]))
        self.head_sup = (np.array(w["head_sup"][0]).ravel(),
                         float(np.array(w["head_sup"][1]).ravel()[0]))
        self.stk_coef = np.array(w["stacker"]["coef"])
        self.stk_b = float(w["stacker"]["intercept"])
        self.metrics = w.get("metrics", {})

    def _heads(self, board: BeliefBoard, key: str) -> tuple[float, float]:
        h = (np.array(key_features(board, key)) - self.mu) / self.sd
        for W, b in self.trunk:
            h = np.maximum(0.0, W @ h + b)
        lw = float(self.head_wrong[0] @ h + self.head_wrong[1])
        ls = float(self.head_sup[0] @ h + self.head_sup[1])
        return _sigmoid(lw), _sigmoid(ls)

    def wrong_now(self, board: BeliefBoard, key: str) -> float:
        if board.current(key) is None:
            return 1.0
        return self._heads(board, key)[0]

    def superseded_next(self, board: BeliefBoard, key: str) -> float:
        if board.current(key) is None:
            return 1.0
        return self._heads(board, key)[1]

    def commit_risk(self, p_wrong: float, disagreement: float, weak: bool) -> float:
        x = np.array([p_wrong, disagreement, float(weak)])
        return _sigmoid(float(self.stk_coef @ x + self.stk_b))


def _sigmoid(z: float) -> float:
    # Numerically stable: the naive 1/(1+exp(-z)) overflows for z << 0
    # (math.exp(-z) blows past the float range around z < -709, raising
    # OverflowError). A large-magnitude logit off a real board hit exactly that
    # in production. Split by sign so the exponent argument is always <= 0.
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


_cached: LearnedWM | None = None


def load_wm(mode: str | None = None) -> LearnedWM | None:
    """The learned WM when trained weights exist and the effective mode is
    "learned"; None ("heuristic") otherwise — callers fall back to the
    hand-set blend.

    `mode` is the explicit, code-determined choice (e.g. the session bench
    harness passes "heuristic" for the majalis arm and "learned" for
    majalis-wm — see bench/session.py). When `mode` is None, the caller has
    no arm-level opinion and the legacy MAJALIS_WM env var (default
    "learned") decides, as before.

    The MAJALIS_WM env var ALWAYS wins when set, even over an explicit
    `mode` — this is the escape hatch for ad-hoc re-tuning — but overriding
    an arm's implied mode silently would make shipped numbers
    unreproducible, so it logs loudly.
    """
    global _cached
    env = os.environ.get("MAJALIS_WM")
    effective = mode if mode is not None else "learned"
    if env is not None:
        if mode is not None and env != mode:
            print(f"MAJALIS_WM={env!r} env var OVERRIDES the arm-implied "
                  f"gate mode {mode!r} — results reproduce the env "
                  f"override, not the arm default. Unset MAJALIS_WM to "
                  f"reproduce the shipped numbers.", file=sys.stderr)
        effective = env
    if effective == "heuristic":
        return None
    if _cached is None and _WEIGHTS.exists():
        _cached = LearnedWM()
    return _cached


_HAZARD_WEIGHTS = Path(__file__).resolve().parents[2] / "data" / "wm_hazard_weights.json"


class HazardWM:
    """Numpy inference for the multi-horizon hazard heads
    (train/train_wm_hazard.py): hazard(board, key, k) = P(an authoritative
    filing overturns this key's current value within k evidence batches).
    The monotone curve h_1 <= h_2 <= h_4 is the world model's rollout
    surface — consumed by scripts/imagine_plan.py (policy audition in
    imagination) and available to serve-time schedulers."""

    def __init__(self, path: Path = _HAZARD_WEIGHTS):
        w = json.loads(Path(path).read_text())
        self.horizons = tuple(w["horizons"])
        self.mu, self.sd = np.array(w["mu"]), np.array(w["sd"])
        self.trunk = [(np.array(W), np.array(b)) for W, b in w["trunk"]]
        self.heads = {int(h): (np.array(Wb[0]).ravel(),
                                float(np.array(Wb[1]).ravel()[0]))
                      for h, Wb in w["heads"].items()}
        self.metrics = w.get("metrics", {})

    def hazard(self, board: BeliefBoard, key: str, k: int) -> float:
        if board.current(key) is None:
            return 1.0
        h = (np.array(key_features(board, key)) - self.mu) / self.sd
        for W, b in self.trunk:
            h = np.maximum(0.0, W @ h + b)
        Wh, bh = self.heads[k]
        return _sigmoid(float(Wh @ h) + bh)


_cached_hazard: HazardWM | None = None


def load_hazard_wm(path: Path = _HAZARD_WEIGHTS) -> HazardWM | None:
    """None when trained hazard weights are absent — callers degrade
    gracefully, same contract as load_wm()/load_action_wm()."""
    global _cached_hazard
    if _cached_hazard is None and path.exists():
        _cached_hazard = HazardWM(path)
    return _cached_hazard
