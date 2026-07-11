"""Numpy inference for the learned world model (trained in train/train_wm.py).

The deploy surface stays dependency-light: the GPU box trains with torch and
exports plain weights; this module is the entire runtime — a 2-layer trunk,
two heads, and the logistic stacker fit on real logged episodes.
"""
from __future__ import annotations

import json
import math
import os
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
    return 1.0 / (1.0 + math.exp(-z))


_cached: LearnedWM | None = None


def load_wm() -> LearnedWM | None:
    """The learned WM when trained weights exist (and AGORA_WM != 'heuristic');
    None otherwise — callers fall back to the hand-set blend."""
    global _cached
    if os.environ.get("AGORA_WM", "learned") == "heuristic":
        return None
    if _cached is None and _WEIGHTS.exists():
        _cached = LearnedWM()
    return _cached
