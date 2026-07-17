"""PlannedGate — a genuine two-branch predicted-consequence gate.

LeCun's operational claim (design_track3_worldmodel.md §1a): a world model
predicts the CONSEQUENCE of a candidate action, and planning SELECTS by
comparing predicted consequences (argmax/search), not by thresholding a
single state-conditioned risk score. AcceptGate (wm.py) is the latter —
p_wrong is a state classifier, fired past a calibrated/uncalibrated
threshold. PlannedGate is the former: it predicts P(correct | skip) (free,
from v1's existing wrong_now head) and P(correct | debate) (new, learned
head_debate below) and argmax-selects.

    U(skip)   = p_correct_skip
    U(debate) = p_correct_skip + [p_correct_debate - p_correct_skip]
                * (1 + expected_future_touches(key) * gamma) - lambda * cost(debate)
    fire = U(debate) > U(skip)

expected_future_touches(key) is approximated by touch_rate(key) itself: the
running "how often has this key resurfaced so far" rate is the only
deployable (non-oracle) signal available at decision time about how often it
will resurface AGAIN — there is no way to know the remaining stream length
without peeking at the future, so a stationary-rate assumption (this
question's rate holds for future questions too) is the honest, minimal
choice. Flagged explicitly, not silently assumed (§2.1 of the design doc
leaves the exact form of expected_future_touches to the implementation).

Same GateDecision shape as AcceptGate — a drop-in for the existing call
sites (MajalisSession.gate.decide(...)).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .beliefs import BeliefBoard
from .bench.tasks import Task
from .handoffs import Proposal
from .llm import Ledger
from .wm import DEBATE_COST_TOKENS, GateDecision
from .wmfeat_action import FEATURES as ACTION_FEATURES
from .wmfeat_action import TouchTracker, key_features_action
from .wmnet import load_wm

_ACTION_WEIGHTS = Path(__file__).resolve().parents[2] / "data" / "wm_action_weights.json"

GAMMA = 1.0
LAMBDA_COST = 1.0  # matches wm.py's breakeven clause, which uses coefficient 1
                   # on `expected_gain < breakeven` (i.e. an implicit lambda=1).


def target_key(board: BeliefBoard, support_keys: list[str], wm=None) -> str | None:
    """The single most-doubted support key. Mirrors AcceptGate's own
    `max_doubt = max(support.values())` pattern (a single riskiest key drives
    the global p_wrong) rather than rank_targets' EIG/entropy ordering (that
    ranking answers a different question — WHICH keys to spend a debate's
    challenges on once firing is already decided — whereas this answers
    WHETHER firing is worth it at all, the p_correct_debate branch's job).
    scripts/gen_action_wm_dataset.py uses this same helper so the mined
    training rows score exactly the key PlannedGate would score at serve
    time — no train/serve skew in which key gets featurized.
    """
    keys = [k for k in support_keys if board.current(k) is not None]
    if not keys:
        return None

    def risk(k: str) -> float:
        return wm.wrong_now(board, k) if wm else board.doubt(k)

    return max(keys, key=risk)


class ActionWM:
    """P(correct | debate) — the new head, same numpy-inference shape as
    wmnet.LearnedWM (2-layer ReLU trunk + one linear head), trained in
    train/train_action_wm.py, exported to a SEPARATE artifact
    (data/wm_action_weights.json) so v1's wm_weights.json is never touched.
    """

    def __init__(self, path: Path = _ACTION_WEIGHTS):
        w = json.loads(Path(path).read_text())
        self.features = w.get("features", ACTION_FEATURES)
        self.mu = np.array(w["mu"])
        self.sd = np.array(w["sd"])
        self.trunk = [(np.array(W), np.array(b)) for W, b in w["trunk"]]
        self.head_debate = (np.array(w["head_debate"][0]).ravel(),
                            float(np.array(w["head_debate"][1]).ravel()[0]))
        self.metrics = w.get("metrics", {})

    def p_correct_debate(self, x: list[float]) -> float:
        h = (np.array(x) - self.mu) / self.sd
        for W, b in self.trunk:
            h = np.maximum(0.0, W @ h + b)
        z = float(self.head_debate[0] @ h + self.head_debate[1])
        return 1.0 / (1.0 + math.exp(-z))


_cached_action_wm: ActionWM | None = None


def load_action_wm(path: Path = _ACTION_WEIGHTS) -> ActionWM | None:
    """None when trained weights are absent — callers (PlannedGate) must
    degrade gracefully, not crash, exactly like wmnet.load_wm()."""
    global _cached_action_wm
    if _cached_action_wm is None and path.exists():
        _cached_action_wm = ActionWM(path)
    return _cached_action_wm


class PlannedGate:
    """Two-branch argmax-utility gate. Constructor mirrors AcceptGate's
    shape (wm_mode) so MajalisSession's existing gate-construction seam
    (`AcceptGate(wm_mode=wm_mode) if wm_mode is not None else _GATE`) can add
    a `gate_mode == "plan"` branch with no change to any other arm's path.
    """

    def __init__(self, wm_mode: str | None = None,
                action_weights_path: Path = _ACTION_WEIGHTS,
                gamma: float = GAMMA, lambda_cost: float = LAMBDA_COST):
        self.wm = load_wm(wm_mode) if wm_mode is not None else load_wm()
        self.action_wm = load_action_wm(action_weights_path)
        self.gamma = gamma
        self.lambda_cost = lambda_cost
        self.touch = TouchTracker()
        # No conformal claim is made over this argmax decision (§2.3 of the
        # design doc states this explicitly) — `calibrated`/`alpha` exist
        # only so callers that duck-type against AcceptGate don't break.
        self.calibrated = False
        self.alpha = None

    def decide(self, task: Task, board: BeliefBoard, proposal: Proposal,
               ledger: Ledger, model: str, seed: int = 0) -> GateDecision:
        support_keys = [k for k in proposal.support_keys if board.current(k) is not None]
        if self.wm:
            wrong_now = {k: self.wm.wrong_now(board, k) for k in support_keys}
        else:
            wrong_now = {k: board.doubt(k) for k in support_keys}
        max_wrong = max(wrong_now.values(), default=0.0)
        p_correct_skip = 1.0 - max_wrong  # free: 1 - v1's existing wrong_now head
        weak = any(board.weak_current(k) for k in support_keys)

        cost_debate = DEBATE_COST_TOKENS / 100_000.0  # same units as wm.py's breakeven

        key = target_key(board, support_keys, wm=self.wm)
        if key is not None and self.action_wm is not None:
            touch_rate = self.touch.rate(key)
            x = key_features_action(board, key, self.touch.touches.get(key, 0),
                                    self.touch.n_questions)
            p_correct_debate = self.action_wm.p_correct_debate(x)
        elif key is not None:
            # No trained action head yet: degrade to v1's breakeven
            # assumption (P(fix|debate)=1) rather than crash — this is the
            # SAME assumption the paper's self-critique flags as unverified
            # for the reactive gate, made explicit here rather than silently
            # inherited.
            touch_rate = 0.0
            p_correct_debate = 1.0
        else:
            touch_rate = 0.0
            p_correct_debate = p_correct_skip  # nothing to debate; branches tie

        delta = p_correct_debate - p_correct_skip
        u_skip = p_correct_skip
        u_debate = (p_correct_skip + delta * (1.0 + touch_rate * self.gamma)
                    - self.lambda_cost * cost_debate)
        fire = u_debate > u_skip

        reason = f"plan(u_skip={u_skip:.4f},u_debate={u_debate:.4f},touch={touch_rate:.3f})"
        if weak:
            # Same hard-fire safety property as AcceptGate: a known
            # weak-source displacement is a policy violation, not a
            # probabilistic risk the utility comparison should arbitrate.
            fire, reason = True, "policy:weak-source"

        return GateDecision(fire=fire, p_wrong=max_wrong, disagreement=0.0,
                            max_doubt=max_wrong, reason=reason, weak_current=weak)

    def record_question(self, support_keys: list[str]) -> None:
        """Advance the running touch-rate stat AFTER this question's
        decide() call (see TouchTracker docstring for the no-peeking
        ordering requirement)."""
        self.touch.record(support_keys)
