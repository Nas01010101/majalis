"""World-model control layer — the six decision points, conformal where it counts.

1. TRIGGER   value_of_debate(): doubt x sampled disagreement x cost breakeven
2. TARGET    rank_targets(): expected-information-gain ordering of beliefs
3. SPEAKER   (society picks the challenger per target; heterogeneous when unlocked)
4. STOP      committable() + round cap; AnytimeAlarm bounds runaway debates
5. ACCEPT    CalibratedGate (preact-wm): E[error | accepted] <= alpha, split-CRC
6. WRITE-BACK society.run_majalis writes adjudicated corrections into the board

The sampling-frequency law is CSD's honest core (wm-reasoner): the model's
own K-sample distribution over answers is the world model's predictive law;
no naive coverage claims (falsified on small models) — we use frequencies as
risk features and put the guarantee where it is valid: conformal risk
control on the ACCEPT decision, calibrated on a held-out split.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:  # prefer the upstream package when present (dev machines)
    from preact_wm.calibrated_gate import CalibratedGate
except ImportError:  # self-contained everywhere else (deploy, judges)
    from .vendor.calibrated_gate import CalibratedGate

from .beliefs import BeliefBoard
from .bench.tasks import Task
from .handoffs import Proposal
from .llm import Ledger, chat
from .wmnet import load_wm

_GATE_STATE = Path(__file__).resolve().parents[2] / "data" / "gate_calibration.json"
_GATE_STATE_LEARNED = _GATE_STATE.with_name("gate_calibration_learned.json")

# Cost model for the breakeven rule (CBS): a debate costs roughly
# challenge + verdict + re-proposal ~ 3 calls vs the ~0 extra calls of
# committing now. Expressed in "expected accuracy points per 1k tokens".
DEBATE_COST_TOKENS = 3500.0


@dataclass
class GateDecision:
    fire: bool
    p_wrong: float  # predicted probability the current proposal is wrong
    disagreement: float  # 1 - modal answer frequency over K samples
    max_doubt: float
    reason: str
    weak_current: bool = False

    def as_dict(self) -> dict:
        return {"fired": self.fire, "p_wrong": round(self.p_wrong, 3),
                "disagreement": round(self.disagreement, 3),
                "max_doubt": round(self.max_doubt, 3),
                "weak_current": self.weak_current, "reason": self.reason}


# Tuned by the autoresearch loop (experiments/gate-cost/journal.jsonl):
# K=2 + skip-below-0.15 cut cost/question 8% on held-out seeds and 12% on
# eval seeds at unchanged accuracy (80/80). Env-overridable for re-tuning.
_K_SAMPLES = int(os.environ.get("MAJALIS_GATE_K", "2"))
_SKIP_SAMPLER_BELOW_DOUBT = float(os.environ.get("MAJALIS_GATE_SKIP_DOUBT", "0.15"))


def sample_disagreement(task: Task, board: BeliefBoard, ledger: Ledger,
                        model: str, k: int = _K_SAMPLES, seed: int = 0) -> float:
    """CSD-style: K cheap high-temperature answers; disagreement = 1 - modal freq."""
    prompt = (
        f"Question:\n{task.question}\n\n"
        f"Facts:\n{board.summary()}\n\n"
        "Give ONLY the short final answer, nothing else."
    )
    votes = []
    for i in range(k):
        out = chat(model, [{"role": "user", "content": prompt}],
                   ledger=ledger, temperature=1.0, max_tokens=16,
                   seed=seed * 100 + i)
        votes.append(out.strip().lower().rstrip("."))
    if not votes:
        return 1.0
    modal = Counter(votes).most_common(1)[0][1]
    return 1.0 - modal / len(votes)


def risk_score(max_doubt: float, disagreement: float, confidence: float,
               weak_current: bool = False) -> float:
    """Monotone risk feature in [0,1]: P(the committed answer is wrong).

    Simple logistic blend — the FEATURE can be crude because the ACCEPT
    guarantee comes from conformal calibration of the threshold, not from
    the feature being a true probability. weak_current carries a large
    weight: it flags board corruption that sampling-based disagreement is
    structurally blind to (all samples read the same corrupted board).
    """
    z = (2.2 * max_doubt + 2.8 * disagreement + 1.2 * (1.0 - confidence)
         + 3.0 * float(weak_current) - 2.5)
    return 1.0 / (1.0 + math.exp(-z))


class AcceptGate:
    """Calibrated ACCEPT decision: commit without debate only when the
    conformal gate (E[error | accepted] <= alpha) admits the risk score.
    Uncalibrated => fail-safe: everything above a conservative floor debates.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.gate = CalibratedGate()
        self.calibrated = False
        # Learned world model (trained on logged episodes; train/train_wm.py).
        # None when weights are absent or MAJALIS_WM=heuristic — the hand-set
        # blend below remains the fallback and the ablation arm.
        self.wm = load_wm()
        if self.wm and _GATE_STATE_LEARNED.exists():
            try:
                rows = json.loads(_GATE_STATE_LEARNED.read_text())
                # Weak-flagged episodes hard-fire in decide(); exclude them by
                # FLAG (not by score band — learned scores don't share the
                # heuristic's band structure) so they don't shape tau.
                kept = [(r["score"], r["harm"]) for r in rows if not r["weak"]]
                self.gate.fit([s for s, _ in kept], [bool(h) for _, h in kept])
                self.gate.calibrate(self.alpha)
                self.calibrated = True
            except Exception:  # noqa: BLE001 — corrupt state = run uncalibrated
                pass
        elif _GATE_STATE.exists():
            try:
                state = json.loads(_GATE_STATE.read_text())
                # Weak-flagged pairs (the +3.0 term puts them >= ~0.7; the
                # non-weak band tops out ~0.43) hard-fire in decide() and must
                # not shape the threshold that governs the rest.
                kept = [(s, h) for s, h in zip(state["scores"], state["harms"])
                        if s < 0.7]
                self.gate.fit([s for s, _ in kept], [bool(h) for _, h in kept])
                self.gate.calibrate(self.alpha)
                self.calibrated = True
            except Exception:  # noqa: BLE001 — corrupt state = run uncalibrated
                pass

    def decide(self, task: Task, board: BeliefBoard, proposal: Proposal,
               ledger: Ledger, model: str, seed: int = 0) -> GateDecision:
        if self.wm:
            # Learned per-key risk: P(the board's current value is wrong),
            # from the trained wrong_now head (val AUROC 0.999 vs 0.79 for
            # the hand blend; 0.937 on real logged LLM-board episodes).
            support = {k: self.wm.wrong_now(board, k)
                       for k in proposal.support_keys
                       if board.current(k) is not None}
        else:
            support = {k: board.doubt(k) for k in proposal.support_keys
                       if board.current(k) is not None}
        max_doubt = max(support.values(), default=0.0)
        weak = any(board.weak_current(k) for k in proposal.support_keys)
        if self.wm and abs(self.wm.stk_coef[1]) < 1e-9:
            # The stacker LEARNED a zero weight on sampled disagreement (on
            # real logged episodes the wrong_now head subsumes it), so the
            # sampler's K calls cannot change p_wrong — skip them entirely.
            # Measured consequence, not an assumption: gate becomes 0-call.
            disagreement = 0.0
        elif not weak and max_doubt < _SKIP_SAMPLER_BELOW_DOUBT:
            # Low-doubt fast path: the sampler's 3 calls rarely flip a clean
            # board's decision — skip them (autoresearch-tuned threshold).
            disagreement = 0.0
        else:
            disagreement = sample_disagreement(task, board, ledger, model, seed=seed)
        if self.wm:
            # Stacker fit on real logged episodes (leave-one-seed-out AUROC
            # 0.95): maps (head risk, sampled disagreement, weak) -> P(wrong).
            p_wrong = self.wm.commit_risk(max_doubt, disagreement, weak)
        else:
            p_wrong = risk_score(max_doubt, disagreement, proposal.confidence, weak)

        if weak:
            # A weak-source displacement is a KNOWN policy violation, not a
            # probabilistic risk — adjudication is mandatory. (Leaving this to
            # the calibrated threshold fails: ~half of weak cases are harmless
            # by luck, which drags tau above the harmful half.)
            return GateDecision(fire=True, p_wrong=p_wrong,
                                disagreement=disagreement, max_doubt=max_doubt,
                                reason="policy:weak-source", weak_current=True)
        if self.calibrated:
            accept = self.gate.trust(p_wrong)
            reason = f"conformal(alpha={self.alpha})"
        else:
            accept = p_wrong < 0.35  # conservative uncalibrated floor
            reason = "uncalibrated-floor(0.35)"

        # CBS breakeven: even a risky answer is not worth debating when the
        # expected flip value is below the debate's token cost equivalent.
        expected_gain = p_wrong  # accuracy points if the debate flips a wrong answer
        breakeven = DEBATE_COST_TOKENS / 100_000.0  # ~0.035 accuracy-pt equivalent
        if not accept and expected_gain < breakeven:
            accept = True
            reason += "+below-breakeven"

        return GateDecision(fire=not accept, p_wrong=p_wrong,
                            disagreement=disagreement, max_doubt=max_doubt,
                            reason=reason, weak_current=weak)

    @staticmethod
    def save_calibration(scores: list[float], harms: list[int]) -> None:
        _GATE_STATE.parent.mkdir(parents=True, exist_ok=True)
        _GATE_STATE.write_text(json.dumps({"scores": scores, "harms": harms}))

    @staticmethod
    def save_calibration_learned(rows: list[dict]) -> None:
        """rows: [{score, harm, weak}] — learned-score calibration pairs
        (scripts/refit_gate_learned.py rebuilds them offline, no LLM spend)."""
        _GATE_STATE_LEARNED.parent.mkdir(parents=True, exist_ok=True)
        _GATE_STATE_LEARNED.write_text(json.dumps(rows))


def rank_targets(board: BeliefBoard, support_keys: list[str],
                 max_targets: int = 2, wm=None) -> list[str]:
    """TARGET: expected information gain ~ risk-entropy of each supporting
    belief. With binary flip outcomes EIG reduces to the Bernoulli entropy of
    P(wrong) — highest-entropy (most uncertain) beliefs are the most
    informative to challenge; ~0 or ~1 teaches us nothing new. P(wrong) is
    the learned wrong_now head when trained weights exist, else the hand-set
    doubt blend.
    """
    def eig(key: str) -> float:
        d = wm.wrong_now(board, key) if wm else board.doubt(key)
        d = min(0.999, max(0.001, d))
        return -(d * math.log(d) + (1 - d) * math.log(1 - d))

    keys = [k for k in support_keys if board.current(k) is not None]
    return sorted(keys, key=eig, reverse=True)[:max_targets]
