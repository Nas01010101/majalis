"""World-model control layer — the six decision points, conformal where it counts.

1. TRIGGER   value_of_debate(): doubt x sampled disagreement x cost breakeven
2. TARGET    rank_targets(): expected-information-gain ordering of beliefs
3. SPEAKER   (society picks the challenger per target; heterogeneous when unlocked)
4. STOP      committable() + round cap; AnytimeAlarm bounds runaway debates
5. ACCEPT    CalibratedGate (preact-wm): E[error | accepted] <= alpha, split-CRC
6. WRITE-BACK society.run_agora writes adjudicated corrections into the board

The sampling-frequency law is CSD's honest core (wm-reasoner): the model's
own K-sample distribution over answers is the world model's predictive law;
no naive coverage claims (falsified on small models) — we use frequencies as
risk features and put the guarantee where it is valid: conformal risk
control on the ACCEPT decision, calibrated on a held-out split.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from preact_wm.calibrated_gate import CalibratedGate

from .beliefs import BeliefBoard
from .bench.tasks import Task
from .handoffs import Proposal
from .llm import Ledger, chat

_GATE_STATE = Path(__file__).resolve().parents[2] / "data" / "gate_calibration.json"

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

    def as_dict(self) -> dict:
        return {"fired": self.fire, "p_wrong": round(self.p_wrong, 3),
                "disagreement": round(self.disagreement, 3),
                "max_doubt": round(self.max_doubt, 3), "reason": self.reason}


def sample_disagreement(task: Task, board: BeliefBoard, ledger: Ledger,
                        model: str, k: int = 3, seed: int = 0) -> float:
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


def risk_score(max_doubt: float, disagreement: float, confidence: float) -> float:
    """Monotone risk feature in [0,1]: P(the committed answer is wrong).

    Simple logistic blend — the FEATURE can be crude because the ACCEPT
    guarantee comes from conformal calibration of the threshold, not from
    the feature being a true probability.
    """
    z = 2.2 * max_doubt + 2.8 * disagreement + 1.2 * (1.0 - confidence) - 2.5
    return 1.0 / (1.0 + math.exp(-z))


class AcceptGate:
    """Calibrated ACCEPT decision: commit without debate only when the
    conformal gate (E[error | accepted] <= alpha) admits the risk score.
    Uncalibrated => fail-safe: everything above a conservative floor debates.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.gate = CalibratedGate()
        self.calibrated = False
        if _GATE_STATE.exists():
            try:
                state = json.loads(_GATE_STATE.read_text())
                self.gate.fit(state["scores"], [bool(h) for h in state["harms"]])
                self.gate.calibrate(self.alpha)
                self.calibrated = True
            except Exception:  # noqa: BLE001 — corrupt state = run uncalibrated
                pass

    def decide(self, task: Task, board: BeliefBoard, proposal: Proposal,
               ledger: Ledger, model: str, seed: int = 0) -> GateDecision:
        support = {k: board.doubt(k) for k in proposal.support_keys
                   if board.current(k) is not None}
        max_doubt = max(support.values(), default=0.0)
        disagreement = sample_disagreement(task, board, ledger, model, seed=seed)
        p_wrong = risk_score(max_doubt, disagreement, proposal.confidence)

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
                            reason=reason)

    @staticmethod
    def save_calibration(scores: list[float], harms: list[int]) -> None:
        _GATE_STATE.parent.mkdir(parents=True, exist_ok=True)
        _GATE_STATE.write_text(json.dumps({"scores": scores, "harms": harms}))


def rank_targets(board: BeliefBoard, support_keys: list[str],
                 max_targets: int = 2) -> list[str]:
    """TARGET: expected information gain ~ doubt-entropy of each supporting
    belief. With binary flip outcomes EIG reduces to the Bernoulli entropy of
    doubt — highest-entropy (most uncertain) beliefs are the most informative
    to challenge; a belief with doubt ~0 or ~1 teaches us nothing new.
    """
    def eig(key: str) -> float:
        d = min(0.999, max(0.001, board.doubt(key)))
        return -(d * math.log(d) + (1 - d) * math.log(1 - d))

    keys = [k for k in support_keys if board.current(k) is not None]
    return sorted(keys, key=eig, reverse=True)[:max_targets]
