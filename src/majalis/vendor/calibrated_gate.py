# Vendored from preact-wm (same author, MIT) — split-conformal risk-controlled
# accept gate. Pure stdlib; vendored so the repo is self-contained for judges
# and deploys without the local editable install.
"""
calibrated_gate.py — the reusable CALIBRATED ABSTENTION GATE (the paper's core system).

One object wraps ANY risk scorer ``s(state, action) -> [0,1]`` with a distribution-free
guarantee on the harm rate among the actions it lets through. It is the *model-independent
safety floor* (Theorem 2): a WRONG world model causes MORE abstention, never more harm.

The gate is the single mechanism shared by every leg of this project:
  - RouterBench cost-routing  — score = P(cheap model wrong); harm = kept-cheap error.
  - tau-bench write gate      — score = P(irreversible write is policy-violating).
  - the synthetic stress test — score = model-predicted trap-prob; harm = realized trap.
Same object, three substrates. That reuse IS the generality claim.

Two calibration modes (the in-expectation vs high-probability distinction this project
learned the hard way — see the conformal-overclaim lesson):

  * ``calibrate(alpha)`` — split-conformal / conformal-risk-control (CRC). Pick the
    LARGEST threshold tau s.t. the empirical harm-rate among kept (score <= tau)
    calibration points is <= alpha. Guarantees ``E[ harm | kept ] <= alpha`` MARGINALLY
    over the exchangeable (calib + test) draw. This holds the MEAN at alpha but NOT every
    individual deployment — per-draw coverage can sit near 50%.

  * ``calibrate_ucb(alpha, delta)`` — high-probability. Target ``alpha - margin`` where
    ``margin = sqrt(log(1/delta) / (2 n_kept))`` is a one-sided Hoeffding bound on the
    kept harm-rate's deviation. Then with probability >= ``1 - delta`` over the calibration
    draw, the realized test harm-rate is <= alpha. The principled fix when you need a floor
    on (almost) every run, paid for in extra abstention.

Stdlib-only. No numpy needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def hoeffding_margin(n: int, delta: float = 0.10) -> float:
    """One-sided Hoeffding margin for a [0,1]-bounded mean over ``n`` samples at level ``delta``.

    If the true kept harm-rate is p and we observe p_hat over n exchangeable kept points,
    Hoeffding gives ``P(p > p_hat + margin) <= delta`` with ``margin = sqrt(ln(1/delta)/2n)``.
    Targeting ``alpha - margin`` on the calibration set therefore makes the realized test
    risk <= alpha with probability >= ``1 - delta`` (a high-probability floor), versus the
    in-expectation-only guarantee of plain CRC.
    """
    return math.sqrt(math.log(1.0 / delta) / (2.0 * max(n, 1)))


def select_tau(scores: list[float], labels: list[bool], alpha: float) -> float:
    """Largest tau s.t. harm-rate among kept (score <= tau) calibration points <= alpha.

    This is the conformal-risk-control threshold for the rule "trust iff score <= tau".
    Scan candidate taus = the observed scores; keep the highest one that still passes, so
    we trust as much as is safely possible. Returns ``-inf`` (trust nothing) if no threshold
    is safe, and the max score (trust all) if everything is safe. Pure stdlib.
    """
    if not scores:
        return float("inf")
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    candidates = sorted({sc for sc, _ in pairs})
    best_tau = float("-inf")
    found = False
    for tau in candidates:
        accepted = [lab for sc, lab in pairs if sc <= tau]
        if not accepted:
            continue
        harm_rate = sum(1 for lab in accepted if lab) / len(accepted)
        if harm_rate <= alpha:
            best_tau = tau
            found = True
    return best_tau if found else float("-inf")


@dataclass
class CalibratedGate:
    """A fitted, model-independent abstention gate over a risk scorer.

    Usage::

        gate = CalibratedGate().fit(cal_scores, cal_harm_labels)
        tau  = gate.calibrate(alpha=0.10)          # in-expectation
        tau  = gate.calibrate_ucb(alpha=0.10)      # high-probability floor
        if gate.trust(risk_score):                 # uses the last-calibrated tau
            execute(action)
        else:
            abstain()                              # escalate / dry-run / ask-human

    ``trust`` is monotone: a lower risk score is never abstained when a higher one is kept.
    The guarantee is on the *kept* set, so abstaining is always the safe fallback — which is
    exactly why a bad scorer degrades into more abstention rather than more harm.
    """

    scores: list[float] = field(default_factory=list)
    labels: list[bool] = field(default_factory=list)
    tau: float = float("-inf")           # current threshold; -inf => trust nothing
    alpha: float | None = None
    mode: str = "uncalibrated"

    def fit(self, scores: list[float], labels: list[bool]) -> "CalibratedGate":
        """Load calibration pairs (risk_score, harmful?). Does not pick a threshold yet."""
        if len(scores) != len(labels):
            raise ValueError("scores and labels must be the same length")
        self.scores = list(map(float, scores))
        self.labels = [bool(x) for x in labels]
        return self

    def calibrate(self, alpha: float) -> float:
        """Split-conformal / CRC threshold (in-expectation E[harm|kept] <= alpha)."""
        self.alpha = float(alpha)
        self.tau = select_tau(self.scores, self.labels, alpha)
        self.mode = "crc"
        return self.tau

    def calibrate_ucb(self, alpha: float, delta: float = 0.10) -> float:
        """High-probability threshold: realized test harm <= alpha w.p. >= 1 - delta.

        Two-pass: first find the CRC tau to learn how many calibration points are kept
        (the n the harm-rate is averaged over), size the Hoeffding margin from that n, then
        re-select at the deflated target ``alpha - margin``. Conservative and self-sizing.
        """
        self.alpha = float(alpha)
        tau0 = select_tau(self.scores, self.labels, alpha)
        n_kept = sum(1 for sc in self.scores if sc <= tau0) or 1
        target = max(alpha - hoeffding_margin(n_kept, delta), 1e-6)
        self.tau = select_tau(self.scores, self.labels, target)
        self.mode = f"ucb(delta={delta})"
        return self.tau

    def trust(self, score: float) -> bool:
        """True => execute the action (risk within the calibrated budget); False => abstain."""
        return float(score) <= self.tau

    # -- evaluation helpers (read-only; used by experiments/tests, never fabricated) --

    @staticmethod
    def kept_harm_rate(scores: list[float], labels: list[bool], tau: float) -> float:
        """Realized harm-rate among kept (score <= tau). 0.0 if nothing is kept."""
        kept = [bool(lab) for sc, lab in zip(scores, labels) if sc <= tau]
        return (sum(kept) / len(kept)) if kept else 0.0

    @staticmethod
    def abstain_rate(scores: list[float], tau: float) -> float:
        """Fraction of actions abstained (score > tau)."""
        if not scores:
            return 0.0
        return sum(1 for sc in scores if sc > tau) / len(scores)
