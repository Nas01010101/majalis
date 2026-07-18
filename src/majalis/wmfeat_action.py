"""Action-conditioned feature schema — v1's 12 features (wmfeat.py, UNMODIFIED)
plus one new deployable feature: touch_rate.

touch_rate(key) = (# times this key has already been asked about in the
session so far) / (# questions asked so far). It is a running statistic
available at decision time inside MajalisSession — NOT computed from the
generator's future questions, so it carries no oracle/future-peeking risk.
The caller (MajalisSession for live serving, scripts/gen_action_wm_dataset.py
for mining) owns a TouchTracker and must read touch.rate(key) BEFORE calling
record() for the current question, so the feature never includes the
question it is about to be used to decide.
"""
from __future__ import annotations

from .beliefs import BeliefBoard
from .wmfeat import FEATURES as _V1_FEATURES
from .wmfeat import key_features as _v1_key_features

FEATURES = _V1_FEATURES + ["touch_rate"]


def key_features_action(board: BeliefBoard, key: str, touch_count: int,
                        questions_so_far: int) -> list[float]:
    """v1's 12-feature vector (unchanged) + touch_rate. `touch_count` and
    `questions_so_far` must both be counts observed strictly BEFORE the
    question currently being decided (see TouchTracker.rate/record)."""
    touch_rate = touch_count / questions_so_far if questions_so_far > 0 else 0.0
    return _v1_key_features(board, key) + [touch_rate]


class TouchTracker:
    """Running (per-key touch count, question count) state, owned by the
    caller across one session/stream — deliberately NOT part of BeliefBoard
    itself, since it is a decision-time statistic about the question
    sequence, not board state."""

    def __init__(self) -> None:
        self.touches: dict[str, int] = {}
        self.n_questions: int = 0

    def rate(self, key: str) -> float:
        """touch_rate(key) using only questions seen so far — read this
        BEFORE calling record() for the current question."""
        if self.n_questions == 0:
            return 0.0
        return self.touches.get(key, 0) / self.n_questions

    def record(self, keys: list[str]) -> None:
        """Advance the running stat with the keys THIS question touched
        (e.g. proposal.support_keys) — call once per question, AFTER the
        gate decision has already read rate() for that same question."""
        self.n_questions += 1
        for k in keys:
            self.touches[k] = self.touches.get(k, 0) + 1
