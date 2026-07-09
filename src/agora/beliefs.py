"""Shared belief board — the society's world-model state.

Keyed, bi-temporal-lite belief store with supersession, stale-echo conflict
tracking, and a closed-form fact-dynamics survival model P(still valid)
(Gamma-Lomax posterior predictive, the same form as tenet-memory's default
dynamics). This in-process board mirrors Tenet's semantics so the real
`tenet-memory` store can back it once Qwen embeddings are unblocked; exact
key lookup is sufficient for the benchmark domains.

Doubt is what the debate layer consumes: doubt(key) rises with observed
churn (supersessions), age, and stale-echo conflicts — exactly the beliefs
worth spending debate tokens on.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Gamma prior over per-key exponential change rate; posterior-predictive
# survival is Lomax: S(dt) = (b / (b + dt))**a. Priors chosen as in Tenet:
# weak pseudo-exposure so observed churn dominates quickly.
_A0 = 0.15
_B0 = 14.0  # days
# The question asks about "now", which sits after the newest dated evidence;
# survival is assessed over this gap so observed churn lowers trust even in a
# freshly-updated value (age 0 would otherwise give S=1 for any churn count).
_HORIZON_DAYS = 30.0

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}


def parse_date_ord(text: str) -> int:
    """'[Mar 2025] ...' or 'Mar 2025' -> comparable day ordinal (approx)."""
    m = re.search(r"([A-Za-z]{3,9})\.?\s+(\d{4})", text)
    if not m:
        return 0
    month = _MONTHS.get(m.group(1).lower()[:3], 1)
    return int(m.group(2)) * 372 + month * 31


@dataclass
class Belief:
    key: str  # "entity::attribute", lowercase
    value: str
    date_ord: int
    source: str
    superseded: bool = False


@dataclass
class BeliefBoard:
    _current: dict[str, Belief] = field(default_factory=dict)
    _history: dict[str, list[Belief]] = field(default_factory=dict)
    _conflicts: dict[str, int] = field(default_factory=dict)  # stale echoes seen
    _now_ord: int = 0

    @staticmethod
    def make_key(entity: str, attr: str) -> str:
        return f"{entity.strip().lower()}::{attr.strip().lower()}"

    def assert_fact(self, key: str, value: str, date_ord: int, source: str = "") -> str:
        """Returns one of: 'new' | 'refresh' | 'superseded' | 'stale-echo' | 'conflict'."""
        self._now_ord = max(self._now_ord, date_ord)
        value_norm = value.strip().lower()
        cur = self._current.get(key)
        belief = Belief(key, value_norm, date_ord, source)
        self._history.setdefault(key, []).append(belief)
        if cur is None:
            self._current[key] = belief
            return "new"
        if value_norm == cur.value:
            cur.date_ord = max(cur.date_ord, date_ord)
            return "refresh"
        if date_ord > cur.date_ord:
            cur.superseded = True
            self._current[key] = belief
            return "superseded"
        if date_ord < cur.date_ord:
            # An older-dated assertion of a different value: a stale echo of a
            # retired value — evidence of churn/noise on this key, raise doubt.
            belief.superseded = True
            self._conflicts[key] = self._conflicts.get(key, 0) + 1
            return "stale-echo"
        # Same date, different value: unresolvable without adjudication.
        self._conflicts[key] = self._conflicts.get(key, 0) + 2
        return "conflict"

    def current(self, key: str) -> Belief | None:
        return self._current.get(key)

    def n_supersessions(self, key: str) -> int:
        return sum(1 for b in self._history.get(key, []) if b.superseded)

    def p_valid(self, key: str) -> float:
        """Lomax survival of the current value, given this key's observed churn."""
        cur = self._current.get(key)
        if cur is None:
            return 0.0
        hist = self._history.get(key, [])
        n_changes = self.n_supersessions(key)
        exposure_days = max(1.0, (self._now_ord - min(b.date_ord for b in hist)) if hist else 1.0)
        a = _A0 + n_changes
        b = _B0 + exposure_days
        age = max(0.0, float(self._now_ord - cur.date_ord)) + _HORIZON_DAYS
        return (b / (b + age)) ** a

    def doubt(self, key: str) -> float:
        """1 - P(valid), inflated by observed stale-echo conflicts on this key."""
        if key not in self._current:
            return 1.0
        base = 1.0 - self.p_valid(key)
        conflict_bump = 1.0 - math.exp(-0.5 * self._conflicts.get(key, 0))
        return min(1.0, base + 0.5 * conflict_bump * (1.0 - base) + 0.15 * conflict_bump)

    def doubts(self, threshold: float = 0.3) -> list[tuple[str, float]]:
        scored = [(k, self.doubt(k)) for k in self._current]
        return sorted([kv for kv in scored if kv[1] >= threshold],
                      key=lambda kv: -kv[1])

    def summary(self, keys: list[str] | None = None, *,
                show_dynamics: bool = False) -> str:
        """Compact textual belief state for prompts.

        show_dynamics stays False for role prompts: exposing p_valid/doubt to
        the reasoner poisons it ("the value is probably stale, so the claim
        must be false") — measured 100%->33% on the churn family, the same
        regression Tenet recorded when doubt touched ranking. Dynamics are
        for the orchestrator's gate, not the debaters.
        """
        lines = []
        for key in (keys or sorted(self._current)):
            cur = self._current.get(key)
            if cur is None:
                continue
            line = f"- {key} = {cur.value}"
            if show_dynamics:
                line += (
                    f" (p_valid={self.p_valid(key):.2f}, doubt={self.doubt(key):.2f}, "
                    f"changes={self.n_supersessions(key)}, echoes={self._conflicts.get(key, 0)})"
                )
            lines.append(line)
        return "\n".join(lines) or "(no beliefs)"
