"""Unit tests for PlannedGate (src/majalis/wm_plan.py) — stubbed heads, no
live LLM calls. Covers the argmax logic, the cost term, and touch_rate
accounting (the three things the task explicitly calls out).
"""
from __future__ import annotations

from majalis.beliefs import BeliefBoard, parse_date_ord
from majalis.bench.tasks import Task
from majalis.handoffs import Proposal
from majalis.llm import Ledger
from majalis.wm_plan import PlannedGate, target_key
from majalis.wmfeat_action import TouchTracker


class _FakeWM:
    """Stands in for wmnet.LearnedWM: a fixed wrong_now() for every key."""

    def __init__(self, wrong_now_value: float):
        self._v = wrong_now_value

    def wrong_now(self, board, key):
        return self._v


class _FakeActionWM:
    """Stands in for wm_plan.ActionWM: a fixed p_correct_debate()."""

    def __init__(self, p: float):
        self._p = p

    def p_correct_debate(self, x):
        return self._p


def _board_with_key(key: str = "e::ceo") -> BeliefBoard:
    b = BeliefBoard()
    b.assert_fact(key, "chen", parse_date_ord("Jan 2025"), source="Filing")
    return b


def _proposal(support_keys: list[str]) -> Proposal:
    return Proposal(answer="chen", rationale="r", support_keys=support_keys,
                    confidence=0.8, author="proposer")


def _task() -> Task:
    return Task(task_id="t0", family="test", context="", question="q?", gold="chen")


def _bare_gate(wm=None, action_wm=None, gamma: float = 1.0,
              lambda_cost: float = 1.0) -> PlannedGate:
    """PlannedGate with stubbed heads, bypassing __init__'s file loads."""
    gate = PlannedGate.__new__(PlannedGate)
    gate.wm = wm
    gate.action_wm = action_wm
    gate.gamma = gamma
    gate.lambda_cost = lambda_cost
    gate.touch = TouchTracker()
    gate.calibrated = False
    gate.alpha = None
    return gate


# --- argmax logic -------------------------------------------------------------

def test_planned_gate_fires_when_debate_utility_dominates():
    """Risky board + debate almost always fixes it + key resurfaces a lot ->
    U(debate) clears U(skip) by a wide margin."""
    gate = _bare_gate(wm=_FakeWM(0.6), action_wm=_FakeActionWM(0.98))
    gate.touch.n_questions = 4
    gate.touch.touches = {"e::ceo": 3}  # touch_rate = 0.75
    decision = gate.decide(_task(), _board_with_key(), _proposal(["e::ceo"]), Ledger(), "m")
    assert decision.fire is True


def test_planned_gate_skips_when_debate_adds_nothing():
    """delta == 0 (debate doesn't change the expected outcome) -> U(debate)
    is strictly U(skip) minus the token cost -> fire False. Isolates the
    cost term in isolation from any expected-gain term."""
    gate = _bare_gate(wm=_FakeWM(0.2), action_wm=_FakeActionWM(0.8))  # p_correct_skip == p_correct_debate
    decision = gate.decide(_task(), _board_with_key(), _proposal(["e::ceo"]), Ledger(), "m")
    assert decision.fire is False


def test_planned_gate_no_support_keys_ties_and_skips():
    gate = _bare_gate(wm=_FakeWM(0.5), action_wm=_FakeActionWM(0.9))
    decision = gate.decide(_task(), BeliefBoard(), _proposal([]), Ledger(), "m")
    assert decision.fire is False
    assert decision.max_doubt == 0.0


def test_planned_gate_weak_current_hard_fires_regardless_of_utility():
    """A weak-source displacement is a policy violation (same as
    AcceptGate's hard-fire), not something the utility comparison should
    arbitrate — even when the utility math alone would say skip."""
    gate = _bare_gate(wm=None, action_wm=_FakeActionWM(0.1))  # utility math would say "skip"
    board = BeliefBoard()
    board.assert_fact("e::ceo", "chen", parse_date_ord("Jan 2025"), source="Filing")
    board.assert_fact("e::ceo", "okafor", parse_date_ord("May 2025"), source="Rumor")
    assert board.weak_current("e::ceo")
    decision = gate.decide(_task(), board, _proposal(["e::ceo"]), Ledger(), "m")
    assert decision.fire is True
    assert decision.reason == "policy:weak-source"


def test_planned_gate_degrades_gracefully_without_trained_action_head():
    """action_wm=None (no data/wm_action_weights.json yet): falls back to
    v1's breakeven assumption P(fix|debate)=1 instead of crashing — the same
    unverified assumption the paper's self-critique flags for the reactive
    gate, made explicit here rather than silently inherited."""
    gate = _bare_gate(wm=_FakeWM(0.5), action_wm=None)
    decision = gate.decide(_task(), _board_with_key(), _proposal(["e::ceo"]), Ledger(), "m")
    assert decision.fire is True  # p_correct_debate assumed 1.0 > p_correct_skip 0.5


# --- cost term + touch_rate multi-step term -----------------------------------

def test_planned_gate_touch_rate_tips_a_marginal_decision():
    """Same small positive delta: at touch_rate=0 the expected one-shot gain
    doesn't clear the debate's token cost (fire False); at a high touch_rate
    the SAME delta, compounded by future resurfacing, does (fire True). This
    is the multi-step value term (the doc's "planning beyond one question"
    piece) earning its keep, isolated from every other factor."""
    def _decide_at_rate(rate: float):
        gate = _bare_gate(wm=_FakeWM(0.5), action_wm=_FakeActionWM(0.53))  # delta = 0.03
        gate.touch.n_questions = 10
        gate.touch.touches = {"e::ceo": round(rate * 10)}
        return gate.decide(_task(), _board_with_key(), _proposal(["e::ceo"]), Ledger(), "m")

    low = _decide_at_rate(0.0)
    high = _decide_at_rate(0.9)
    assert low.fire is False
    assert high.fire is True


def test_touch_tracker_no_future_peeking():
    t = TouchTracker()
    assert t.rate("k") == 0.0  # nothing asked yet
    t.record(["k", "j"])
    assert t.n_questions == 1
    assert t.rate("k") == 1.0  # k touched in 1 of 1 questions so far
    assert t.rate("missing") == 0.0
    t.record(["j"])
    assert t.rate("k") == 0.5  # k touched in 1 of 2 questions
    assert t.rate("j") == 1.0  # j touched in both


def test_planned_gate_record_question_advances_touch_after_decide():
    """record_question() must be callable AFTER decide() (mirroring
    MajalisSession.ask()'s ordering) and must not retroactively change the
    decision that already ran."""
    gate = _bare_gate(wm=_FakeWM(0.5), action_wm=_FakeActionWM(0.53))
    board = _board_with_key()
    decision1 = gate.decide(_task(), board, _proposal(["e::ceo"]), Ledger(), "m")
    assert gate.touch.n_questions == 0  # decide() must not itself advance the tracker
    gate.record_question(["e::ceo"])
    assert gate.touch.n_questions == 1
    assert gate.touch.rate("e::ceo") == 1.0
    assert decision1.fire is False  # unchanged by the subsequent record_question


# --- target_key (shared between PlannedGate and the mining script) -----------

def test_target_key_picks_highest_risk_support_key():
    board = BeliefBoard()
    board.assert_fact("e::churned", "v1", parse_date_ord("Jan 2025"))
    board.assert_fact("e::churned", "v2", parse_date_ord("Mar 2025"))
    board.assert_fact("e::stable", "w", parse_date_ord("Jan 2025"))
    assert target_key(board, ["e::stable", "e::churned"]) == "e::churned"


def test_target_key_none_when_no_support_key_resolves():
    assert target_key(BeliefBoard(), ["missing::key"]) is None
    assert target_key(BeliefBoard(), []) is None


# --- constructor / drop-in shape ----------------------------------------------

def test_planned_gate_constructs_and_decides_end_to_end(monkeypatch):
    """The real constructor path (no stubbed heads) must not crash even when
    data/wm_action_weights.json doesn't exist yet, and must return the same
    GateDecision shape AcceptGate returns (drop-in requirement)."""
    import majalis.wm_plan as wm_plan
    monkeypatch.setattr(wm_plan, "load_action_wm", lambda path=None: None)
    gate = PlannedGate(wm_mode="heuristic")
    assert gate.action_wm is None
    decision = gate.decide(_task(), _board_with_key(), _proposal(["e::ceo"]), Ledger(), "m")
    d = decision.as_dict()
    assert set(d) == {"fired", "p_wrong", "disagreement", "max_doubt", "weak_current", "reason"}
