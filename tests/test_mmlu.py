"""Unit tests for bench/mmlu.py — letter extraction, the gate adaptation, and the
vanilla-MAD baseline. A stubbed chat() (no live Qwen calls); assertions check both
behavior and call count, so a regression that adds or removes a gate call is caught.

Call-count coverage is the point: the MMLU headline claim is a COST result (the gate
matches vanilla debate at ~1/6 the calls), so the per-arm call counts are load-bearing
numbers, not incidental.
"""
from __future__ import annotations

import pytest

from majalis.bench import mmlu

CHOICES = ["ten", "eleven", "twelve", "thirteen"]


def _queue_chat(monkeypatch, responses: list[str]):
    calls = {"n": 0}
    queue = list(responses)

    def _chat(model, messages, *, ledger, max_tokens=1024, temperature=0.7, seed=None):
        calls["n"] += 1
        return queue.pop(0)

    monkeypatch.setattr(mmlu, "chat", _chat)
    return calls


# --- format_choices ---------------------------------------------------------------

def test_format_choices_labels_a_through_d():
    assert mmlu.format_choices(CHOICES) == "A. ten\nB. eleven\nC. twelve\nD. thirteen"


# --- extract_pred / letter_match ---------------------------------------------------

def test_extract_pred_from_json_tail():
    out = 'Working through it...\n{"answer": "C", "confidence": 0.9}'
    assert mmlu.extract_pred(out) == "C"


def test_extract_pred_json_wins_over_earlier_letters():
    out = 'Option A looks plausible, but B is wrong.\n{"answer": "D", "confidence": 0.8}'
    assert mmlu.extract_pred(out) == "D"


def test_extract_pred_lowercase_json_letter_is_normalized():
    assert mmlu.extract_pred('{"answer": "b", "confidence": 0.7}') == "B"


def test_extract_pred_falls_back_to_last_bare_letter():
    assert mmlu.extract_pred("Eliminating the others, the best choice is C") == "C"


def test_extract_pred_no_letter_returns_none():
    assert mmlu.extract_pred("I cannot determine this.") is None


def test_letter_match():
    assert mmlu.letter_match("B", "B")
    assert not mmlu.letter_match("B", "C")
    assert not mmlu.letter_match(None, "B")


# --- single_arm --------------------------------------------------------------------

def test_single_arm_one_call_only(monkeypatch):
    calls = _queue_chat(monkeypatch, ['reasoning...\n{"answer": "C", "confidence": 0.9}'])
    result = mmlu.single_arm("q", CHOICES, seed=0)
    assert result.answer == "C"
    assert result.gate is None
    assert calls["n"] == 1


# --- majalis_gated_arm: confidence-accept fast path (1 call total) -----------------

def test_gated_arm_high_confidence_skips_sampler_and_debate(monkeypatch):
    calls = _queue_chat(monkeypatch, ['reasoning...\n{"answer": "C", "confidence": 0.95}'])
    result = mmlu.majalis_gated_arm("q", CHOICES, seed=0)
    assert result.answer == "C"
    assert result.gate["fired"] is False
    assert result.gate["reason"] == "confidence-accept"
    assert result.gate["sampled"] is False
    assert calls["n"] == 1  # propose only — the zero-extra-call fast path


# --- majalis_gated_arm: mid confidence, samples agree -> accept, no debate ---------

def test_gated_arm_mid_confidence_samples_agree_no_debate(monkeypatch):
    calls = _queue_chat(monkeypatch, [
        'reasoning...\n{"answer": "C", "confidence": 0.7}',  # propose
        "C",  # self-consistency sample 1
        "C",  # self-consistency sample 2
    ])
    result = mmlu.majalis_gated_arm("q", CHOICES, seed=0)
    assert result.answer == "C"
    assert result.gate["fired"] is False
    assert result.gate["reason"] == "self-consistency-agrees"
    assert result.gate["sampled"] is True
    assert result.gate["disagreement"] == 0.0
    assert calls["n"] == 3  # propose + k=2 sampler, no skeptic/judge


# --- majalis_gated_arm: mid confidence, samples disagree -> debate fires -----------

def test_gated_arm_disagreement_fires_debate(monkeypatch):
    calls = _queue_chat(monkeypatch, [
        'reasoning...\n{"answer": "C", "confidence": 0.7}',  # propose
        "C",  # sample 1 agrees
        "A",  # sample 2 disagrees -> disagreement > 0
        "you misread the second premise",  # skeptic
        'reasoning...\n{"answer": "C", "confidence": 0.9}',  # judge upholds
    ])
    result = mmlu.majalis_gated_arm("q", CHOICES, seed=0)
    assert result.answer == "C"
    assert result.gate["fired"] is True
    assert result.gate["reason"] == "self-consistency-disagreement"
    assert result.gate["disagreement"] > 0.0
    assert calls["n"] == 5  # propose + 2 samples + skeptic + judge


# --- majalis_gated_arm: low confidence -> debate fires, judge corrects -------------

def test_gated_arm_low_confidence_fires_debate_and_corrects(monkeypatch):
    calls = _queue_chat(monkeypatch, [
        'reasoning...\n{"answer": "A", "confidence": 0.3}',  # propose (wrong)
        "A", "A",  # samples agree, but confidence < FIRE_CONF so it fires anyway
        "option A contradicts the stated constraint",  # skeptic
        'corrected...\n{"answer": "C", "confidence": 0.9}',  # judge
    ])
    result = mmlu.majalis_gated_arm("q", CHOICES, seed=0)
    assert result.answer == "C"  # judge's corrected answer wins
    assert result.gate["fired"] is True
    assert result.gate["reason"] == "low-confidence"
    assert calls["n"] == 5


# --- judge output unparseable -> falls back to the proposer's answer ---------------

def test_gated_arm_judge_unparseable_falls_back_to_proposal(monkeypatch):
    calls = _queue_chat(monkeypatch, [
        'reasoning...\n{"answer": "D", "confidence": 0.2}',  # propose
        "D", "D",  # samples agree; conf < FIRE_CONF -> fires
        "no error found",  # skeptic
        "I cannot determine the answer.",  # judge — unparseable
    ])
    result = mmlu.majalis_gated_arm("q", CHOICES, seed=0)
    assert result.answer == "D"  # falls back to the original proposal
    assert calls["n"] == 5


# --- mad_arm: the vanilla-debate cost baseline ------------------------------------

def test_mad_arm_costs_n_agents_times_rounds_calls(monkeypatch):
    """3 agents x 2 rounds = 6 calls. This is the 6x-cost number the README and
    paper quote against the gate's ~1.05, so it is asserted exactly."""
    calls = _queue_chat(monkeypatch, [
        # round 1: three independent proposals
        'a1\n{"answer": "C", "confidence": 0.8}',
        'a2\n{"answer": "A", "confidence": 0.6}',
        'a3\n{"answer": "C", "confidence": 0.7}',
        # round 2: three revisions after seeing each other
        'a1\n{"answer": "C", "confidence": 0.9}',
        'a2\n{"answer": "C", "confidence": 0.8}',
        'a3\n{"answer": "C", "confidence": 0.9}',
    ])
    result = mmlu.mad_arm("q", CHOICES, seed=0)
    assert result.answer == "C"
    assert result.gate is None  # MAD always debates; there is no gate to report
    assert calls["n"] == 6


def test_mad_arm_takes_majority_not_first_agent(monkeypatch):
    """Final answer is the majority vote of the last round, so a dissenting
    agent 1 must not win."""
    _queue_chat(monkeypatch, [
        'a1\n{"answer": "A", "confidence": 0.9}',
        'a2\n{"answer": "B", "confidence": 0.9}',
        'a3\n{"answer": "B", "confidence": 0.9}',
        'a1\n{"answer": "A", "confidence": 0.9}',  # holds out
        'a2\n{"answer": "B", "confidence": 0.9}',
        'a3\n{"answer": "B", "confidence": 0.9}',
    ])
    result = mmlu.mad_arm("q", CHOICES, seed=0)
    assert result.answer == "B"


def test_mad_arm_all_unparseable_returns_none(monkeypatch):
    _queue_chat(monkeypatch, ["nothing usable here"] * 6)
    result = mmlu.mad_arm("q", CHOICES, seed=0)
    assert result.answer is None


# --- the arms registry the runner dispatches through -------------------------------

def test_arms_registry_exposes_the_three_benchmarked_arms():
    assert set(mmlu.ARMS_MMLU) == {"single", "majalis-gated", "mad"}
    assert all(callable(fn) for fn in mmlu.ARMS_MMLU.values())


@pytest.mark.parametrize("threshold,expected", [(mmlu.GATE_CONF_SKIP, 0.85),
                                                (mmlu.GATE_FIRE_CONF, 0.6)])
def test_gate_thresholds_are_the_documented_values(threshold, expected):
    """These are quoted as the uncalibrated thresholds in the paper; pin them so a
    silent retune cannot invalidate the published fire rate."""
    assert threshold == expected
