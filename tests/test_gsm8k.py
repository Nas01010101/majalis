"""Unit tests for bench/gsm8k.py — answer extraction and the gate adaptation.
A stubbed chat() (no live Qwen calls); assertions check both behavior and
call count so a regression that adds/removes a gate call is caught.
"""
from __future__ import annotations

from majalis.bench import gsm8k


def _queue_chat(monkeypatch, responses: list[str]):
    calls = {"n": 0}
    queue = list(responses)

    def _chat(model, messages, *, ledger, max_tokens=1024, temperature=0.7, seed=None):
        calls["n"] += 1
        return queue.pop(0)

    monkeypatch.setattr(gsm8k, "chat", _chat)
    return calls


# --- extract_gold ----------------------------------------------------------------

def test_extract_gold_simple():
    assert gsm8k.extract_gold("She has 3 eggs.\n#### 18") == 18.0


def test_extract_gold_negative_and_commas():
    assert gsm8k.extract_gold("#### -1,234.5") == -1234.5


def test_extract_gold_missing_marker_raises():
    import pytest
    with pytest.raises(ValueError):
        gsm8k.extract_gold("no marker here")


# --- extract_pred / numeric_match --------------------------------------------------

def test_extract_pred_from_json_tail():
    out = 'Step 1: 9 - 3 = 6\nFinal: {"answer": 6, "confidence": 0.9}'
    assert gsm8k.extract_pred(out) == 6.0


def test_extract_pred_falls_back_to_last_bare_number():
    out = "The ducks lay 16 eggs, she sells 9, so the answer is 18"
    assert gsm8k.extract_pred(out) == 18.0


def test_extract_pred_no_number_returns_none():
    assert gsm8k.extract_pred("I cannot compute this.") is None


def test_numeric_match_tolerance():
    assert gsm8k.numeric_match(18.00001, 18.0)
    assert not gsm8k.numeric_match(18.1, 18.0)
    assert not gsm8k.numeric_match(None, 18.0)


# --- single_arm --------------------------------------------------------------------

def test_single_arm_one_call_only(monkeypatch):
    calls = _queue_chat(monkeypatch, ['reasoning...\n{"answer": 18, "confidence": 0.9}'])
    result = gsm8k.single_arm("q", seed=0)
    assert result.answer == 18.0
    assert result.gate is None
    assert calls["n"] == 1


# --- majalis_gated_arm: confidence-accept fast path (1 call total) ----------------

def test_gated_arm_high_confidence_skips_sampler_and_debate(monkeypatch):
    calls = _queue_chat(monkeypatch, ['reasoning...\n{"answer": 18, "confidence": 0.95}'])
    result = gsm8k.majalis_gated_arm("q", seed=0)
    assert result.answer == 18.0
    assert result.gate["fired"] is False
    assert result.gate["reason"] == "confidence-accept"
    assert result.gate["sampled"] is False
    assert calls["n"] == 1  # propose only — the whole point of the fast path


# --- majalis_gated_arm: mid confidence, samples agree -> accept, no debate --------

def test_gated_arm_mid_confidence_samples_agree_no_debate(monkeypatch):
    calls = _queue_chat(monkeypatch, [
        'reasoning...\n{"answer": 18, "confidence": 0.7}',  # propose
        "18",  # self-consistency sample 1
        "18",  # self-consistency sample 2
    ])
    result = gsm8k.majalis_gated_arm("q", seed=0)
    assert result.answer == 18.0
    assert result.gate["fired"] is False
    assert result.gate["reason"] == "self-consistency-agrees"
    assert result.gate["sampled"] is True
    assert calls["n"] == 3  # propose + k=2 sampler, no skeptic/judge


# --- majalis_gated_arm: low confidence -> debate fires, judge corrects -----------

def test_gated_arm_low_confidence_fires_debate_and_corrects(monkeypatch):
    calls = _queue_chat(monkeypatch, [
        'reasoning...\n{"answer": 20, "confidence": 0.3}',  # propose (wrong)
        "20",  # sample 1 (still wrong, but doesn't matter — conf < FIRE_CONF)
        "18",  # sample 2 (disagrees)
        "the arithmetic in step 2 is wrong",  # skeptic attack
        'corrected...\n{"answer": 18, "confidence": 0.9}',  # judge
    ])
    result = gsm8k.majalis_gated_arm("q", seed=0)
    assert result.answer == 18.0  # judge's corrected answer wins
    assert result.gate["fired"] is True
    assert result.gate["reason"] == "low-confidence"
    assert calls["n"] == 5  # propose + 2 samples + skeptic + judge


# --- majalis_gated_arm: mid confidence, samples disagree -> debate fires ---------

def test_gated_arm_disagreement_fires_debate(monkeypatch):
    calls = _queue_chat(monkeypatch, [
        'reasoning...\n{"answer": 18, "confidence": 0.7}',  # propose
        "18",  # sample 1 agrees
        "5",   # sample 2 disagrees -> disagreement > 0
        "step 3 dropped a term",  # skeptic
        'reasoning...\n{"answer": 18, "confidence": 0.85}',  # judge upholds
    ])
    result = gsm8k.majalis_gated_arm("q", seed=0)
    assert result.gate["fired"] is True
    assert result.gate["reason"] == "self-consistency-disagreement"
    assert calls["n"] == 5


# --- judge output unparseable -> falls back to proposer's answer -----------------

def test_gated_arm_judge_unparseable_falls_back_to_proposal(monkeypatch):
    calls = _queue_chat(monkeypatch, [
        'reasoning...\n{"answer": 20, "confidence": 0.2}',  # propose
        "20", "20",  # samples agree with proposer (disagreement 0, but conf < FIRE_CONF -> fires)
        "no issue found",  # skeptic
        "I cannot determine a number.",  # judge — unparseable
    ])
    result = gsm8k.majalis_gated_arm("q", seed=0)
    assert result.answer == 20.0  # falls back to the original proposal
    assert calls["n"] == 5


# --- browseconf_arm: dedicated cheap confidence call, separate from propose -------

def test_browseconf_high_confidence_skips_debate(monkeypatch):
    calls = _queue_chat(monkeypatch, [
        'reasoning...\n{"answer": 18, "confidence": 0.4}',  # propose (own conf ignored)
        '{"confidence": 0.9}',  # dedicated confidence-only call
    ])
    result = gsm8k.browseconf_arm("q", seed=0)
    assert result.answer == 18.0
    assert result.gate["fired"] is False
    assert calls["n"] == 2  # propose + one dedicated confidence call, no debate


def test_browseconf_low_confidence_fires_debate(monkeypatch):
    calls = _queue_chat(monkeypatch, [
        'reasoning...\n{"answer": 20, "confidence": 0.9}',  # propose (own conf ignored)
        '{"confidence": 0.3}',  # dedicated confidence-only call: below threshold
        "step 2 arithmetic is wrong",  # skeptic
        'corrected...\n{"answer": 18, "confidence": 0.9}',  # judge
    ])
    result = gsm8k.browseconf_arm("q", seed=0)
    assert result.answer == 18.0
    assert result.gate["fired"] is True
    assert calls["n"] == 4  # propose + confidence-only + skeptic + judge
