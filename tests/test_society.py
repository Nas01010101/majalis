"""Unit tests for society.py's role functions — a stubbed chat(), no live
LLM calls. Each role gets a happy path (well-formed LLM JSON) and one
adversarial case (garbage/malformed output) exercising the parse-failure
fallback.
"""
from __future__ import annotations

from majalis import society
from majalis.beliefs import BeliefBoard, parse_date_ord
from majalis.bench.tasks import Task
from majalis.handoffs import Challenge, Proposal
from majalis.llm import Ledger


def _stub_chat(response: str):
    def _chat(model, messages, *, ledger, max_tokens=1024, temperature=0.7,
              seed=None):
        return response
    return _chat


def _task(question: str = "Who is Acme's CEO?", gold: str = "jane doe") -> Task:
    return Task(task_id="t0", family="test", context="", question=question, gold=gold)


# --- extract_facts ------------------------------------------------------------

def test_extract_facts_happy_path(monkeypatch):
    monkeypatch.setattr(society, "chat", _stub_chat(
        '[{"entity": "Acme", "attribute": "ceo", "value": "Jane Doe", '
        '"date": "Jan 2026", "source": "Filing"}]'))
    facts = society.extract_facts("evidence text", Ledger(), "qwen3.6-flash")
    assert facts == [{"entity": "Acme", "attribute": "ceo", "value": "Jane Doe",
                      "date": "Jan 2026", "source": "Filing"}]


def test_extract_facts_garbage_returns_empty_list(monkeypatch):
    monkeypatch.setattr(society, "chat", _stub_chat("not valid JSON, sorry about that"))
    assert society.extract_facts("evidence text", Ledger(), "qwen3.6-flash") == []


def test_extract_facts_wrong_shape_json_returns_empty_list(monkeypatch):
    # Adversarial: valid JSON, but an object instead of the expected list.
    monkeypatch.setattr(society, "chat", _stub_chat('{"entity": "Acme"}'))
    assert society.extract_facts("evidence text", Ledger(), "qwen3.6-flash") == []


# --- propose -------------------------------------------------------------------

def test_propose_happy_path(monkeypatch):
    monkeypatch.setattr(society, "chat", _stub_chat(
        '{"answer": "Jane Doe", "rationale": "the latest filing says so", '
        '"support_keys": ["Acme::CEO"], "confidence": 0.8}'))
    board = BeliefBoard()
    p = society.propose(_task(), board, Ledger(), "qwen3.7-max")
    assert p.answer == "Jane Doe"
    assert p.support_keys == ["acme::ceo"]  # lowercased
    assert p.confidence == 0.8
    assert p.author == "proposer"


def test_propose_adversarial_garbage_falls_back_to_defaults(monkeypatch):
    monkeypatch.setattr(society, "chat", _stub_chat("garbage, not json at all"))
    board = BeliefBoard()
    p = society.propose(_task(), board, Ledger(), "qwen3.7-max")
    assert p.answer == ""
    assert p.support_keys == []
    assert p.confidence == 0.5  # documented default when confidence is unparseable


# --- skeptic_challenge -----------------------------------------------------------

def test_skeptic_challenge_happy_path(monkeypatch):
    monkeypatch.setattr(society, "chat", _stub_chat(
        '{"attack": "value is stale", '
        '"sub_questions": ["q1", "q2", "q3", "q4", "q5"]}'))
    board = BeliefBoard()
    board.assert_fact("acme::ceo", "jane doe", parse_date_ord("Jan 2026"), source="Filing")
    proposal = Proposal(answer="jane doe", rationale="r", support_keys=["acme::ceo"],
                        confidence=0.9, author="proposer")
    ch = society.skeptic_challenge(_task(), board, "acme::ceo", proposal,
                                   Ledger(), "qwen3.7-plus")
    assert ch.attack == "value is stale"
    assert ch.sub_questions == ["q1", "q2", "q3", "q4"]  # truncated to 4
    assert ch.author == "skeptic"
    assert ch.target_key == "acme::ceo"


def test_skeptic_challenge_adversarial_garbage_falls_back_to_raw_text(monkeypatch):
    prose = "no json here, just unstructured prose. " * 10
    monkeypatch.setattr(society, "chat", _stub_chat(prose))
    board = BeliefBoard()
    board.assert_fact("acme::ceo", "jane doe", parse_date_ord("Jan 2026"), source="Filing")
    proposal = Proposal(answer="jane doe", rationale="r", support_keys=["acme::ceo"],
                        confidence=0.9, author="proposer")
    ch = society.skeptic_challenge(_task(), board, "acme::ceo", proposal,
                                   Ledger(), "qwen3.7-plus")
    assert ch.sub_questions == []
    assert ch.attack == prose[:300]  # documented fallback


# --- adjudicate ------------------------------------------------------------------

def test_adjudicate_happy_path_corrects_belief(monkeypatch):
    monkeypatch.setattr(society, "chat", _stub_chat(
        '{"upheld": false, "corrected_value": "John Roe", '
        '"rationale": "the later filing wins"}'))
    board = BeliefBoard()
    board.assert_fact("acme::ceo", "jane doe", parse_date_ord("Jan 2026"), source="Rumor")
    proposal = Proposal(answer="jane doe", rationale="r", support_keys=["acme::ceo"],
                        confidence=0.5, author="proposer")
    challenge = Challenge(target_key="acme::ceo", attack="stale", sub_questions=["q1"],
                          author="skeptic")
    v = society.adjudicate(_task(), board, proposal, challenge, Ledger(), "qwen3.7-max")
    assert v.upheld is False
    assert v.corrected_value == "john roe"  # lowercased
    assert v.author == "judge"


def test_adjudicate_adversarial_garbage_defaults_to_upheld(monkeypatch):
    monkeypatch.setattr(society, "chat", _stub_chat("total garbage, not json"))
    board = BeliefBoard()
    board.assert_fact("acme::ceo", "jane doe", parse_date_ord("Jan 2026"), source="Filing")
    proposal = Proposal(answer="jane doe", rationale="r", support_keys=["acme::ceo"],
                        confidence=0.9, author="proposer")
    challenge = Challenge(target_key="acme::ceo", attack="stale", sub_questions=["q1"],
                          author="skeptic")
    v = society.adjudicate(_task(), board, proposal, challenge, Ledger(), "qwen3.7-max")
    assert v.upheld is True  # fail-safe default: belief survives on unparseable output
    assert v.corrected_value is None
