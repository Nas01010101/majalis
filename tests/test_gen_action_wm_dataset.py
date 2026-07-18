"""Tests for scripts/gen_action_wm_dataset.py's shared-board-once mining —
stubbed chat(), zero live LLM calls.

The property under test is the design's core rejection of the naive
"run majalis-nogate and majalis-nodebate as two separate arms" approach
(design_track3_worldmodel.md §2.1): perception (extract_facts) and the
pre-debate proposal must each happen EXACTLY ONCE per question, with BOTH
the skip_correct and debate_correct labels derived from that single shared
pass — never two independently-rebuilt boards.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from majalis import society  # noqa: E402
from majalis.bench.stream import SessionEvent  # noqa: E402
from majalis.bench.tasks import Task  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "gen_action_wm_dataset", ROOT / "scripts" / "gen_action_wm_dataset.py")
gen_action_wm_dataset = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gen_action_wm_dataset)


def _make_stub_chat(call_log: list[str]):
    def _chat(model, messages, *, ledger, max_tokens=1024, temperature=0.7, seed=None):
        prompt = messages[0]["content"]
        if "Extract every dated factual assertion" in prompt:
            call_log.append("extract_facts")
            return ('[{"entity": "Acme", "attribute": "ceo", "value": "Jane Doe", '
                    '"date": "Jan 2025", "source": "Filing"}]')
        if "adversarial fact-checker" in prompt:
            call_log.append("skeptic_challenge")
            return '{"attack": "stale", "sub_questions": ["q1"]}'
        if "You are a judge" in prompt:
            call_log.append("adjudicate")
            return '{"upheld": true, "corrected_value": null, "rationale": "r"}'
        call_log.append("propose")
        return ('{"answer": "true", "rationale": "r", '
                '"support_keys": ["acme::ceo"], "confidence": 0.8}')
    return _chat


def _one_question_session(seed, n_steps=1):
    return [
        SessionEvent("evidence", lines=["[Jan 2025] Filing: Acme's ceo is Jane Doe."]),
        SessionEvent("question", task=Task(
            task_id=f"s{seed}-q0", family="stream", context="",
            question='Claim: "Acme\'s ceo is currently Jane Doe." true or false?',
            gold="true", meta={"entity": "Acme", "attr": "ceo"})),
    ]


def test_mine_seed_shares_one_perception_and_proposal_pass(monkeypatch):
    call_log: list[str] = []
    monkeypatch.setattr(society, "chat", _make_stub_chat(call_log))
    monkeypatch.setattr(gen_action_wm_dataset, "make_session", _one_question_session)

    rows, cost = gen_action_wm_dataset.mine_seed(100, n_steps=1)

    # Exactly ONE extract_facts (shared perception) + ONE pre-debate propose
    # + ONE skeptic_challenge + ONE adjudicate + ONE post-debate re-propose.
    # A second, independent perception/proposal pass would show up here as
    # extra "extract_facts"/"propose" entries.
    assert call_log == ["extract_facts", "propose", "skeptic_challenge",
                        "adjudicate", "propose"]
    assert len(rows) == 1


def test_mine_seed_pairs_skip_and_debate_labels_from_the_same_row(monkeypatch):
    call_log: list[str] = []
    monkeypatch.setattr(society, "chat", _make_stub_chat(call_log))
    monkeypatch.setattr(gen_action_wm_dataset, "make_session", _one_question_session)

    rows, _ = gen_action_wm_dataset.mine_seed(100, n_steps=1)
    row = rows[0]
    assert row["uid"] == "stream:100:s100-q0"
    assert row["key"] == "acme::ceo"
    assert row["skip_correct"] == 1  # pre-debate "true" matches gold "true"
    assert row["debate_correct"] == 1  # post-debate "true" also matches gold
    assert len(row["x"]) == 13  # 12 wmfeat features + touch_rate
    assert row["touch_rate"] == 0.0  # first (and only) question in this session


def test_mine_seed_rejects_eval_seeds():
    import pytest
    with pytest.raises(AssertionError):
        gen_action_wm_dataset.mine_seed(42, n_steps=1)  # eval seeds 0-99 untouchable


def test_mine_seed_touch_rate_advances_across_questions(monkeypatch):
    """Two questions targeting the SAME key in one session: the second
    question's touch_rate must reflect the first (no future peeking, no
    double counting of the current question)."""
    call_log: list[str] = []
    monkeypatch.setattr(society, "chat", _make_stub_chat(call_log))

    def _two_question_session(seed, n_steps=1):
        return [
            SessionEvent("evidence", lines=["[Jan 2025] Filing: Acme's ceo is Jane Doe."]),
            SessionEvent("question", task=Task(
                task_id=f"s{seed}-q0", family="stream", context="",
                question='Claim: "Acme\'s ceo is currently Jane Doe." true or false?',
                gold="true", meta={"entity": "Acme", "attr": "ceo"})),
            SessionEvent("question", task=Task(
                task_id=f"s{seed}-q1", family="stream", context="",
                question='Claim: "Acme\'s ceo is currently Jane Doe." true or false?',
                gold="true", meta={"entity": "Acme", "attr": "ceo"})),
        ]

    monkeypatch.setattr(gen_action_wm_dataset, "make_session", _two_question_session)
    rows, _ = gen_action_wm_dataset.mine_seed(100, n_steps=1)
    assert len(rows) == 2
    assert rows[0]["touch_rate"] == 0.0  # nothing asked before q0
    assert rows[1]["touch_rate"] == 1.0  # acme::ceo was touched in 1 of 1 prior questions
