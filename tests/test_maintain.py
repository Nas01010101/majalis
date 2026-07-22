"""Zero-latency maintenance mode — real-debate window between batches.

Stubbed chat() throughout (test_society.py's convention): these verify the
orchestration — ranking, budget, write-back, ledger routing, arm wiring —
not LLM quality (that's the live cells' job).
"""
from __future__ import annotations

import json

from majalis import society
from majalis.society import MajalisSession


def _seed_board(session: MajalisSession, facts: list[tuple[str, str, int, str]]):
    for key, value, date_ord, source in facts:
        session.board.assert_fact(key, value, date_ord, source=source)


def _scripted_chat(responses: list[str]):
    """Pop one canned response per chat() call; fail loudly if over-called."""
    calls = []

    def _chat(model, messages, *, ledger, max_tokens=1024, temperature=0.7,
              seed=None):
        calls.append(messages[0]["content"])
        assert responses, "chat() called more times than scripted"
        return responses.pop(0)
    _chat.calls = calls
    return _chat


def test_maintain_corrects_wrong_belief(monkeypatch):
    session = MajalisSession(gate_mode="never", wm_mode="learned")
    _seed_board(session, [("acme::ceo", "jane doe", 100, "Filing"),
                          ("acme::ceo", "john roe", 105, "Rumor")])
    assert session.board.current("acme::ceo").value == "john roe"
    monkeypatch.setattr(society, "chat", _scripted_chat([
        json.dumps({"attack": "rumor postdates filing", "sub_questions": ["q1"]}),
        json.dumps({"upheld": False, "corrected_value": "jane doe",
                    "rationale": "filing wins"}),
    ]))
    repairs = session.maintain(budget=1)
    assert len(repairs) == 1
    assert repairs[0]["corrected"] == "jane doe"
    assert session.board.current("acme::ceo").value == "jane doe"
    assert session.board.current("acme::ceo").source == "debate"


def test_maintain_upheld_leaves_board_untouched(monkeypatch):
    session = MajalisSession(gate_mode="never", wm_mode="learned")
    _seed_board(session, [("acme::ceo", "jane doe", 100, "Filing")])
    monkeypatch.setattr(society, "chat", _scripted_chat([
        json.dumps({"attack": "looks fine", "sub_questions": ["q1"]}),
        json.dumps({"upheld": True, "corrected_value": None, "rationale": "ok"}),
    ]))
    repairs = session.maintain(budget=1)
    assert repairs[0]["upheld"] is True
    assert session.board.current("acme::ceo").value == "jane doe"
    assert session.board.current("acme::ceo").source == "Filing"


def test_maintain_budget_and_risk_ordering(monkeypatch):
    """With budget=1 only the HIGHEST-risk key gets the debate."""
    session = MajalisSession(gate_mode="never", wm_mode="learned")
    _seed_board(session, [("acme::ceo", "jane doe", 100, "Filing"),
                          ("zorp::hq", "berlin", 100, "Rumor")])

    class _FakeWM:
        def wrong_now(self, board, key):
            return {"acme::ceo": 0.1, "zorp::hq": 0.9}[key]
    monkeypatch.setattr(session.gate, "wm", _FakeWM())
    monkeypatch.setattr(society, "chat", _scripted_chat([
        json.dumps({"attack": "a", "sub_questions": []}),
        json.dumps({"upheld": True, "corrected_value": None, "rationale": ""}),
    ]))
    repairs = session.maintain(budget=1)
    assert [r["key"] for r in repairs] == ["zorp::hq"]


def test_maintain_spend_lands_on_ingest_ledger(monkeypatch):
    """Maintenance is a between-batch cost — it must be amortized on
    ingest_ledger, never on a question's ledger."""
    session = MajalisSession(gate_mode="never", wm_mode="learned")
    _seed_board(session, [("acme::ceo", "jane doe", 100, "Filing")])
    seen_ledgers = []

    def _chat(model, messages, *, ledger, max_tokens=1024, temperature=0.7,
              seed=None):
        seen_ledgers.append(ledger)
        return json.dumps({"attack": "a", "sub_questions": [],
                           "upheld": True, "corrected_value": None,
                           "rationale": ""})
    monkeypatch.setattr(society, "chat", _chat)
    session.maintain(budget=1)
    assert seen_ledgers and all(l is session.ingest_ledger for l in seen_ledgers)


def test_maintain_empty_board_is_noop(monkeypatch):
    session = MajalisSession(gate_mode="never", wm_mode="learned")
    monkeypatch.setattr(society, "chat", _scripted_chat([]))  # would fail if called
    assert session.maintain(budget=3) == []


def test_maintain_arm_wired_into_replays(monkeypatch):
    """The majalis-maintain arm must call maintain() once per evidence batch
    and never fire an ask-time debate (gate_mode='never')."""
    from majalis.bench import session as bench_session
    calls = {"maintain": 0}

    class _FakeSession:
        def __init__(self, **kw):
            assert kw["gate_mode"] == "never"
            self.ingest_ledger = society.Ledger()

        def ingest(self, lines):
            pass

        def maintain(self, budget):
            calls["maintain"] += 1
            return []

        def ask(self, task):
            from majalis.bench.arms import ArmResult
            return ArmResult("true", society.Ledger(),
                             [{"role": "trace", "gate": {"fired": False}}])
    monkeypatch.setattr(bench_session, "MajalisSession", _FakeSession)
    from majalis.bench.stream import make_session
    events = make_session(0, n_steps=3)
    n_batches = sum(1 for e in events if e.kind == "evidence")
    records = bench_session.REPLAYS["majalis-maintain"](events, 0)
    assert calls["maintain"] == n_batches
    assert all(not r["gate"]["fired"] for r in records)
