"""Live API: replay-schema events, WM scores on /board, spend guard.

All offline — the society is faked; no Qwen calls.
"""
import importlib

import pytest
from fastapi.testclient import TestClient

from agora import api
from agora.beliefs import BeliefBoard, parse_date_ord


class _Ledger:
    cost_usd = 0.001
    total_tokens = 123


class _FakeSession:
    """Real board (so WM scoring is real), faked LLM paths."""

    def __init__(self):
        self.board = BeliefBoard()
        self.board.assert_fact("acme::ceo", "jane doe",
                               parse_date_ord("Jan 2026"), source="Filing")
        self.board.assert_fact("acme::ceo", "john roe",
                               parse_date_ord("Mar 2026"), source="Rumor")
        self.ingest_ledger = _Ledger()

    def ingest(self, lines, trace=None):
        if trace is not None:
            trace.append({"key": "acme::hq", "value": "berlin",
                          "source": "Filing", "outcome": "new"})
        self.board.assert_fact("acme::hq", "berlin",
                               parse_date_ord("Feb 2026"), source="Filing")

    def ask(self, task):
        class R:
            answer = "true"
            ledger = _Ledger()
            transcript = [{"gate": {"fired": False, "p_wrong": 0.02,
                                    "reason": "conformal(alpha=0.05)"},
                           "events": [{"kind": "proposal", "answer": "true",
                                       "confidence": 0.9, "support": []}]}]
        return R()


@pytest.fixture()
def client(monkeypatch):
    fake = _FakeSession()
    monkeypatch.setattr(api, "_session", lambda sid: fake)
    api._spent.update(day="", calls=0)
    monkeypatch.delenv("AGORA_LIVE_TOKEN", raising=False)
    monkeypatch.delenv("AGORA_LIVE_DAILY_CAP", raising=False)
    return TestClient(api.app)


def test_board_has_wm_scores(client):
    rows = client.get("/board").json()["beliefs"]
    row = next(r for r in rows if r["key"] == "acme::ceo")
    assert {"wrong_now", "superseded_next", "weak", "source"} <= set(row)
    assert 0.0 <= row["wrong_now"] <= 1.0
    # a rumor displaced a filing: the learned WM must flag it hard
    if api._WM is not None:
        assert row["wrong_now"] > 0.5 and row["weak"]


def test_ingest_returns_replay_schema_event(client):
    r = client.post("/ingest", json={"lines": ["[Feb 2026] Filing: Acme's HQ is Berlin."]})
    assert r.status_code == 200
    ev = r.json()["event"]
    assert ev["type"] == "evidence" and ev["asserts"][0]["outcome"] == "new"
    assert all("wrong_now" in b for b in ev["board"])


def test_ask_returns_ungraded_question_event(client):
    r = client.post("/ask", json={"question": "Who is Acme's CEO?"})
    assert r.status_code == 200
    ev = r.json()["event"]
    assert ev["type"] == "question" and ev["correct"] is None and ev["gold"] is None
    assert ev["gate"]["fired"] is False and ev["board"]


def test_spend_guard_cap_and_token(client, monkeypatch):
    monkeypatch.setenv("AGORA_LIVE_DAILY_CAP", "0")
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 429
    assert "paused" in r.json()["detail"]  # cap 0 = paused, not "try tomorrow"
    monkeypatch.setenv("AGORA_LIVE_TOKEN", "s3cret")
    ok = client.post("/ask", json={"question": "q"},
                     headers={"X-Agora-Token": "s3cret"})
    assert ok.status_code == 200
    bad = client.post("/ask", json={"question": "q"},
                      headers={"X-Agora-Token": "wrong"})
    assert bad.status_code == 429


def test_input_limits(client):
    assert client.post("/ingest", json={"lines": []}).status_code == 422
    assert client.post("/ingest", json={"lines": ["x"] * 13}).status_code == 422
    assert client.post("/ingest", json={"lines": ["y" * 301]}).status_code == 422
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "q" * 401}).status_code == 422


def test_live_page_has_live_mode():
    importlib.import_module("agora.api")
    page = (api.Path(api.__file__).resolve().parents[2] / "dashboard" / "live.html").read_text()
    for needle in ("mode-live", "ingest-btn", "X-Agora-Token", "livebar"):
        assert needle in page, needle
