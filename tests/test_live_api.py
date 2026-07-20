"""Live API: replay-schema events, WM scores on /board, spend guard.

All offline — the society is faked; no Qwen calls.
"""
import importlib

import pytest

# fastapi is the [api] extra, not a base dep — skip this module cleanly on a bare
# `pip install -e .` instead of failing collection for the whole suite.
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from majalis import api
from majalis.beliefs import BeliefBoard, parse_date_ord


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
    monkeypatch.delenv("MAJALIS_LIVE_TOKEN", raising=False)
    monkeypatch.delenv("MAJALIS_LIVE_DAILY_CAP", raising=False)
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
    monkeypatch.setenv("MAJALIS_LIVE_DAILY_CAP", "0")
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 429
    assert "paused" in r.json()["detail"]  # cap 0 = paused, not "try tomorrow"
    monkeypatch.setenv("MAJALIS_LIVE_TOKEN", "s3cret")
    ok = client.post("/ask", json={"question": "q"},
                     headers={"X-Majalis-Token": "s3cret"})
    assert ok.status_code == 200
    bad = client.post("/ask", json={"question": "q"},
                      headers={"X-Majalis-Token": "wrong"})
    assert bad.status_code == 429


class _FailingSession(_FakeSession):
    """Simulates a Qwen outage: llm.chat()'s terminal RuntimeError (retry
    exhaustion or fast-fail quota/auth) surfacing through society.py."""

    def ingest(self, lines, trace=None):
        raise RuntimeError("chat failed after 3 retries: 503 Service Unavailable")

    def ask(self, task):
        raise RuntimeError("chat failed after 3 retries: 503 Service Unavailable")


@pytest.fixture()
def failing_client(monkeypatch):
    fake = _FailingSession()
    monkeypatch.setattr(api, "_session", lambda sid: fake)
    api._spent.update(day="", calls=0)
    monkeypatch.delenv("MAJALIS_LIVE_TOKEN", raising=False)
    monkeypatch.delenv("MAJALIS_LIVE_DAILY_CAP", raising=False)
    return TestClient(api.app)


def test_ingest_returns_503_on_llm_failure(failing_client):
    r = failing_client.post(
        "/ingest", json={"lines": ["[Feb 2026] Filing: Acme's HQ is Berlin."]})
    assert r.status_code == 503
    assert "LLM backend unavailable" in r.json()["detail"]


def test_ask_returns_503_on_llm_failure(failing_client):
    r = failing_client.post("/ask", json={"question": "Who is Acme's CEO?"})
    assert r.status_code == 503
    assert "LLM backend unavailable" in r.json()["detail"]


def test_ask_input_validation_still_422_not_503(failing_client):
    """The 503 handler must only catch RuntimeError from the LLM call path —
    input-validation errors (checked before ask() is ever invoked) must not
    be swallowed into a misleading 503."""
    r = failing_client.post("/ask", json={"question": ""})
    assert r.status_code == 422


def test_input_limits(client):
    assert client.post("/ingest", json={"lines": []}).status_code == 422
    assert client.post("/ingest", json={"lines": ["x"] * 13}).status_code == 422
    assert client.post("/ingest", json={"lines": ["y" * 301]}).status_code == 422
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "q" * 401}).status_code == 422


def test_live_page_has_live_mode():
    importlib.import_module("majalis.api")
    page = (api.Path(api.__file__).resolve().parents[2] / "dashboard" / "live.html").read_text()
    for needle in ("mode-live", "ingest-btn", "X-Majalis-Token", "livebar"):
        assert needle in page, needle
