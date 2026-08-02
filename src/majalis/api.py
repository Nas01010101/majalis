"""Live Majalis service — the demo backend (runs on Alibaba Cloud ECS).

    uvicorn majalis.api:app --host 0.0.0.0 --port 8080

One MajalisSession per session_id: feed evidence, ask questions, watch the
board and the gate decide. The dashboard's live view drives this API.
"""
from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .beliefs import BeliefBoard
from .bench.tasks import Task
from .society import MajalisSession
from .wmnet import load_wm

# docs_url/openapi_url off: the schema was served publicly next to unauthenticated
# billable endpoints, advertising exactly how to spend the owner's credits.
app = FastAPI(title="Majalis", version="0.1.0", docs_url=None, openapi_url=None)
_sessions: dict[str, MajalisSession] = {}
_WM = load_wm()

# Live-demo spend guard: /ingest and /ask cost real Qwen calls. Anonymous
# callers share a daily budget; MAJALIS_LIVE_TOKEN in X-Majalis-Token bypasses it.
_spent = {"day": "", "calls": 0}


def _spend_guard(token: str | None) -> None:
    secret = os.environ.get("MAJALIS_LIVE_TOKEN", "")
    if secret:
        # A token is configured: it authenticates, it does not merely exempt.
        # compare_digest so a wrong guess leaks nothing through response timing.
        if token is None or not secrets.compare_digest(token, secret):
            raise HTTPException(401, detail="X-Majalis-Token required")
        return
    # No token configured. One /ask fans out to several Qwen calls (propose,
    # samplers, skeptic/adjudicate rounds), all billed to the owner, so an open
    # deployment is opt-in rather than the default. The cap below is a courtesy
    # brake, NOT a control: it is per-process, resets on restart, and is counted
    # separately by every uvicorn worker.
    if os.environ.get("MAJALIS_LIVE_OPEN", "") != "1":
        raise HTTPException(401, detail=(
            "this endpoint spends model credits and is closed by default — set "
            "MAJALIS_LIVE_TOKEN and send X-Majalis-Token, or set MAJALIS_LIVE_OPEN=1 "
            "to accept an open, billable endpoint"))
    cap = int(os.environ.get("MAJALIS_LIVE_DAILY_CAP", "25"))
    today = datetime.now(timezone.utc).date().isoformat()
    if _spent["day"] != today:
        _spent.update(day=today, calls=0)
    if cap <= 0:
        raise HTTPException(429, detail=(
            "the shared live demo is paused — watch the recorded run, "
            "or send X-Majalis-Token to run live"))
    if _spent["calls"] >= cap:
        raise HTTPException(429, detail=(
            f"today's shared live-demo budget ({cap} calls) is spent — "
            "try tomorrow, or send X-Majalis-Token"))
    _spent["calls"] += 1


def _snapshot(s: MajalisSession) -> list[dict]:
    """Full board with world-model scores — offline numpy, 0 LLM calls."""
    b = s.board
    snap = []
    for key in sorted(b._current):
        cur = b.current(key)
        if _WM is not None:
            wrong = _WM.wrong_now(b, key)
            sup = _WM.superseded_next(b, key)
        else:  # heuristic fallback when weights are absent
            wrong, sup = b.doubt(key), 1.0 - b.p_valid(key)
        snap.append({
            "key": key, "value": cur.value, "source": cur.source,
            "wrong_now": round(wrong, 3), "superseded_next": round(sup, 3),
            "weak": b.weak_current(key), "churn": b.n_supersessions(key),
            "conflicts": b._conflicts.get(key, 0),
        })
    return snap


def _session(sid: str) -> MajalisSession:
    if sid not in _sessions:
        if len(_sessions) >= 64:  # drop the oldest live session, not the box
            _sessions.pop(next(iter(_sessions)))
        _sessions[sid] = MajalisSession(seed=0)
    return _sessions[sid]


def _existing_session(sid: str) -> MajalisSession:
    """Read-only lookup: never creates.

    /board is a GET and takes the session id straight from the query string. If it
    created sessions, an unauthenticated caller could spin up 64 of them with no
    body and no model spend and evict every real session from the dict.
    """
    if sid not in _sessions:
        raise HTTPException(404, detail="no such session")
    return _sessions[sid]


class IngestBody(BaseModel):
    session_id: str = "default"
    lines: list[str]


class AskBody(BaseModel):
    session_id: str = "default"
    question: str


@app.post("/ingest")
def ingest(body: IngestBody,
           x_majalis_token: str | None = Header(default=None)) -> dict:
    if not (0 < len(body.lines) <= 12) or any(len(x) > 300 for x in body.lines):
        raise HTTPException(422, detail="1-12 evidence lines, each ≤300 chars")
    _spend_guard(x_majalis_token)
    s = _session(body.session_id)
    before = len(s.board._current)
    cost0 = s.ingest_ledger.cost_usd
    asserts: list[dict] = []
    try:
        s.ingest(body.lines, trace=asserts)
    except RuntimeError as exc:
        # llm.chat()'s terminal failure mode (retry exhaustion or a fast-fail
        # quota/auth error, both RuntimeError) — a Qwen-side outage, not a
        # bug in this request. Surface it as a clean 503, not a bare 500.
        raise HTTPException(503, detail=f"LLM backend unavailable: {exc}") from exc
    return {
        "beliefs": len(s.board._current),
        "new_beliefs": len(s.board._current) - before,
        "ingest_tokens": s.ingest_ledger.total_tokens,
        # replay-schema event: the live viewer feeds it straight to the renderer
        "event": {"type": "evidence", "lines": body.lines, "asserts": asserts,
                  "cost_usd": round(s.ingest_ledger.cost_usd - cost0, 5),
                  "board": _snapshot(s)},
    }


@app.post("/ask")
def ask(body: AskBody,
        x_majalis_token: str | None = Header(default=None)) -> dict:
    if not (0 < len(body.question) <= 400):
        raise HTTPException(422, detail="question must be 1-400 chars")
    _spend_guard(x_majalis_token)
    s = _session(body.session_id)
    task = Task(task_id="live", family="live", context="",
                question=body.question, gold="")
    try:
        result = s.ask(task)
    except RuntimeError as exc:
        raise HTTPException(503, detail=f"LLM backend unavailable: {exc}") from exc
    trace = result.transcript[0]
    return {
        "answer": result.answer,
        "gate": trace.get("gate"),
        "events": trace.get("events", []),
        "tokens": result.ledger.total_tokens,
        "cost_usd": round(result.ledger.cost_usd, 6),
        "event": {"type": "question", "task_id": "live",
                  "question": body.question, "gold": None,
                  "answer": result.answer, "correct": None,
                  "gate": trace.get("gate"), "events": trace.get("events", []),
                  "cost_usd": round(result.ledger.cost_usd, 5),
                  "tokens": result.ledger.total_tokens,
                  "board": _snapshot(s)},
    }


@app.get("/board")
def board(session_id: str = "default") -> dict:
    s = _existing_session(session_id)
    b: BeliefBoard = s.board
    return {
        "beliefs": [
            {"key": r["key"], "value": r["value"],
             "p_valid": round(b.p_valid(r["key"]), 3),
             "doubt": round(b.doubt(r["key"]), 3),
             "changes": r["churn"], "wrong_now": r["wrong_now"],
             "superseded_next": r["superseded_next"], "weak": r["weak"],
             "source": r["source"]}
            for r in _snapshot(s)
        ],
        "doubted": [k for k, _ in b.doubts(0.3)],
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    page = Path(__file__).resolve().parents[2] / "dashboard" / "index.html"
    if page.exists():
        return page.read_text()
    return "<h1>Majalis</h1><p>Dashboard not built; see /docs for the API.</p>"


@app.get("/zh", response_class=HTMLResponse)
def index_zh() -> str:
    page = Path(__file__).resolve().parents[2] / "dashboard" / "index.zh.html"
    if page.exists():
        return page.read_text()
    return "<h1>Majalis</h1><p>中文页面未构建；请运行 scripts/translate_zh.py。</p>"


@app.get("/zh/live", response_class=HTMLResponse)
def live_zh() -> str:
    page = Path(__file__).resolve().parents[2] / "dashboard" / "live.zh.html"
    if page.exists():
        return page.read_text()
    return "<h1>Majalis</h1><p>中文页面未构建；请运行 scripts/translate_zh.py。</p>"


@app.get("/live", response_class=HTMLResponse)
def live() -> str:
    """The society view: a recorded run replayed with the learned world
    model's per-belief risk visible (scripts/build_live.py)."""
    page = Path(__file__).resolve().parents[2] / "dashboard" / "live.html"
    if page.exists():
        return page.read_text()
    return "<h1>Majalis</h1><p>Live view not built; run scripts/build_live.py.</p>"
