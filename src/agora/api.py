"""Live Agora service — the demo backend (runs on Alibaba Cloud ECS).

    uvicorn agora.api:app --host 0.0.0.0 --port 8080

One AgoraSession per session_id: feed evidence, ask questions, watch the
board and the gate decide. The dashboard's live view drives this API.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .beliefs import BeliefBoard
from .bench.tasks import Task
from .society import AgoraSession

app = FastAPI(title="Agora", version="0.1.0")
_sessions: dict[str, AgoraSession] = {}


def _session(sid: str) -> AgoraSession:
    if sid not in _sessions:
        _sessions[sid] = AgoraSession(seed=0)
    return _sessions[sid]


class IngestBody(BaseModel):
    session_id: str = "default"
    lines: list[str]


class AskBody(BaseModel):
    session_id: str = "default"
    question: str


@app.post("/ingest")
def ingest(body: IngestBody) -> dict:
    s = _session(body.session_id)
    before = len(s.board._current)
    s.ingest(body.lines)
    return {
        "beliefs": len(s.board._current),
        "new_beliefs": len(s.board._current) - before,
        "ingest_tokens": s.ingest_ledger.total_tokens,
    }


@app.post("/ask")
def ask(body: AskBody) -> dict:
    s = _session(body.session_id)
    task = Task(task_id="live", family="live", context="",
                question=body.question, gold="")
    result = s.ask(task)
    trace = result.transcript[0]
    return {
        "answer": result.answer,
        "gate": trace.get("gate"),
        "events": trace.get("events", []),
        "tokens": result.ledger.total_tokens,
        "cost_usd": round(result.ledger.cost_usd, 6),
    }


@app.get("/board")
def board(session_id: str = "default") -> dict:
    b: BeliefBoard = _session(session_id).board
    return {
        "beliefs": [
            {"key": k, "value": b.current(k).value,
             "p_valid": round(b.p_valid(k), 3),
             "doubt": round(b.doubt(k), 3),
             "changes": b.n_supersessions(k)}
            for k in sorted(b._current)
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
    return "<h1>Agora</h1><p>Dashboard not built; see /docs for the API.</p>"


@app.get("/zh", response_class=HTMLResponse)
def index_zh() -> str:
    page = Path(__file__).resolve().parents[2] / "dashboard" / "index.zh.html"
    if page.exists():
        return page.read_text()
    return "<h1>Agora</h1><p>中文页面未构建；请运行 scripts/translate_zh.py。</p>"


@app.get("/zh/live", response_class=HTMLResponse)
def live_zh() -> str:
    page = Path(__file__).resolve().parents[2] / "dashboard" / "live.zh.html"
    if page.exists():
        return page.read_text()
    return "<h1>Agora</h1><p>中文页面未构建；请运行 scripts/translate_zh.py。</p>"


@app.get("/live", response_class=HTMLResponse)
def live() -> str:
    """The society view: a recorded run replayed with the learned world
    model's per-belief risk visible (scripts/build_live.py)."""
    page = Path(__file__).resolve().parents[2] / "dashboard" / "live.html"
    if page.exists():
        return page.read_text()
    return "<h1>Agora</h1><p>Live view not built; run scripts/build_live.py.</p>"
