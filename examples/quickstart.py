"""Majalis in 60 seconds — no API key needed for the world-model half.

    python examples/quickstart.py

Part 1 (offline): the belief board + the learned world model. Feed it
contradictory evidence and watch the trained wrong_now head flag the
poisoned belief — zero LLM calls.

Part 2 (needs DASHSCOPE_API_KEY in the env): the full society — ingest a
stream, ask a question, see the gate route debate only where it's needed.
"""
import os

from majalis.beliefs import BeliefBoard, parse_date_ord
from majalis.wmnet import load_wm

# ---- Part 1: board + learned world model, fully offline -------------------
board = BeliefBoard()
board.assert_fact("acme::ceo", "Jane Doe", parse_date_ord("Jan 2026"), source="Filing")
board.assert_fact("acme::hq", "Berlin", parse_date_ord("Jan 2026"), source="Filing")
# A later-dated rumor displaces the authoritative filing — board corrupted.
board.assert_fact("acme::ceo", "John Roe", parse_date_ord("Mar 2026"), source="Rumor")

wm = load_wm()
for key in ("acme::ceo", "acme::hq"):
    print(f"{key:12s} value={board.current(key).value!r:12s} "
          f"P(wrong)={wm.wrong_now(board, key):.3f} "
          f"P(overturned soon)={wm.superseded_next(board, key):.3f}")
# acme::ceo  -> P(wrong) ~1.0  (rumor displaced a filing: debate-worthy)
# acme::hq   -> P(wrong) ~0.0  (clean: committing costs zero debate tokens)

# ---- Part 2: the full society (Qwen backbones via DashScope) ---------------
if os.environ.get("DASHSCOPE_API_KEY"):
    from majalis.bench.tasks import Task
    from majalis.society import MajalisSession

    session = MajalisSession(seed=0)
    session.ingest([
        "[Jan 2026] Filing: Acme Corp's ceo is Jane Doe.",
        "[Mar 2026] Rumor: Acme Corp's ceo is now John Roe.",
    ])
    result = session.ask(Task(
        task_id="demo", family="stream", context="",
        question=('Claim: "Acme Corp\'s ceo is currently John Roe." '
                  "Policy: filings are authoritative; rumors never override "
                  "a filing. Is this claim true or false? "
                  "Answer 'true' or 'false'."),
        gold="false", meta={"entity": "Acme Corp", "attr": "ceo"}))
    gate = result.transcript[0]["gate"]
    print(f"\nsociety answer: {result.answer!r} | gate fired: {gate['fired']} "
          f"({gate['reason']})")
else:
    print("\n(set DASHSCOPE_API_KEY to run the full-society half)")
