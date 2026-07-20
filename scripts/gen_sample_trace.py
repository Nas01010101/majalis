#!/usr/bin/env python3
"""Generate examples/sample_trace.jsonl — a deterministic, zero-LLM-call
Majalis run for `majalis replay` to render with no API key required.

Belief-board asserts/supersessions, the per-key risk score, AND the gate's
fire/no-fire decision all run through the REAL board + gate math
(majalis.beliefs.assert_fact/doubt/weak_current, majalis.wm.risk_score + the
AcceptGate accept floor) so every number a judge sees is genuine, not
fabricated. Only the role *dialogue* — proposer/skeptic/judge text, which
normally comes from a Qwen call — is scripted, the same way
tests/test_society.py stubs chat() with canned strings. Condenses
the demo_company.py story (rumor-poisoned ARR belief, gate fires, skeptic
decomposes, judge repairs) into a single committed fixture.

    python scripts/gen_sample_trace.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majalis.beliefs import BeliefBoard, parse_date_ord  # noqa: E402
from majalis.wm import GateDecision, risk_score  # noqa: E402

OUT = ROOT / "examples" / "sample_trace.jsonl"


def board_snapshot(board: BeliefBoard, keys: list[str]) -> list[dict]:
    return [{
        "key": k, "value": board.current(k).value, "source": board.current(k).source,
        "risk": round(board.doubt(k), 3), "weak": board.weak_current(k),
        "churn": board.n_supersessions(k),
    } for k in keys if board.current(k) is not None]


def assert_batch(board: BeliefBoard, lines: list[dict]) -> list[dict]:
    """lines: [{"entity","attr","value","date","source"}] -> assert records."""
    out = []
    for ln in lines:
        key = BeliefBoard.make_key(ln["entity"], ln["attr"])
        outcome = board.assert_fact(key, ln["value"], parse_date_ord(ln["date"]),
                                    source=ln["source"])
        out.append({"key": key, "value": ln["value"].strip().lower(),
                    "source": ln["source"], "outcome": outcome})
    return out


def clean_gate(board: BeliefBoard, key: str, confidence: float = 0.9) -> GateDecision:
    """The gate decision for a non-weak key, computed the way wm.AcceptGate.decide
    does on its no-sampler path: p_wrong = risk_score(board doubt, no disagreement,
    proposer confidence), fire iff the uncalibrated accept floor (0.35) is breached.
    No hardcoded p_wrong, no LLM call — real board doubt through the real risk math."""
    doubt = board.doubt(key)
    p_wrong = risk_score(doubt, disagreement=0.0, confidence=confidence,
                         weak_current=False)
    return GateDecision(fire=not (p_wrong < 0.35), p_wrong=round(p_wrong, 3),
                        disagreement=0.0, max_doubt=round(doubt, 3),
                        reason="uncalibrated-floor(0.35)", weak_current=False)


def main() -> None:
    board = BeliefBoard()
    records: list[dict] = [{"type": "act", "title": (
        "MAJALIS SAMPLE RUN — investment-committee due-diligence "
        "(synthetic fixture, zero LLM calls)")}]

    # --- Evidence round 1: two clean Q1 filings ------------------------------
    round1 = [
        {"entity": "Meridian Labs", "attr": "ceo", "value": "Chen",
         "date": "Jan 2026", "source": "Filing"},
        {"entity": "Meridian Labs", "attr": "arr", "value": "$42M",
         "date": "Jan 2026", "source": "Filing"},
        {"entity": "Halcyon Systems", "attr": "arr", "value": "$61M",
         "date": "Feb 2026", "source": "Filing"},
        {"entity": "Halcyon Systems", "attr": "litigation status", "value": "clear",
         "date": "Mar 2026", "source": "Filing"},
    ]
    keys = [BeliefBoard.make_key(ln["entity"], ln["attr"]) for ln in round1]
    asserts1 = assert_batch(board, round1)
    records.append({"type": "evidence",
                    "lines": [f"[{ln['date']}] {ln['source']}: {ln['entity']}'s "
                              f"{ln['attr']} is {ln['value']}." for ln in round1],
                    "asserts": asserts1, "board": board_snapshot(board, keys)})

    # Clean question — board is clean, gate stays shut. The fire/no-fire and
    # p_wrong are COMPUTED by the real gate math (risk_score + the uncalibrated
    # accept floor from wm.AcceptGate.decide), not narrated — same as q2 below.
    q1_gate = clean_gate(board, BeliefBoard.make_key("Halcyon Systems", "arr"))
    records.append({
        "type": "question", "task_id": "Q1",
        "question": "Claim: \"Halcyon Systems's current ARR is above $50M.\" True or false?",
        "gold": "true", "answer": "true", "correct": True,
        "gate": q1_gate.as_dict(), "events": [],
        "board": board_snapshot(board, [BeliefBoard.make_key("Halcyon Systems", "arr")]),
    })

    # --- Evidence round 2: a legit filing update + a rumor poisoning --------
    round2 = [
        {"entity": "Meridian Labs", "attr": "arr", "value": "$47M",
         "date": "Apr 2026", "source": "Filing"},  # filing supersedes filing: clean
        {"entity": "Meridian Labs", "attr": "arr", "value": "$88M",
         "date": "May 2026", "source": "Rumor"},   # rumor displaces a filing: POISON
    ]
    asserts2 = assert_batch(board, round2)
    arr_key = BeliefBoard.make_key("Meridian Labs", "arr")
    records.append({"type": "evidence",
                    "lines": [f"[{ln['date']}] {ln['source']}: {ln['entity']}'s "
                              f"{ln['attr']} is {ln['value']}." for ln in round2],
                    "asserts": asserts2, "board": board_snapshot(board, [arr_key])})
    assert asserts2[-1]["outcome"] == "superseded-by-weaker-source", (
        "generator drifted from beliefs.py's tier logic — fix the fixture, not the assertion")

    # Poisoned question — gate MUST fire (policy:weak-source), skeptic
    # decomposes, judge reverts to the filing, belief is written back.
    weak_risk = risk_score(board.doubt(arr_key), disagreement=0.0, confidence=0.6,
                           weak_current=True)
    q2_gate = GateDecision(fire=True, p_wrong=weak_risk, disagreement=0.0,
                           max_doubt=round(board.doubt(arr_key), 3),
                           reason="policy:weak-source", weak_current=True)
    challenge_event = {
        "kind": "challenge", "key": arr_key,
        "attack": ("The current value ($88M) comes from a Rumor dated after the "
                   "most recent Filing ($47M). Policy: filings are authoritative "
                   "and rumors never override one — this looks like a poisoned "
                   "belief, not a legitimate update."),
        "sub_questions": [
            "Is the $88M assertion sourced from a Filing or a Rumor?",
            "Is there a Filing dated later than the $88M Rumor?",
            "Does company policy allow a Rumor to override a Filing?",
        ],
    }
    verdict_event = {"kind": "verdict", "key": arr_key, "upheld": False,
                     "corrected": "$47m"}
    # write-back: judge's correction supersedes the rumor on the shared board.
    board.assert_fact(arr_key, "$47M", board._now_ord + 1, source="debate")
    records.append({
        "type": "question", "task_id": "Q2",
        "question": ("Claim: \"Meridian Labs's ARR is above $50M per its most "
                     "recent filing.\" True or false?"),
        "gold": "false", "answer": "false", "correct": True,
        "gate": q2_gate.as_dict(), "events": [challenge_event, verdict_event],
        "board": board_snapshot(board, [arr_key]),
    })

    # Repeat question — the repaired belief now commits for free (computed).
    q3_gate = clean_gate(board, arr_key)
    records.append({
        "type": "question", "task_id": "Q3",
        "question": ("Claim: \"Meridian Labs's ARR is above $50M per its most "
                     "recent filing.\" True or false? (asked again, post-repair)"),
        "gold": "false", "answer": "false", "correct": True,
        "gate": q3_gate.as_dict(), "events": [],
        "board": board_snapshot(board, [arr_key]),
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"wrote {len(records)} events -> {OUT}")


if __name__ == "__main__":
    main()
