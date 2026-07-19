#!/usr/bin/env python3
"""Investment-committee due-diligence demo — the meaty scripted story.

A research firm runs diligence on two companies over two evidence rounds,
builds a versioned belief board, gets poisoned by two later-dated rumors, and
then answers a committee memo's decision questions. The learned world model
routes skeptic->judge debate ONLY to the beliefs a rumor corrupted; every
clean question is answered instantly at zero debate cost.

Run (key is read from the environment, never argv):

    cd <repo> && set -a && source .env && set +a && python scripts/demo_company.py
    python scripts/demo_company.py --maintain   # repair the poison proactively

It drives a local MajalisSession directly (no HTTP), with real Qwen calls.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from majalis.bench.tasks import Task
from majalis.society import MajalisSession

# --- terminal styling (video script: cyan Q / green A / gray notes) ----------
_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def cyan(s: str) -> str: return _c("36", s)
def green(s: str) -> str: return _c("32", s)
def red(s: str) -> str: return _c("31", s)
def yellow(s: str) -> str: return _c("33", s)
def gray(s: str) -> str: return _c("90", s)
def bold(s: str) -> str: return _c("1", s)
def magenta(s: str) -> str: return _c("35", s)


def act(title: str) -> None:
    bar = "=" * 74
    print("\n" + bold(cyan(bar)))
    print(bold(cyan(f"  {title}")))
    print(bold(cyan(bar)))


def note(s: str) -> None:
    print(gray("  " + s))


# --- the story ----------------------------------------------------------------
POLICY = (
    "Policy: SEC filings are authoritative; analyst notes, blog recaps and "
    "market rumors are unreliable and NEVER override a filing."
)

# Round 1: clean Q1-2026 filings for two diligence targets.
ROUND1 = [
    "[Jan 2026] Filing: Meridian Labs's ceo is Chen.",
    "[Jan 2026] Filing: Meridian Labs's arr is $42M.",
    "[Feb 2026] Filing: Meridian Labs's flagship product is Atlas.",
    "[Feb 2026] Filing: Meridian Labs's litigation status is clear.",
    "[Jan 2026] Filing: Halcyon Systems's ceo is Okafor.",
    "[Feb 2026] Filing: Halcyon Systems's arr is $61M.",
    "[Feb 2026] Filing: Halcyon Systems's flagship product is Kestrel.",
    "[Mar 2026] Filing: Halcyon Systems's litigation status is clear.",
]

# Round 2: legitimate Q2 filing updates + TWO later-dated rumor poisonings.
ROUND2 = [
    # Legit, authoritative updates (filing supersedes filing -> no poison).
    "[Apr 2026] Filing: Meridian Labs's ceo is Okafor.",
    "[Apr 2026] Filing: Meridian Labs's arr is $47M.",
    # POISON 1 — a rumor inflates Meridian's ARR above the committee's bar.
    "[May 2026] Rumor: Meridian Labs's arr is now $88M.",
    # POISON 2 — a rumor invents a litigation scare on Halcyon.
    "[May 2026] Rumor: Halcyon Systems's litigation status is an active SEC investigation.",
]

# The two beliefs a rumor corrupted (later-dated, weaker source than a filing).
POISONED_KEYS = {"meridian labs::arr", "halcyon systems::litigation status"}


class Q:
    def __init__(self, label, claim, gold, must_fire, note_):
        self.label = label
        self.claim = claim
        self.gold = gold            # expected token in the final answer
        self.must_fire = must_fire  # True if the gate SHOULD debate this one
        self.note = note_


ROUND1_QUESTIONS = [
    Q("Q1", 'Claim: "Halcyon Systems\'s current ARR is above $50M." '
            f"{POLICY} True or false? Answer 'true' or 'false'.",
      "true", False, "clean filing ($61M) — committed instantly, gate stays shut"),
    Q("Q2", 'Claim: "Meridian Labs\'s flagship product is Atlas." '
            f"{POLICY} True or false? Answer 'true' or 'false'.",
      "true", False, "clean filing — instant answer"),
]

MEMO_QUESTIONS = [
    Q("Q3", 'Claim: "Halcyon Systems\'s current CEO is Okafor." '
            f"{POLICY} True or false? Answer 'true' or 'false'.",
      "true", False, "clean single-filing key — gate stays shut"),
    Q("Q4", 'Claim: "Meridian Labs\'s current CEO is Okafor." '
            f"{POLICY} True or false? Answer 'true' or 'false'.",
      "true", False, "CEO legitimately changed Chen->Okafor by FILING — no debate"),
    Q("Q5", 'Claim: "Meridian Labs\'s ARR is above $50M per its most recent '
            f'filing." {POLICY} True or false? Answer \'true\' or \'false\'.',
      "false", True, "POISONED by the $88M rumor — gate must fire, judge reverts to $47M filing"),
    Q("Q6", 'Claim: "Halcyon Systems is clear of material litigation per its '
            f'filings." {POLICY} True or false? Answer \'true\' or \'false\'.',
      "true", True, "POISONED by the SEC rumor — gate must fire, judge reverts to the filing"),
    Q("Q7", 'GO / NO-GO: "Meridian Labs clears the committee\'s hard revenue '
            'gate of ARR strictly above $50M, based on its filings." '
            f"{POLICY} Answer 'true' (GO) or 'false' (NO-GO).",
      "false", False, "final buy/pass — Q5 already REPAIRED Meridian's ARR on the shared "
                      "board, so this rides the corrected $47M for free => NO-GO, gate shut"),
]


def _key_source(session, key):
    cur = session.board.current(key)
    return cur.source if cur else "?"


def show_board(session, header):
    act(header)
    wm = getattr(session.gate, "wm", None)
    note("current belief board with the world model's per-key P(wrong) risk head "
         "(zero LLM calls):")
    print()
    for key in sorted(session.board._current):
        cur = session.board.current(key)
        risk = wm.wrong_now(session.board, key) if wm else session.board.doubt(key)
        weak = session.board.weak_current(key)
        tag = ""
        if weak:
            tag = red("  <-- POISONED (rumor displaced a filing)")
        elif risk >= 0.5:
            tag = yellow("  <-- elevated risk")
        bar = "#" * int(round(risk * 20))
        colour = red if risk >= 0.5 else green
        print(f"  {key:38s} = {cur.value:26s} "
              f"[{_key_source(session, key):6s}] "
              f"P(wrong)={colour(f'{risk:5.3f}')} {colour(bar)}{tag}")
    print()


def ask_and_report(session, q, running):
    print()
    print(cyan(bold(f"  {q.label}. ")) + cyan(q.claim.split(' Policy:')[0]))
    note(q.note)
    t0 = time.monotonic()
    task = Task(task_id=q.label, family="diligence", context="",
                question=q.claim, gold=q.gold)
    result = session.ask(task)
    dt = time.monotonic() - t0

    trace = result.transcript[0]
    gate = trace["gate"]
    fired = gate["fired"]

    # gate trace
    gcol = red if fired else green
    print("     " + gray("gate: ") + gcol(f"{'FIRED -> debate' if fired else 'CLOSED -> commit now'}")
          + gray(f"  (reason={gate['reason']}, P(wrong)={gate['p_wrong']}, "
                 f"weak_source={gate['weak_current']})"))

    # debate events, if any
    for ev in trace["events"]:
        if ev["kind"] == "challenge":
            print("     " + magenta("skeptic  ") + gray("attacks ")
                  + magenta(ev["key"]) + gray(": " + ev["attack"][:120]))
        elif ev["kind"] == "verdict":
            if ev.get("corrected"):
                print("     " + magenta("judge    ") + gray("rules ")
                      + red("OVERTURNED") + gray(" -> corrects ")
                      + green(ev["key"]) + gray(" to ") + green(str(ev["corrected"])))
            else:
                print("     " + magenta("judge    ") + gray("rules ")
                      + green("UPHELD") + gray(f" ({ev['key']})"))

    # answer + grade
    ans = result.answer.strip()
    ok = q.gold.lower() in ans.lower()
    verdict = green("  PASS") if ok else red("  FAIL")
    print("     " + gray("answer: ") + green(bold(ans)) + verdict
          + gray(f"  (expected '{q.gold}')"))

    # cost (ask-time debate spend; perception is added once at the end)
    cost = result.ledger.cost_usd
    running["ask_cost"] += cost
    running["ask_calls"] += len(result.ledger.calls)
    print("     " + gray(f"cost: ${cost:.4f}  ({len(result.ledger.calls)} calls, "
                         f"{dt:.1f}s)   running answer spend: ${running['ask_cost']:.4f}"))

    # bookkeeping for the end-of-run audit
    running["results"].append({
        "label": q.label, "fired": fired, "must_fire": q.must_fire,
        "pass": ok, "answer": ans, "gold": q.gold})
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maintain", action="store_true",
                    help="repair the poisoned beliefs in a maintenance window "
                         "BEFORE the memo (instead of debating at ask time)")
    args = ap.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print(red("DASHSCOPE_API_KEY not set — run: set -a && source .env && set +a"))
        return 2

    running = {"ask_cost": 0.0, "ask_calls": 0, "results": []}
    session = MajalisSession(seed=0)

    act("MAJALIS INVESTMENT COMMITTEE  —  due-diligence on 2 targets")
    note("A research firm shares ONE versioned belief board across its analyst")
    note("agents. A learned world model scores every belief's P(wrong) and routes")
    note("skeptic->judge debate ONLY where a source-authority rule was violated.")
    note(POLICY)

    # ------------------------------------------------------------------ ACT 1
    act("ACT 1  —  Evidence round 1: Q1-2026 filings ingested ONCE")
    for line in ROUND1:
        print("  " + gray(line))
    session.ingest(ROUND1)
    print()
    note(f"perception cost so far: ${session.ingest_ledger.cost_usd:.4f} "
         f"({len(session.ingest_ledger.calls)} extraction calls) — amortized over every question")

    show_board(session, "ACT 1  —  board after round 1 (all clean, all low-risk)")
    act("ACT 1  —  two instant answers (board is clean, gate stays shut)")
    for q in ROUND1_QUESTIONS:
        ask_and_report(session, q, running)

    # ------------------------------------------------------------------ ACT 2
    act("ACT 2  —  Evidence round 2: real Q2 updates + TWO rumor poisonings")
    for line in ROUND2:
        tag = red("   <-- RUMOR (later-dated, weaker than a filing)") if "Rumor" in line else ""
        print("  " + gray(line) + tag)
    session.ingest(ROUND2)
    show_board(session, "ACT 2  —  board after round 2: risk SPIKES on exactly the poisoned keys")

    # optional headline live mode: repair before questions
    if args.maintain:
        act("ACT 2.5  —  maintenance window: repair the riskiest beliefs off the clock")
        note("session.maintain() runs REAL key-scoped debates on the top-risk beliefs")
        note("between evidence rounds, so the memo can be answered with the gate OFF.")
        repairs = session.maintain(budget=2)
        for r in repairs:
            arrow = (red("OVERTURNED -> ") + green(str(r["corrected"]))) if r["corrected"] \
                else green("upheld")
            print(f"  repaired {magenta(r['key']):40s} risk={r['risk']:.3f}  {arrow}")
        show_board(session, "ACT 2.5  —  board after maintenance (poison cleaned, risk collapsed)")
        session.gate_mode = "never"  # questions now commit instantly

    # ------------------------------------------------------------------ ACT 3
    act("ACT 3  —  Committee memo: decision questions from the corrected board")
    note("The gate fires ONLY on the poisoned decision-critical keys; clean")
    note("questions commit instantly. Watch the skeptic/judge repair by docket.")
    for q in MEMO_QUESTIONS:
        ask_and_report(session, q, running)

    # ------------------------------------------------------------------ MEMO
    act("COMMITTEE MEMO  —  final answers from the adjudicated board")
    for r in running["results"]:
        mark = green("PASS") if r["pass"] else red("FAIL")
        gatetag = (yellow("debated") if r["fired"] else gray("instant"))
        print(f"  [{mark}] {r['label']}  ({gatetag:>18s})  -> "
              + green(bold(r["answer"][:70])))
    print()

    # audit: gate fired on exactly the poisoned/decision keys, all answers correct
    failed = [r["label"] for r in running["results"] if not r["pass"]]
    print(bold("  Audit:"))
    if args.maintain:
        wrong_gate = []  # debate happened in the maintenance window, not at ask time
        print(green("    poison was repaired in the maintenance window; every memo "
                    "question then committed INSTANTLY (gate off) and correctly"))
    else:
        wrong_gate = [r["label"] for r in running["results"] if r["fired"] != r["must_fire"]]
        if wrong_gate:
            print(red(f"    gate routing MISMATCH on: {wrong_gate}"))
        else:
            n_debated = sum(1 for r in running["results"] if r["fired"])
            print(green(f"    gate fired on exactly the {n_debated} poisoned decision keys and "
                        "stayed shut on every clean one"))
            print(green("    (a repaired belief persists on the shared board — later questions "
                        "on it commit for free)"))
    if failed:
        print(red(f"    WRONG final answers on: {failed}"))
    else:
        print(green("    every committee answer is correct post-adjudication"))
    perception_cost = session.ingest_ledger.cost_usd
    perception_calls = len(session.ingest_ledger.calls)
    total_cost = perception_cost + running["ask_cost"]
    total_calls = perception_calls + running["ask_calls"]
    print()
    print(gray(f"    perception + maintenance : ${perception_cost:.4f}  "
               f"({perception_calls} calls, ingested once, amortized over all questions)"))
    print(gray(f"    question answering       : ${running['ask_cost']:.4f}  "
               f"({running['ask_calls']} calls)"))
    print(bold(f"  TOTAL SPEND: ${total_cost:.4f}  "
               f"across {total_calls} Qwen calls  "
               f"(every gate decision itself cost $0 — pure world-model inference)"))
    print()

    return 1 if (wrong_gate or failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
