"""Real end-to-end smoke test against the DEPLOYED Majalis service.

    python scripts/e2e_live.py [--base http://47.237.187.157:8080]

Spends ~$0.05 of real Qwen calls (2 ingests + 3 asks) through the live
box — the whole pipeline: extractor -> board -> learned WM -> gate ->
debate -> write-back. Token read from <repo>/.env (MAJALIS_LIVE_TOKEN).
Exit 0 only if every invariant holds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _token() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("MAJALIS_LIVE_TOKEN="):
            return line.split("=", 1)[1].strip()
    sys.exit("MAJALIS_LIVE_TOKEN not found in .env")


def _call(base: str, path: str, payload: dict, token: str) -> dict:
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Majalis-Token": token})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


CHECKS: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://47.237.187.157:8080")
    args = ap.parse_args()
    token = _token()
    sid = f"e2e-{int(time.time())}"
    cost = 0.0

    with urllib.request.urlopen(args.base + "/healthz", timeout=10) as r:
        check("healthz", json.load(r).get("ok") is True)

    # 1. clean filings -> low risk everywhere
    ev = _call(args.base, "/ingest", {"session_id": sid, "lines": [
        "[Jan 2026] Filing: Vega Dynamics's CEO is Mara Voss.",
        "[Feb 2026] Filing: Vega Dynamics's headcount is 480.",
    ]}, token)["event"]
    cost += ev["cost_usd"]
    check("ingest: filings asserted", len(ev["asserts"]) >= 2,
          str([(a["key"], a["outcome"]) for a in ev["asserts"]]))
    check("ingest: clean board scores low", all(b["wrong_now"] < 0.35 for b in ev["board"]),
          str([(b["key"], b["wrong_now"]) for b in ev["board"]]))

    # 2. unpoisoned ask -> gate must NOT fire (this is the cost story)
    q = _call(args.base, "/ask", {"session_id": sid, "question":
        'Claim: "Vega Dynamics\'s CEO is currently Mara Voss." Policy: filings are '
        'authoritative; rumors are unreliable and never override a filing. True or false?'},
        token)["event"]
    cost += q["cost_usd"]
    check("ask(clean): gate commits, no debate", not q["gate"]["fired"],
          f"p_wrong={q['gate']['p_wrong']} reason={q['gate']['reason']}")
    check("ask(clean): answer is true", "true" in q["answer"].lower(), q["answer"])

    # 3. rumor displaces a filing -> WM must flag it hard
    ev2 = _call(args.base, "/ingest", {"session_id": sid, "lines": [
        "[Mar 2026] Rumor: vega dynamics's ceo is Del Kray.",
    ]}, token)["event"]
    cost += ev2["cost_usd"]
    poisoned = next(b for b in ev2["board"] if b["key"].endswith("::ceo"))
    check("ingest(rumor): weak source flagged", poisoned["weak"], str(poisoned))
    check("ingest(rumor): wrong_now spikes", poisoned["wrong_now"] > 0.7,
          f"wrong_now={poisoned['wrong_now']}")

    # 4. poisoned ask -> gate fires, debate corrects, board written back
    q2 = _call(args.base, "/ask", {"session_id": sid, "question":
        'Claim: "Vega Dynamics\'s CEO is currently Del Kray." Policy: filings are '
        'authoritative; rumors are unreliable and never override a filing. True or false?'},
        token)["event"]
    cost += q2["cost_usd"]
    kinds = [t["kind"] for t in q2["events"]]
    check("ask(poisoned): gate fires", q2["gate"]["fired"], q2["gate"]["reason"])
    check("ask(poisoned): full debate ran", "challenge" in kinds and "verdict" in kinds, str(kinds))
    check("ask(poisoned): answer is false", "false" in q2["answer"].lower(), q2["answer"])
    ceo = next(b for b in q2["board"] if b["key"].endswith("::ceo"))
    check("write-back: board corrected to filing value", "mara voss" in ceo["value"].lower(),
          ceo["value"])
    check("write-back: risk collapses after correction", ceo["wrong_now"] < 0.35,
          f"wrong_now={ceo['wrong_now']}")

    # 5. ask again on the healed board -> gate must be quiet again
    q3 = _call(args.base, "/ask", {"session_id": sid, "question":
        'Claim: "Vega Dynamics\'s CEO is currently Mara Voss." Policy: filings are '
        'authoritative; rumors are unreliable and never override a filing. True or false?'},
        token)["event"]
    cost += q3["cost_usd"]
    check("ask(healed): gate commits again", not q3["gate"]["fired"],
          f"p_wrong={q3['gate']['p_wrong']}")
    check("ask(healed): answer is true", "true" in q3["answer"].lower(), q3["answer"])

    failed = [n for n, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} invariants hold · "
          f"session {sid} · total spend ${cost:.4f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
