#!/usr/bin/env bash
# Scripted demo flow for the <3min video. Drives the live API (local or ECS).
#   ./scripts/demo.sh [http://<host>:8080]
# Acts out the full story: evidence arrives -> board builds -> a rumor poisons
# a belief -> the gate fires ONLY there -> the judge corrects it by docket.
set -euo pipefail
HOST="${1:-http://localhost:8080}"
# The shared box gates writes behind X-Majalis-Token (daily-budget bypass).
# Export MAJALIS_LIVE_TOKEN to run live; the header is fed to curl via a
# config on stdin so the token never appears in argv.
j() {
  if [ -n "${MAJALIS_LIVE_TOKEN:-}" ]; then
    printf 'header = "X-Majalis-Token: %s"\n' "$MAJALIS_LIVE_TOKEN" \
      | curl -s -K - -X POST "$HOST/$1" -H 'content-type: application/json' -d "$2" \
      | python3 -m json.tool
  else
    curl -s -X POST "$HOST/$1" -H 'content-type: application/json' -d "$2" | python3 -m json.tool
  fi
}

echo "=== 1. Evidence batch arrives (filings; ingested ONCE into the board)"
j ingest '{"lines": [
  "[Jan 2026] Filing: Meridian Labs'\''s ceo is Chen.",
  "[Feb 2026] Filing: Meridian Labs'\''s headcount is 780.",
  "[Mar 2026] Filing: Borealis AI'\''s flagship product is Kestrel."
]}'

echo "=== 2. The board answers from beliefs — no re-reading, gate stays closed"
j ask '{"question": "Claim: \"Meridian Labs'\''s ceo is currently Chen.\" Policy: filings are authoritative; blog recaps, industry notes and rumors are unreliable and never override a filing. True or false? Answer '\''true'\'' or '\''false'\''."}'

echo "=== 3. An update + a RUMOR arrive (rumor postdates the filing = poison)"
j ingest '{"lines": [
  "[Apr 2026] Filing: Meridian Labs'\''s ceo is Okafor.",
  "[May 2026] Rumor: Meridian Labs'\''s ceo is now Larsson."
]}'

echo "=== 4. Board state: the poisoned belief carries doubt + weak-source flag"
curl -s "$HOST/board" | python3 -m json.tool

echo "=== 5. Same claim again — the WORLD MODEL fires a debate (policy:weak-source),"
echo "===    the skeptic attacks, the judge corrects from the key's docket"
j ask '{"question": "Claim: \"Meridian Labs'\''s ceo is currently Okafor.\" Policy: filings are authoritative; blog recaps, industry notes and rumors are unreliable and never override a filing. True or false? Answer '\''true'\'' or '\''false'\''."}'

echo "=== done — note the gate/events trace and per-question cost in each reply"
