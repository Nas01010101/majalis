#!/usr/bin/env python3
"""Mine (skip, debate) paired outcomes for the action-conditioned world model.

    .venv/bin/python scripts/gen_action_wm_dataset.py \
        --seeds 100:150 --out data/action_wm_train.jsonl
    .venv/bin/python scripts/gen_action_wm_dataset.py \
        --seeds 150:180 --out data/action_wm_heldout.jsonl

Shared-board-once design (design_track3_worldmodel.md §2.1). Per question,
extract_facts + propose run EXACTLY ONCE — perception/proposal is shared
between the two labels below, avoiding the board-divergence risk of running
two separate pipeline invocations (the LLM API `seed` param is documented
best-effort, not exact-reproducing, per docs/paper/majalis.md's own
Reproducibility section — two independent runs of extract_facts could build
subtly different boards, breaking the pairing exactly where it matters most,
on rumor/churn cases).

Two labels come off that single shared pass:
  skip_correct   — grade(task, PRE-debate proposal.answer): the closed-form,
                   zero-extra-cost outcome of committing without debate,
                   using the generator's own ground truth (task.gold).
  debate_correct — grade(task, POST-debate proposal.answer): a REAL
                   skeptic_challenge + adjudicate + re-propose run, exactly
                   society.py's debate branch, targeting the single most-
                   doubted support key (wm_plan.target_key — the SAME key
                   PlannedGate would score at serve time, so there is no
                   train/serve skew in which key gets featurized).

Documented design choice: the debate's write-back is applied to the SAME
live board object the session continues on (not a scratch copy) — every
mined session is, structurally, an "always debate" (majalis-nogate-style)
trajectory. This (a) matches the design doc's literal instruction to run
the debate branch "on the same shared board object", (b) is the richest
trajectory to mine along (challenging every question surfaces more
corrections than a mixed/gated trajectory would), and (c) does not
compromise the skip_correct label, which only needs the PRE-debate
proposal for THIS question — unaffected by what the board does afterward.

Seeds 100-149 = action-model TRAIN band, 150-179 = HELD-OUT band (both
sub-ranges of the existing 100-999 "gate calibration" band already
documented as disjoint from eval in docs/paper/majalis.md). Eval seeds
0-99 are UNTOUCHED by this script (asserted below — never mine from them).

Budget discipline mirrors scripts/gsm8k_bench.py's BudgetTracker pattern,
in a SEPARATE state file (results/action_wm_mining_spend_state.json) so
this script's spend is never conflated with the GSM8K budget. Mining is
per-seed atomic: a killed run may lose at most the seed in progress (no
partial-seed rows are ever persisted), and already-completed seeds are
skipped on resume for free.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from majalis.beliefs import BeliefBoard, parse_date_ord  # noqa: E402
from majalis.bench.stream import make_session  # noqa: E402
from majalis.bench.tasks import grade  # noqa: E402
from majalis.config import MODEL_FAST, MODEL_MID, MODEL_STRONG  # noqa: E402
from majalis.llm import Ledger  # noqa: E402
from majalis.society import adjudicate, extract_facts, propose, skeptic_challenge  # noqa: E402
from majalis.wm_plan import target_key  # noqa: E402
from majalis.wmfeat_action import TouchTracker, key_features_action  # noqa: E402
from majalis.wmnet import load_wm  # noqa: E402

SPEND_STATE_PATH = ROOT / "results" / "action_wm_mining_spend_state.json"


def _load_spend_state() -> float:
    if SPEND_STATE_PATH.exists():
        return json.loads(SPEND_STATE_PATH.read_text()).get("cost_usd", 0.0)
    return 0.0


def _save_spend_state(cost: float) -> None:
    SPEND_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPEND_STATE_PATH.write_text(json.dumps({"cost_usd": round(cost, 6)}))


def mine_seed(seed: int, n_steps: int = 8) -> tuple[list[dict], float]:
    """Mines one stream session. Returns (rows, cost_usd). Raises on any
    error WITHOUT having mutated any on-disk state — the caller decides
    whether to persist partial in-memory rows (it does not: see module
    docstring on per-seed atomicity)."""
    assert seed >= 100, "mining seeds live at 100+; eval uses 0-99"
    board = BeliefBoard()
    touch = TouchTracker()
    wm = load_wm()  # v1's free wrong_now head, for target-key selection only
    ledger = Ledger()  # one shared ledger: ingest + every question this seed
    rows: list[dict] = []

    for ev in make_session(seed, n_steps=n_steps):
        if ev.kind == "evidence":
            for fact in extract_facts("\n".join(ev.lines), ledger, MODEL_FAST):
                try:
                    key = BeliefBoard.make_key(str(fact["entity"]), str(fact["attribute"]))
                    board.assert_fact(key, str(fact["value"]),
                                      parse_date_ord(str(fact.get("date", ""))),
                                      source=str(fact.get("source", "")))
                except (KeyError, TypeError):
                    continue
            continue

        task = ev.task
        # --- ONE shared perception+proposal call for this question -------
        proposal = propose(task, board, ledger, MODEL_STRONG)
        support_keys = [k for k in proposal.support_keys if board.current(k) is not None]
        skip_correct = int(grade(task, proposal.answer))

        key = target_key(board, support_keys, wm=wm)
        if key is None:
            # Nothing on the board this proposal rests on — no debate is
            # possible, so there is no (skip, debate) PAIR to mine here.
            touch.record(proposal.support_keys)
            continue

        touch_count_before = touch.touches.get(key, 0)
        questions_before = touch.n_questions
        x = key_features_action(board, key, touch_count_before, questions_before)

        # --- REAL debate branch, on the SAME board object -----------------
        challenge = skeptic_challenge(task, board, key, proposal, ledger, MODEL_MID)
        verdict = adjudicate(task, board, proposal, challenge, ledger, MODEL_STRONG)
        if not verdict.upheld and verdict.corrected_value:
            board.assert_fact(key, verdict.corrected_value, board._now_ord + 1, source="debate")
            note = f"{key}: CORRECTED to {verdict.corrected_value}"
        else:
            note = f"{key}: belief UPHELD as stated"
        debate_note = ("Adjudication results for the challenged beliefs:\n"
                       f"- {note}\nAnswer strictly from the belief state below.\n\n")
        debate_proposal = propose(task, board, ledger, MODEL_STRONG,
                                  correction_note=debate_note)
        debate_correct = int(grade(task, debate_proposal.answer))

        rows.append({
            "uid": f"stream:{seed}:{task.task_id}", "seed": seed,
            "task_id": task.task_id, "key": key, "x": x,
            "touch_rate": x[-1], "skip_correct": skip_correct,
            "debate_correct": debate_correct,
        })
        touch.record(proposal.support_keys)

    return rows, ledger.cost_usd


def _done_seeds(state_path: Path) -> set[int]:
    if not state_path.exists():
        return set()
    return set(json.loads(state_path.read_text()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="lo:hi (hi exclusive), e.g. 100:150")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-steps", type=int, default=8)
    ap.add_argument("--budget-cap", type=float, default=8.0,
                    help="hard stop, cumulative USD across all invocations "
                         "(tracked in results/action_wm_mining_spend_state.json, "
                         "SEPARATE from the GSM8K budget)")
    ap.add_argument("--pilot", type=int, default=0,
                    help="mine only this many seeds, print a projected "
                         "full-range cost, then stop (no --budget-cap check)")
    args = ap.parse_args()

    lo, hi = (int(v) for v in args.seeds.split(":"))
    assert lo >= 100, "mining seeds live at 100+; eval uses 0-99"
    seeds = list(range(lo, hi))
    if args.pilot:
        seeds = seeds[:args.pilot]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = out_path.with_suffix(out_path.suffix + ".doneseeds.json")
    done = _done_seeds(state_path)

    spend = _load_spend_state()
    n_rows_total = 0
    t0 = time.time()
    for seed in seeds:
        if seed in done:
            print(f"seed {seed}: already mined, skip", flush=True)
            continue
        if not args.pilot and spend >= args.budget_cap:
            print(f"BUDGET CAP reached (${spend:.4f} >= ${args.budget_cap}) "
                  f"before seed {seed} — stopping.", flush=True)
            break
        try:
            rows, cost = mine_seed(seed, n_steps=args.n_steps)
        except Exception as exc:  # noqa: BLE001
            print(f"seed {seed}: FAILED ({exc!r}) — no partial rows persisted, stopping.",
                  file=sys.stderr, flush=True)
            break
        spend += cost
        _save_spend_state(spend)
        with out_path.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        done.add(seed)
        state_path.write_text(json.dumps(sorted(done)))
        n_rows_total += len(rows)
        print(f"seed {seed}: {len(rows)} rows, seed_cost=${cost:.4f}, "
              f"cumulative=${spend:.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"\ndone: {n_rows_total} new rows -> {out_path}, "
          f"cumulative spend ${spend:.4f}, wall {elapsed:.1f}s")
    if args.pilot and seeds:
        per_seed = spend / len(seeds) if spend else 0.0
        full_lo, full_hi = (int(v) for v in args.seeds.split(":"))
        n_full = full_hi - full_lo
        print(f"PILOT projection: {len(seeds)} seeds cost ${spend:.4f} "
              f"(${per_seed:.4f}/seed) -> full range [{full_lo}:{full_hi}) "
              f"({n_full} seeds) projects to ${per_seed * n_full:.2f}")


if __name__ == "__main__":
    main()
