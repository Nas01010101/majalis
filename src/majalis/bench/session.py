"""Session benchmark runner — the deployment-shaped eval.

    python -m majalis.bench.session --arms single,mad,majalis --seeds 0,1,2

Every arm sees the SAME event sequence. Baselines re-read the full stream-
so-far per question (that is the honest no-memory best practice); the majalis
arm ingests incrementally and answers from its persistent board.

"majalis" (heuristic gate) and "majalis-wm" (learned gate) are pinned to
their gate mode by the arm name itself (REPLAYS below), not by an ambient
MAJALIS_WM env var — a fresh clone reproduces either row exactly with no
undocumented setup. MAJALIS_WM, if set, still overrides for ad-hoc
re-tuning (logged loudly by wmnet.load_wm) — leave it unset to reproduce
results/session_summary.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..config import MODEL_STRONG
from ..llm import Ledger, chat
from ..society import MAX_DEBATES_PER_TASK, MajalisSession
from .arms import _extract, vanilla_mad
from .stats import fmt_acc
from .stream import make_session
from .tasks import Task, grade

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"

_RULES = ("Think step by step, then give your final answer on the last "
          "line as: ANSWER: <short answer>")

# Stress-regime defaults — MUST match bench.stream.make_session's own
# rumor_rate default and society.MAX_DEBATES_PER_TASK exactly, so "no
# override passed" and "override passed but equal to the default" are
# indistinguishable and both resolve to the LEGACY (untagged) raw filename.
_DEFAULT_RUMOR_RATE = 0.35
_DEFAULT_MAX_DEBATES = MAX_DEBATES_PER_TASK


def _replay_single(events, seed: int, *, max_debates: int | None = None) -> list[dict]:
    lines: list[str] = []
    records = []
    for ev in events:
        if ev.kind == "evidence":
            lines += ev.lines
            continue
        ledger = Ledger()
        prompt = (f"Evidence:\n" + "\n".join(lines)
                  + f"\n\n{ev.task.question}\n\n{_RULES}")
        out = chat(MODEL_STRONG, [{"role": "user", "content": prompt}],
                   ledger=ledger, temperature=0.0, seed=seed)
        records.append({"task": ev.task, "answer": _extract(out), "ledger": ledger})
    return records


def _replay_mad(events, seed: int, *, max_debates: int | None = None) -> list[dict]:
    lines: list[str] = []
    records = []
    for ev in events:
        if ev.kind == "evidence":
            lines += ev.lines
            continue
        task = Task(task_id=ev.task.task_id, family="stream",
                    context="\n".join(lines), question=ev.task.question,
                    gold=ev.task.gold)
        result = vanilla_mad(task, seed=seed)
        records.append({"task": ev.task, "answer": result.answer,
                        "ledger": result.ledger})
    return records


def _replay_majalis(events, seed: int, gate_mode: str = "wm",
                    wm_mode: str | None = None,
                    max_debates: int | None = None) -> list[dict]:
    session = MajalisSession(seed=seed, gate_mode=gate_mode, wm_mode=wm_mode,
                             max_debates=max_debates)
    records = []
    for ev in events:
        if ev.kind == "evidence":
            session.ingest(ev.lines)
            continue
        result = session.ask(ev.task)
        records.append({"task": ev.task, "answer": result.answer,
                        "ledger": result.ledger,
                        "gate": result.transcript[0].get("gate")})
    # Perception cost is real: spread it across the session's questions.
    if records:
        records[0]["ingest_ledger"] = session.ingest_ledger
    return records


# Every REPLAYS entry accepts the same (events, seed, *, max_debates=None)
# shape so run_session_arm can call REPLAYS[arm](events, seed,
# max_debates=max_debates) uniformly; single/mad simply ignore it (no
# per-question debate-budget concept for those arms).
REPLAYS = {
    "single": _replay_single,
    "mad": _replay_mad,
    # The arm name IS the gate mode — pinned in code, not by an ambient
    # MAJALIS_WM env var, so a fresh clone reproduces either row exactly
    # without knowing which env setting produced the shipped numbers.
    # MAJALIS_WM, if set, still overrides (logged loudly by wmnet.load_wm)
    # for ad-hoc re-tuning — leave it unset to reproduce the shipped numbers.
    "majalis": lambda ev, seed, max_debates=None: _replay_majalis(
        ev, seed, wm_mode="heuristic", max_debates=max_debates),
    # Ablation: same (heuristic) board scoring, debates never fire —
    # isolates what debate adds on top of the heuristic-gate row above.
    "majalis-nodebate": lambda ev, seed, max_debates=None: _replay_majalis(
        ev, seed, gate_mode="never", wm_mode="heuristic", max_debates=max_debates),
    # Learned world model (trained heads + stacker; wm.py auto-loads
    # data/wm_weights.json). A separate arm name means separate raw files —
    # the frozen heuristic-gate numbers are never clobbered.
    "majalis-wm": lambda ev, seed, max_debates=None: _replay_majalis(
        ev, seed, wm_mode="learned", max_debates=max_debates),
    # Planned (two-branch argmax-utility) gate — see src/majalis/wm_plan.py.
    # Same belief board, same debate mechanics; only the decision rule
    # differs from majalis-wm above (threshold-on-one-branch-risk vs
    # argmax-over-two-predicted-branch-utilities), per the reactive-vs-
    # planned ablation design (design_track3_worldmodel.md §2.2). Purely
    # additive: majalis-wm's code path above is untouched.
    "majalis-wm-plan": lambda ev, seed, max_debates=None: _replay_majalis(
        ev, seed, gate_mode="plan", wm_mode="learned", max_debates=max_debates),
}


def _normalize_regime(rumor_rate: float | None,
                      max_debates: int | None) -> tuple[float | None, int | None]:
    """Canonicalize regime params: a value equal to the default collapses to
    None, so 'not passed' and 'passed but equal to the default' are
    indistinguishable everywhere (raw filename, summary key, aggregation) —
    otherwise `--rumor-rate 0.35` (the default, spelled out explicitly)
    would resume the SAME raw file but mint a SECOND, differently-keyed
    session_summary.json row for it, double-counting that seed."""
    eff_rumor = _DEFAULT_RUMOR_RATE if rumor_rate is None else rumor_rate
    eff_debates = _DEFAULT_MAX_DEBATES if max_debates is None else max_debates
    norm_rumor = None if eff_rumor == _DEFAULT_RUMOR_RATE else eff_rumor
    norm_debates = None if eff_debates == _DEFAULT_MAX_DEBATES else eff_debates
    return norm_rumor, norm_debates


def _raw_path(arm: str, seed: int, n_steps: int, rumor_rate: float | None,
             max_debates: int | None) -> Path:
    """Regime-tagged filename for a non-default stress regime, so a stress
    run can NEVER collide with (or silently "resume" from) a cached baseline
    file — _resume_summary's check is row-COUNT-only, not regime-aware, so
    without this tag a stress run reading a same-shaped baseline file would
    "resume" with the WRONG regime's numbers instead of running fresh.
    Default regime (both params unset or equal to the defaults) keeps the
    exact legacy filename, so every existing results/raw/session_*.jsonl
    cache still resumes for free."""
    norm_rumor, norm_debates = _normalize_regime(rumor_rate, max_debates)
    tag = "" if norm_rumor is None and norm_debates is None else (
        f"_r{_DEFAULT_RUMOR_RATE if norm_rumor is None else norm_rumor:g}"
        f"_d{_DEFAULT_MAX_DEBATES if norm_debates is None else norm_debates}")
    return RESULTS_DIR / "raw" / f"session_{arm}_s{seed}_t{n_steps}{tag}.jsonl"


def _resume_summary(arm: str, seed: int, n_steps: int, raw_path: Path,
                    rumor_rate: float | None, max_debates: int | None) -> dict | None:
    """A cell whose raw file already holds every question is done — reload
    its summary instead of re-spending API calls (kills are cheap)."""
    if not raw_path.exists():
        return None
    rows = [json.loads(l) for l in raw_path.read_text().splitlines() if l]
    if len(rows) != 2 * n_steps:
        return None
    tokens = sum(r["total_tokens"] + (r.get("ingest", {}).get("total_tokens", 0))
                 for r in rows)
    cost = sum(r["cost_usd"] + (r.get("ingest", {}).get("cost_usd", 0))
               for r in rows)
    norm_rumor, norm_debates = _normalize_regime(rumor_rate, max_debates)
    return {"arm": arm, "seed": seed, "steps": n_steps, "n": len(rows),
            "correct": sum(r["correct"] for r in rows), "tokens": tokens,
            "cost_usd": round(cost, 4), "latency_s": 0.0, "resumed": True,
            "rumor_rate": norm_rumor, "max_debates": norm_debates}


def run_session_arm(arm: str, seed: int, n_steps: int = 8, *,
                    rumor_rate: float | None = None,
                    max_debates: int | None = None) -> dict:
    raw_path = _raw_path(arm, seed, n_steps, rumor_rate, max_debates)
    resumed = _resume_summary(arm, seed, n_steps, raw_path, rumor_rate, max_debates)
    if resumed:
        print(f"  (resumed from {raw_path.name})", flush=True)
        return resumed
    events = make_session(seed, n_steps=n_steps,
                          rumor_rate=(_DEFAULT_RUMOR_RATE if rumor_rate is None
                                     else rumor_rate))
    t0 = time.monotonic()
    records = REPLAYS[arm](events, seed, max_debates=max_debates)
    latency = time.monotonic() - t0
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    correct = tokens = 0
    cost = 0.0
    with raw_path.open("w") as fh:
        for rec in records:
            ok = grade(rec["task"], rec["answer"])
            correct += ok
            tokens += rec["ledger"].total_tokens
            cost += rec["ledger"].cost_usd
            extra = {}
            if "ingest_ledger" in rec:
                tokens += rec["ingest_ledger"].total_tokens
                cost += rec["ingest_ledger"].cost_usd
                extra["ingest"] = rec["ingest_ledger"].as_dict()
            fh.write(json.dumps({
                "task_id": rec["task"].task_id, "gold": rec["task"].gold,
                "answer": rec["answer"], "correct": ok,
                "churned": rec["task"].meta.get("churned"),
                "gate": rec.get("gate"), **rec["ledger"].as_dict(), **extra,
            }) + "\n")
    norm_rumor, norm_debates = _normalize_regime(rumor_rate, max_debates)
    return {"arm": arm, "seed": seed, "steps": n_steps, "n": len(records),
            "correct": correct, "tokens": tokens, "cost_usd": round(cost, 4),
            "latency_s": round(latency, 1),
            "rumor_rate": norm_rumor, "max_debates": norm_debates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="single,majalis")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", default="8", help="comma list of stream lengths")
    ap.add_argument("--rumor-rate", type=float, default=None,
                    help="override bench.stream.make_session's rumor_rate "
                         f"(default {_DEFAULT_RUMOR_RATE}); a non-default value "
                         "gets a regime-tagged raw filename + summary key so it "
                         "can never collide with the cached baseline")
    ap.add_argument("--max-debates", type=int, default=None,
                    help="override MajalisSession's per-question debate budget "
                         f"(default {_DEFAULT_MAX_DEBATES}, society.MAX_DEBATES_PER_TASK); "
                         "same regime-tagging as --rumor-rate")
    args = ap.parse_args()

    summaries = []
    for n_steps in (int(t) for t in args.steps.split(",")):
        for seed in (int(s) for s in args.seeds.split(",")):
            for arm in args.arms.split(","):
                print(f"session: {arm} seed={seed} steps={n_steps} "
                      f"rumor_rate={args.rumor_rate} max_debates={args.max_debates} ...",
                      flush=True)
                summaries.append(run_session_arm(arm, seed, n_steps,
                                                 rumor_rate=args.rumor_rate,
                                                 max_debates=args.max_debates))

    out = RESULTS_DIR / "session_summary.json"
    existing = json.loads(out.read_text()) if out.exists() else []
    # Latest run wins per (arm, seed, steps, rumor_rate, max_debates) — the
    # regime fields default to None (via .get) for pre-existing rows that
    # predate this flag, which is exactly the baseline regime, so old
    # baseline entries and new baseline entries still collide/merge as
    # before; only a stress-regime run gets its OWN summary slot.
    def key(s):
        return (s["arm"], s["seed"], s.get("steps", 8),
                s.get("rumor_rate"), s.get("max_debates"))
    merged = {key(s): s for s in existing}
    merged.update({key(s): s for s in summaries})
    out.write_text(json.dumps(list(merged.values()), indent=2))

    by_cell: dict[tuple, dict] = {}
    for s in merged.values():
        agg = by_cell.setdefault(
            (s["arm"], s.get("steps", 8), s.get("rumor_rate"), s.get("max_debates")),
            {"n": 0, "correct": 0, "tokens": 0, "cost": 0.0})
        agg["n"] += s["n"]
        agg["correct"] += s["correct"]
        agg["tokens"] += s["tokens"]
        agg["cost"] += s["cost_usd"]
    print(f"\n{'arm':<8} {'steps':>5} {'regime':<14} {'accuracy (Wilson 95%)':<28} "
          f"{'tok/q':>8} {'$/q':>9}")
    for (arm, steps, rumor_rate, max_debates), a in sorted(
            by_cell.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        regime = "baseline" if rumor_rate is None and max_debates is None else (
            f"r={rumor_rate},d={max_debates}")
        print(f"{arm:<8} {steps:>5} {regime:<14} {fmt_acc(a['correct'], a['n']):<28} "
              f"{a['tokens'] // max(1, a['n']):>8} "
              f"{a['cost'] / max(1, a['n']):>9.5f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
