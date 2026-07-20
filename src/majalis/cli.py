"""`majalis` console entry point.

Packaging + one renderer, nothing more: `replay` reads a recorded society
trace (see scripts/gen_sample_trace.py for the format — one JSON record per
line, `type` in {act, evidence, question}) and prints the conflict->consensus
timeline; `demo` execs the existing scripted committee demo
(scripts/demo_company.py); `--version` reports the installed package version.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- terminal styling — mirrors scripts/demo_company.py's palette exactly ---
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


def _act(title: str) -> None:
    bar = "=" * 74
    print("\n" + bold(cyan(bar)))
    print(bold(cyan(f"  {title}")))
    print(bold(cyan(bar)))


def _note(s: str) -> None:
    print(gray("  " + s))


# --- replay ------------------------------------------------------------------

def _render_board(board: list[dict]) -> None:
    for row in board:
        risk = float(row.get("risk", 0.0))
        colour = red if row.get("weak") or risk >= 0.5 else green
        tag = red("  <-- POISONED (weaker source displaced an authoritative one)") \
            if row.get("weak") else (yellow("  <-- elevated risk") if risk >= 0.5 else "")
        bar = "#" * int(round(risk * 20))
        print(f"     {row['key']:38s} = {row['value']:16s} "
              f"[{row.get('source', '?'):6s}] "
              f"P(wrong)={colour(f'{risk:5.3f}')} {colour(bar)}{tag}")


def _render_evidence(rec: dict) -> None:
    _act(f"evidence batch — {len(rec['lines'])} line(s)")
    for line in rec["lines"]:
        print("  " + gray(line))
    print()
    for a in rec["asserts"]:
        outcome = a["outcome"]
        if outcome in ("new", "refresh"):
            tag = green(outcome)
        elif outcome == "superseded":
            tag = yellow("superseded") + gray(" (filing supersedes filing — clean update)")
        elif outcome == "superseded-by-weaker-source":
            tag = red("SUPERSEDED-BY-WEAKER-SOURCE") + gray(" (rumor displaced a filing — poisoned)")
        else:  # stale-echo / conflict
            tag = red(outcome)
        print(f"     assert {a['key']:32s} = {a['value']:12s} [{a['source']}] -> {tag}")
    print()
    _note("belief board after this batch (P(wrong) risk head, zero LLM calls):")
    _render_board(rec["board"])


def _render_question(rec: dict) -> None:
    print()
    print(cyan(bold(f"  {rec['task_id']}. ")) + cyan(rec["question"]))
    gate = rec["gate"]
    fired = gate["fired"]
    gcol = red if fired else green
    print("     " + gray("gate: ") + gcol("FIRED -> debate" if fired else "CLOSED -> commit now")
          + gray(f"  (reason={gate['reason']}, P(wrong)={gate['p_wrong']}, "
                 f"weak_source={gate['weak_current']})"))
    for ev in rec["events"]:
        if ev["kind"] == "challenge":
            print("     " + magenta("skeptic  ") + gray("attacks ")
                  + magenta(ev["key"]) + gray(": " + ev["attack"][:160]))
            for sq in ev.get("sub_questions", []):
                print("     " + gray("         decomposes into binary check: ") + magenta(sq))
        elif ev["kind"] == "verdict":
            if ev.get("corrected"):
                print("     " + magenta("judge    ") + gray("rules ") + red("OVERTURNED")
                      + gray(" -> corrects ") + green(ev["key"]) + gray(" to ")
                      + green(str(ev["corrected"])))
            else:
                print("     " + magenta("judge    ") + gray("rules ") + green("UPHELD")
                      + gray(f" ({ev['key']})"))
    ans, gold = rec["answer"], rec["gold"]
    ok = bool(rec.get("correct", gold.lower() in ans.lower()))
    verdict = green("  PASS") if ok else red("  FAIL")
    print("     " + gray("answer: ") + green(bold(ans)) + verdict + gray(f"  (expected '{gold}')"))
    if rec.get("board"):
        _note("belief board after write-back:")
        _render_board(rec["board"])


def _load_trace(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not valid JSON ({exc})") from exc
    return records


def cmd_replay(args: argparse.Namespace) -> int:
    path = Path(args.trace)
    if not path.exists():
        print(red(f"trace file not found: {path}"), file=sys.stderr)
        return 1
    try:
        records = _load_trace(path)
    except ValueError as exc:
        print(red(str(exc)), file=sys.stderr)
        return 1

    if args.json:
        for rec in records:
            print(json.dumps(rec))
        return 0

    n_questions = n_fired = n_correct = 0
    for rec in records:
        rtype = rec.get("type")
        if rtype == "act":
            _act(rec["title"])
        elif rtype == "evidence":
            _render_evidence(rec)
        elif rtype == "question":
            _render_question(rec)
            n_questions += 1
            n_fired += bool(rec["gate"]["fired"])
            n_correct += bool(rec.get("correct", False))
        else:
            _note(f"(skipping unrecognized record type: {rtype!r})")
        if args.speed > 0:
            time.sleep(args.speed)

    print()
    _act("REPLAY SUMMARY")
    _note(f"{n_questions} question(s), {n_fired} debated, {n_correct}/{n_questions} correct")
    return 0


# --- demo ----------------------------------------------------------------

def cmd_demo(args: argparse.Namespace) -> int:
    demo_path = ROOT / "scripts" / "demo_company.py"
    if not demo_path.exists():
        print(red(f"demo script not found at {demo_path} "
                  "(are you running from an editable install of the repo?)"), file=sys.stderr)
        return 1
    result = subprocess.run([sys.executable, str(demo_path), *args.demo_args])
    return result.returncode


# --- entry point -----------------------------------------------------------

def _version() -> str:
    try:
        return version("majalis")
    except PackageNotFoundError:
        from . import __version__
        return __version__


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="majalis",
                                 description="Majalis: a world-model-modulated debate society.")
    ap.add_argument("--version", action="version", version=f"majalis {_version()}")
    sub = ap.add_subparsers(dest="command")

    p_replay = sub.add_parser("replay", help="render a recorded society trace as a "
                                             "conflict->consensus timeline")
    p_replay.add_argument("trace", help="path to a trace .jsonl file "
                                        "(see examples/sample_trace.jsonl)")
    p_replay.add_argument("--speed", type=float, default=0.0,
                          help="seconds to pause between events (default: 0, no pause)")
    p_replay.add_argument("--json", action="store_true",
                          help="print the raw JSON records instead of the styled render")
    p_replay.set_defaults(func=cmd_replay)

    sub.add_parser("demo", help="run the scripted investment-committee demo "
                              "(needs DASHSCOPE_API_KEY; scripts/demo_company.py); "
                              "any trailing flags are forwarded, e.g. `majalis demo --maintain`")

    return ap


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # `demo` forwards arbitrary flags to scripts/demo_company.py (e.g.
    # --maintain) — argparse's REMAINDER interacts badly with subparsers
    # (a leading "--x" right after the subcommand gets swallowed by the
    # top-level parser), so intercept it before argparse ever sees the tail.
    if argv[:1] == ["demo"]:
        return cmd_demo(argparse.Namespace(demo_args=argv[1:]))

    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "command", None):
        ap.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
