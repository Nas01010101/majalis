"""Tests for the `majalis` CLI (src/majalis/cli.py) — no live LLM calls, no
subprocess spawn of the real demo. Covers replay's styled render, --json
round-trip, missing-file handling, and demo's subprocess forwarding.
"""
from __future__ import annotations

import json

import pytest

from majalis import cli

TRACE = [
    {"type": "act", "title": "TEST RUN"},
    {"type": "evidence",
     "lines": ["[Jan 2026] Filing: Acme's ceo is Doe."],
     "asserts": [{"key": "acme::ceo", "value": "doe", "source": "Filing", "outcome": "new"}],
     "board": [{"key": "acme::ceo", "value": "doe", "source": "Filing", "risk": 0.1,
               "weak": False, "churn": 0}]},
    {"type": "evidence",
     "lines": ["[Mar 2026] Rumor: Acme's ceo is Roe."],
     "asserts": [{"key": "acme::ceo", "value": "roe", "source": "Rumor",
                 "outcome": "superseded-by-weaker-source"}],
     "board": [{"key": "acme::ceo", "value": "roe", "source": "Rumor", "risk": 0.9,
               "weak": True, "churn": 1}]},
    {"type": "question", "task_id": "Q1",
     "question": "Claim: \"Acme's ceo is Roe.\" True or false?",
     "gold": "false", "answer": "false", "correct": True,
     "gate": {"fired": True, "p_wrong": 0.9, "disagreement": 0.0, "max_doubt": 0.9,
              "weak_current": True, "reason": "policy:weak-source"},
     "events": [
         {"kind": "challenge", "key": "acme::ceo", "attack": "rumor displaced a filing",
          "sub_questions": ["Is the source a Filing or a Rumor?",
                            "Is there a later Filing?"]},
         {"kind": "verdict", "key": "acme::ceo", "upheld": False, "corrected": "doe"},
     ],
     "board": [{"key": "acme::ceo", "value": "doe", "source": "debate", "risk": 0.4,
               "weak": False, "churn": 2}]},
]


def _write_trace(tmp_path, records=TRACE):
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


# --- replay: styled render --------------------------------------------------

def test_replay_reports_supersession_and_gate_and_subquestions(tmp_path, capsys):
    path = _write_trace(tmp_path)
    rc = cli.main(["replay", str(path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SUPERSEDED-BY-WEAKER-SOURCE" in out
    assert "policy:weak-source" in out
    assert "FIRED -> debate" in out
    assert "Is the source a Filing or a Rumor?" in out
    assert "OVERTURNED" in out
    assert "PASS" in out


def test_replay_summary_counts_questions_and_debates(tmp_path, capsys):
    path = _write_trace(tmp_path)
    cli.main(["replay", str(path)])
    out = capsys.readouterr().out
    assert "1 question(s), 1 debated, 1/1 correct" in out


# --- replay: --json round trip ----------------------------------------------

def test_replay_json_round_trips_every_record(tmp_path, capsys):
    path = _write_trace(tmp_path)
    rc = cli.main(["replay", str(path), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    lines = [json.loads(l) for l in out.strip().splitlines()]
    assert lines == TRACE


# --- replay: error handling --------------------------------------------------

def test_replay_missing_file_is_a_clean_error(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.jsonl"
    rc = cli.main(["replay", str(missing)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "not found" in err


def test_replay_malformed_json_line_is_a_clean_error(tmp_path, capsys):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n")
    rc = cli.main(["replay", str(path)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "not valid JSON" in err


# --- the shipped example trace ------------------------------------------------

def test_sample_trace_replays_with_a_supersession_and_a_gate_fire(capsys):
    sample = cli.ROOT / "examples" / "sample_trace.jsonl"
    assert sample.exists(), "examples/sample_trace.jsonl must be committed"
    rc = cli.main(["replay", str(sample)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SUPERSEDED-BY-WEAKER-SOURCE" in out
    assert "FIRED -> debate" in out
    assert "OVERTURNED" in out


# --- demo: subprocess forwarding, no real spawn -----------------------------

def test_demo_forwards_extra_args_and_returncode(monkeypatch):
    calls = []

    class _Result:
        returncode = 3

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Result()

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    rc = cli.main(["demo", "--maintain"])
    assert rc == 3
    assert calls[0][-2:] == [str(cli.ROOT / "scripts" / "demo_company.py"), "--maintain"] \
        or calls[0][-1] == "--maintain"  # tolerate REMAINDER placement


def test_demo_missing_script_is_a_clean_error(monkeypatch):
    monkeypatch.setattr(cli, "ROOT", cli.ROOT / "nonexistent-root")
    rc = cli.main(["demo"])
    assert rc == 1


# --- --version -----------------------------------------------------------

def test_version_flag_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "majalis" in capsys.readouterr().out


def test_no_command_prints_help_and_exits_nonzero(capsys):
    rc = cli.main([])
    out = capsys.readouterr().out
    assert rc == 1
    assert "usage: majalis" in out
