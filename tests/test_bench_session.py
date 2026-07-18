"""Tests for bench/session.py's stress-regime plumbing (--rumor-rate,
--max-debates) and the regime-tagged raw-file naming/collision guard, plus
MajalisSession's max_debates override — the thing the CLI flag actually
threads into. Stubbed chat()/REPLAYS, zero live LLM calls.
"""
from __future__ import annotations

import json

from majalis import society
from majalis.beliefs import parse_date_ord
from majalis.bench import session as bench_session
from majalis.bench.tasks import Task


# --- _raw_path / _normalize_regime: naming + collision guard -----------------

def test_raw_path_default_regime_keeps_legacy_name():
    p_implicit = bench_session._raw_path("majalis-wm", 0, 8, None, None)
    p_explicit_default = bench_session._raw_path("majalis-wm", 0, 8, 0.35, 2)
    assert p_implicit.name == "session_majalis-wm_s0_t8.jsonl"
    assert p_implicit == p_explicit_default  # "not passed" == "passed but == default"


def test_raw_path_nondefault_regime_is_tagged_and_distinct_from_baseline():
    baseline = bench_session._raw_path("majalis-wm", 0, 8, None, None)
    stress_rumor = bench_session._raw_path("majalis-wm", 0, 8, 0.55, None)
    stress_debates = bench_session._raw_path("majalis-wm", 0, 8, None, 1)
    assert stress_rumor != baseline
    assert stress_debates != baseline
    assert stress_rumor != stress_debates
    assert "r0.55" in stress_rumor.name and "d2" in stress_rumor.name  # untouched param still shown
    assert "d1" in stress_debates.name and "r0.35" in stress_debates.name


def test_normalize_regime_collapses_explicit_defaults_to_none():
    assert bench_session._normalize_regime(None, None) == (None, None)
    assert bench_session._normalize_regime(0.35, 2) == (None, None)  # explicit-but-default
    assert bench_session._normalize_regime(0.6, None) == (0.6, None)
    assert bench_session._normalize_regime(None, 1) == (None, 1)


def _fake_raw_row(idx: int) -> str:
    return json.dumps({"task_id": f"fake-{idx}", "gold": "true", "answer": "true",
                       "correct": True, "churned": False, "gate": None,
                       "n_calls": 1, "total_tokens": 10, "completion_tokens": 5,
                       "latency_s": 0.1, "cost_usd": 0.0001}) + "\n"


def test_run_session_arm_stress_regime_never_resumes_from_baseline_cache(tmp_path, monkeypatch):
    """The core collision-guard property: a baseline raw file with the
    RIGHT ROW COUNT must not be mistaken for a completed stress-regime run
    — resume is regime-aware via the filename, not row-count-only."""
    monkeypatch.setattr(bench_session, "RESULTS_DIR", tmp_path)
    baseline_path = bench_session._raw_path("majalis-wm", 0, 1, None, None)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_path.open("w") as fh:
        for i in range(2):  # 2 rows == 2 * n_steps(1) -> "complete" by row count alone
            fh.write(_fake_raw_row(i))

    calls = {"n": 0}

    def _fake_replay(events, seed, max_debates=None):
        calls["n"] += 1
        return []

    monkeypatch.setitem(bench_session.REPLAYS, "majalis-wm", _fake_replay)
    monkeypatch.setattr(bench_session, "make_session",
                        lambda seed, n_steps=8, rumor_rate=0.35: [])

    result = bench_session.run_session_arm("majalis-wm", 0, 1, rumor_rate=0.55)
    assert calls["n"] == 1  # NOT resumed from the baseline cache — ran fresh
    assert result["rumor_rate"] == 0.55
    stress_path = bench_session._raw_path("majalis-wm", 0, 1, 0.55, None)
    assert stress_path != baseline_path
    assert stress_path.exists()  # the stress run wrote its OWN file
    assert baseline_path.read_text().count("fake-") == 2  # baseline cache untouched


def test_run_session_arm_baseline_still_resumes_from_legacy_cache(tmp_path, monkeypatch):
    """Sanity check that the collision guard didn't break the existing
    resume-for-free behavior for the (untagged) baseline regime."""
    monkeypatch.setattr(bench_session, "RESULTS_DIR", tmp_path)
    baseline_path = bench_session._raw_path("majalis-wm", 0, 1, None, None)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_path.open("w") as fh:
        for i in range(2):
            fh.write(_fake_raw_row(i))

    calls = {"n": 0}

    def _fake_replay(events, seed, max_debates=None):
        calls["n"] += 1
        return []

    monkeypatch.setitem(bench_session.REPLAYS, "majalis-wm", _fake_replay)

    result = bench_session.run_session_arm("majalis-wm", 0, 1)  # no overrides = baseline
    assert calls["n"] == 0  # resumed — zero new work
    assert result["resumed"] is True
    assert result["rumor_rate"] is None and result["max_debates"] is None


def test_run_session_arm_rumor_rate_flows_to_make_session(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_session, "RESULTS_DIR", tmp_path)
    seen = {}

    def _fake_make_session(seed, n_steps=8, rumor_rate=0.35):
        seen["rumor_rate"] = rumor_rate
        return []

    monkeypatch.setattr(bench_session, "make_session", _fake_make_session)
    monkeypatch.setitem(bench_session.REPLAYS, "majalis-wm",
                        lambda events, seed, max_debates=None: [])
    bench_session.run_session_arm("majalis-wm", 0, 1, rumor_rate=0.6)
    assert seen["rumor_rate"] == 0.6


def test_run_session_arm_max_debates_flows_to_replays(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_session, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(bench_session, "make_session", lambda seed, n_steps=8, rumor_rate=0.35: [])
    seen = {}

    def _fake_replay(events, seed, max_debates=None):
        seen["max_debates"] = max_debates
        return []

    monkeypatch.setitem(bench_session.REPLAYS, "majalis-wm-plan", _fake_replay)
    bench_session.run_session_arm("majalis-wm-plan", 0, 1, max_debates=1)
    assert seen["max_debates"] == 1


# --- MajalisSession.max_debates: the actual effect the flag threads into ----

def _stub_chat_two_churned_keys(call_log: list[str]):
    def _chat(model, messages, *, ledger, max_tokens=1024, temperature=0.7, seed=None):
        prompt = messages[0]["content"]
        if "Extract every dated factual assertion" in prompt:
            return "[]"
        if "adversarial fact-checker" in prompt:
            call_log.append("skeptic")
            return '{"attack": "x", "sub_questions": ["q1"]}'
        if "You are a judge" in prompt:
            call_log.append("adjudicate")
            return '{"upheld": true, "corrected_value": null, "rationale": "r"}'
        return ('{"answer": "x", "rationale": "r", '
                '"support_keys": ["e::a", "e::b"], "confidence": 0.9}')
    return _chat


def _churned_session(max_debates, call_log, monkeypatch):
    stub = _stub_chat_two_churned_keys(call_log)
    monkeypatch.setattr(society, "chat", stub)
    # AcceptGate.decide() calls wm.sample_disagreement(), which references
    # wm.py's OWN module-level `chat` (bound to llm.chat at import time) —
    # patching society.chat alone does NOT cover it, and leaving it live
    # fires a REAL network call (this bit us once: caught by a hung/failed
    # background test run). Both call sites its stub's catch-all branch
    # returns the same harmless propose-shaped JSON either way.
    import majalis.wm as wm
    monkeypatch.setattr(wm, "chat", stub)
    session = society.MajalisSession(seed=0, gate_mode="always", wm_mode="heuristic",
                                     max_debates=max_debates)
    session.board.assert_fact("e::a", "v1", parse_date_ord("Jan 2025"), source="Filing")
    session.board.assert_fact("e::a", "v2", parse_date_ord("Mar 2025"), source="Filing")
    session.board.assert_fact("e::b", "w1", parse_date_ord("Jan 2025"), source="Filing")
    session.board.assert_fact("e::b", "w2", parse_date_ord("Mar 2025"), source="Filing")
    task = Task(task_id="t0", family="stream", context="", question="q?", gold="x")
    session.ask(task)


def test_majalis_session_max_debates_caps_targets_below_module_default(monkeypatch):
    call_log: list[str] = []
    _churned_session(max_debates=1, call_log=call_log, monkeypatch=monkeypatch)
    assert call_log.count("skeptic") == 1  # capped, not the module default of 2


def test_majalis_session_max_debates_none_falls_back_to_module_default(monkeypatch):
    call_log: list[str] = []
    _churned_session(max_debates=None, call_log=call_log, monkeypatch=monkeypatch)
    assert call_log.count("skeptic") == 2  # society.MAX_DEBATES_PER_TASK


def test_majalis_session_max_debates_default_matches_module_constant():
    session = society.MajalisSession(seed=0)
    assert session.max_debates == society.MAX_DEBATES_PER_TASK
