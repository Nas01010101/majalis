import re

from agora.beliefs import parse_date_ord
from agora.bench.stream import make_session

_VALUE_RE = re.compile(r"\b(?:is|as|remains) (.+?)\.$")


def _latest_value(lines: list[str], entity: str, attr: str) -> str | None:
    latest, val = -1, None
    for line in lines:
        if f"{entity}'s {attr}" in line:
            m = _VALUE_RE.search(line)
            d = parse_date_ord(line)
            if m and d > latest:
                latest, val = d, m.group(1)
    return val


def test_stream_gold_consistent_with_dated_lines():
    """Property: at each question, gold must equal comparing the claim to the
    latest-dated assertion in ALL evidence seen so far — echoes (dated in the
    past, arriving late) must never flip it."""
    for seed in range(6):
        lines: list[str] = []
        n_q = 0
        for ev in make_session(seed):
            if ev.kind == "evidence":
                lines += ev.lines
                continue
            n_q += 1
            t = ev.task
            claim = re.search(r'currently ([^."]+)', t.question).group(1).strip()
            latest = _latest_value(lines, t.meta["entity"], t.meta["attr"])
            truth = "true" if claim.lower() == (latest or "").lower() else "false"
            assert truth == t.gold, (seed, t.task_id)
        assert n_q == 16  # 8 steps x 2 questions


def test_stream_deterministic_and_balanced():
    a = make_session(1)
    b = make_session(1)
    assert [e.lines for e in a] == [e.lines for e in b]
    golds = [e.task.gold for e in make_session(2) if e.kind == "question"]
    assert 3 <= sum(g == "true" for g in golds) <= 13


def test_stream_has_churned_questions():
    churned = [e.task.meta["churned"] for e in make_session(0)
               if e.kind == "question"]
    assert any(churned)
