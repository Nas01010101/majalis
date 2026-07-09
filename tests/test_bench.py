import re

from agora.beliefs import parse_date_ord
from agora.bench.stats import wilson_ci
from agora.bench.tasks import grade, load_tasks


def test_churn_gold_matches_latest_dated_evidence():
    """Property: gold must equal comparing the claim against the latest-dated
    line about the asked (entity, attr). Guards the generator's date math —
    a naive month-cycling version produced echoes dated after later filings."""
    for t in load_tasks("churn", 150, seed=3):
        ent, attr = t.meta["entity"], t.meta["attr"]
        latest, latest_val = -1, None
        for line in t.context.splitlines():
            if f"{ent}'s {attr}" in line:
                m = re.search(r"\b(?:is|as) (.+?)\.$", line)
                d = parse_date_ord(line)
                if m and d > latest:
                    latest, latest_val = d, m.group(1)
        claim = re.search(r'currently ([^."]+)', t.question).group(1).strip()
        truth = "true" if claim.lower() == (latest_val or "").lower() else "false"
        assert truth == t.gold, t.task_id


def test_tasks_deterministic():
    a = load_tasks("churn", 10, seed=42)
    b = load_tasks("churn", 10, seed=42)
    assert [(t.context, t.question, t.gold) for t in a] == \
        [(t.context, t.question, t.gold) for t in b]
    assert load_tasks("churn", 10, seed=7)[0].context != a[0].context


def test_churn_gold_balance():
    tasks = load_tasks("churn", 200, seed=0)
    trues = sum(t.gold == "true" for t in tasks)
    assert 60 <= trues <= 140  # roughly balanced


def test_multihop_gold_is_surname():
    for t in load_tasks("multihop", 20, seed=1):
        assert t.gold.isalpha()


def test_grade_answer_line():
    t = load_tasks("churn", 1, seed=0)[0]
    assert grade(t, f"blah blah\nANSWER: {t.gold}") or grade(t, t.gold)
    wrong = "false" if t.gold == "true" else "true"
    assert not grade(t, wrong)


def test_wilson_bounds():
    lo, hi = wilson_ci(45, 50)
    assert 0.78 < lo < 0.9 < hi <= 1.0
    assert wilson_ci(0, 0) == (0.0, 1.0)
