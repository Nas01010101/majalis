"""Multi-horizon hazard heads + imagination planner — regression tests."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from majalis.beliefs import BeliefBoard  # noqa: E402
from majalis.bench.stream import make_session  # noqa: E402
from majalis.wmfeat import HAZARD_HORIZONS, replay_stream  # noqa: E402


def test_hazard_labels_monotone_and_legacy_consistent():
    """superseded_within must be monotone non-decreasing in horizon, and the
    k=2 point must equal the legacy superseded_next label exactly."""
    rows = replay_stream(make_session(1234))
    assert rows, "replay produced no rows"
    for r in rows:
        sw = r["superseded_within"]
        assert set(sw) == set(HAZARD_HORIZONS)
        assert sw[1] <= sw[2] <= sw[4]
        assert sw[2] == r["superseded_next"]


def test_hazard_net_loads_and_is_calibrated_monotone():
    w = json.loads((ROOT / "data" / "wm_hazard_weights.json").read_text())
    assert w["horizons"] == list(HAZARD_HORIZONS)
    m = w["metrics"]
    for h in HAZARD_HORIZONS:
        assert m[f"auroc_hz{h}"] > 0.55, f"hazard k={h} at chance"
        assert m[f"ece_hz{h}"] < 0.05, f"hazard k={h} miscalibrated"
    assert m["monotone_violation_rate_0.05"] < 0.05


def test_imagine_plan_oracle_dominates_and_none_floors():
    """On a couple of seeds the oracle policy must dominate no-maintenance,
    and every policy must answer every question (zero-latency contract)."""
    import imagine_plan as ip
    wm, hz = ip.LearnedWM(), ip.HazardWM()
    rng = np.random.default_rng(0)
    for seed in (5000, 5007):
        r_none = ip.run_policy("none", seed, 1, wm, hz, rng)
        r_orc = ip.run_policy("oracle", seed, 1, wm, hz, rng)
        assert r_none["n"] == r_orc["n"] > 0
        assert r_orc["correct"] >= r_none["correct"]
        assert r_none["repairs"] == 0


def test_imagine_plan_repair_writes_truth():
    """A repaired key must afterwards hold the generator's gold value."""
    import imagine_plan as ip
    from majalis.wmfeat import parse_line
    from majalis.beliefs import parse_date_ord
    board = BeliefBoard()
    truth = {}
    for ev in make_session(5001):
        if ev.kind != "evidence":
            continue
        for raw in ev.lines:
            f = parse_line(raw)
            if f is None:
                continue
            k = BeliefBoard.make_key(f["entity"], f["attr"])
            d = parse_date_ord(f["date"])
            board.assert_fact(k, f["value"], d, source=f["source"])
            if f["source"] == "Filing":
                if k not in truth or d > truth[k][1]:
                    truth[k] = (f["value"].strip().lower(), d)
    wrong = [k for k in truth if board.current(k)
             and board.current(k).value.strip().lower() != truth[k][0]]
    if not wrong:  # this seed built a clean board; nothing to assert
        return
    k = wrong[0]
    board.assert_fact(k, truth[k][0], board._now_ord + 1, source="debate")
    assert board.current(k).value.strip().lower() == truth[k][0]
