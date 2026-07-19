"""Pin every headline number to its committed evidence artifact.

The docs (README, paper, site, Devpost copy) cite specific figures; this suite
makes the artifacts the single source of truth and fails loudly if a re-run
ever changes one — the stale-citation failure mode becomes a red test instead
of a doc bug found by hand.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(rel):
    return json.loads((ROOT / rel).read_text())


def test_gate_quality_learned_dominates_handset():
    d = _load("results/offline_gate_eval.json")["learned_vs_heuristic"]
    learned, hand = d["learned"], d["heuristic"]
    # cited: fires 12.4% / catches 86.2% / 0.9% false-fire vs 23.8 / 78.8 / 15.1
    assert round(learned["fire_rate"] * 100, 1) == 12.4
    assert round(learned["recall_poisoned"] * 100, 1) == 86.2
    assert round(learned["false_fire_rate"] * 100, 1) == 0.9
    assert round(hand["fire_rate"] * 100, 1) == 23.8
    assert round(hand["recall_poisoned"] * 100, 1) == 78.8
    assert round(hand["false_fire_rate"] * 100, 1) == 15.1
    # dominance on every axis, not just the cited snapshot
    assert learned["fire_rate"] < hand["fire_rate"]
    assert learned["recall_poisoned"] > hand["recall_poisoned"]
    assert learned["false_fire_rate"] < hand["false_fire_rate"]
    assert learned["score_auroc"] > hand["score_auroc"]


def test_conformal_coverage_holds_at_alpha():
    g = _load("results/offline_gate_eval.json")["gate_quality"]
    assert g["alpha"] == 0.05
    assert g["coverage_holds"] is True
    # cited: accepted-error 2.1% <= alpha on 1,600 held-out questions
    assert g["n_questions"] == 1600
    assert round(g["accepted_error_rate"] * 100, 1) == 2.1
    assert g["accepted_error_rate"] <= g["alpha"]


def test_imagination_frontier_cited_points():
    d = _load("results/imagination_frontier.json")["policies"]
    # cited: none 92.2% -> learned-risk 99.5% vs oracle 99.9% at B=1, n=1,600
    assert d["none@B1"]["n"] == 1600
    assert round(d["none@B1"]["acc"] * 100, 2) == 92.25
    assert round(d["myopic@B1"]["acc"] * 100, 2) == 99.50
    assert round(d["oracle@B1"]["acc"] * 100, 2) == 99.88
    # honest null #2: hazard-discounted "planned" never beats myopic at either budget
    for b in ("B1", "B2"):
        assert d[f"planned@{b}"]["correct"] <= d[f"oracle@{b}"]["correct"]
        assert d[f"planned@{b}"]["correct"] - d[f"myopic@{b}"]["correct"] <= 2
    # gap closure ~96% at B=1
    closure = (d["myopic@B1"]["acc"] - d["none@B1"]["acc"]) / (
        d["oracle@B1"]["acc"] - d["none@B1"]["acc"]
    )
    assert closure >= 0.94


def test_counterfactual_mining_debate_never_hurts():
    m = _load("results/wm_action_eval.json")["counterfactual_mining"]
    pairs = m["train_band"]["rows"] + m["heldout_band"]["rows"]
    helps = m["train_band"]["debate_helps"] + m["heldout_band"]["debate_helps"]
    hurts = m["train_band"]["debate_hurts"] + m["heldout_band"]["debate_hurts"]
    # cited: helps 27/592 (4.6%), hurts 0/592
    assert pairs == 592 and helps == 27 and hurts == 0


def test_platt_calibration_shipped_and_effective():
    d = _load("results/wm_action_eval.json")["action_wm_training"]
    assert d["ece_heldout_platt"] < 0.01 < d["ece_heldout_raw_head"]
    w = _load("data/wm_action_weights.json")
    assert "platt" in w and len(w["platt"]) == 2


def test_live_maintain_cited_numbers():
    d = _load("results/wm_action_eval.json")["live_maintain"]
    # cited: 112/112 over 7 seeds, zero ask-time debates, $0.0092/q
    assert (d["correct"], d["n"], d["seeds"]) == (112, 112, 7)
    assert d["ask_time_debates"] == 0
    assert round(d["cost_usd_per_q"], 4) == 0.0092


def test_hazard_weights_metrics_match_citations():
    d = _load("data/wm_hazard_weights.json")["metrics"]
    aurocs = [round(d[f"auroc_hz{k}"], 2) for k in (1, 2, 4)]
    # cited: AUROC 0.63/0.66/0.70 at k=1/2/4, ECE < 0.01, 0% monotone violations
    assert aurocs == [0.63, 0.66, 0.70]
    assert all(d[f"ece_hz{k}"] < 0.01 for k in (1, 2, 4))
    assert d["monotone_violation_rate_0.05"] == 0.0
