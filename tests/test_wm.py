import json

from agora.beliefs import BeliefBoard, parse_date_ord
from agora.wm import AcceptGate, rank_targets, risk_score


def test_risk_score_monotone():
    assert risk_score(0.0, 0.0, 1.0) < 0.1
    assert risk_score(0.9, 0.7, 0.4) > 0.8
    assert risk_score(0.5, 0.0, 0.9) < risk_score(0.9, 0.0, 0.9)
    assert risk_score(0.5, 0.2, 0.9) < risk_score(0.5, 0.6, 0.9)


def test_rank_targets_prefers_uncertain_over_settled():
    b = BeliefBoard()
    churned, stable = "e::churned", "e::stable"
    b.assert_fact(churned, "v1", parse_date_ord("Jan 2025"))
    b.assert_fact(churned, "v2", parse_date_ord("Mar 2025"))
    b.assert_fact(stable, "w", parse_date_ord("Jan 2025"))
    b.assert_fact(stable, "w", parse_date_ord("May 2025"))  # refresh, no churn
    ranked = rank_targets(b, [stable, churned], max_targets=2)
    assert ranked[0] == churned
    assert rank_targets(b, ["missing::key"]) == []


def test_accept_gate_calibration_roundtrip(tmp_path, monkeypatch):
    import agora.wm as wm
    monkeypatch.setattr(wm, "_GATE_STATE", tmp_path / "gate.json")
    # Separable data: low scores harmless, high scores harmful.
    scores = [0.1] * 20 + [0.9] * 20
    harms = [0] * 20 + [1] * 20
    AcceptGate.save_calibration(scores, harms)
    assert json.loads((tmp_path / "gate.json").read_text())["scores"]
    gate = wm.AcceptGate(alpha=0.10)
    assert gate.calibrated
    assert gate.gate.trust(0.1)  # low risk accepted
    assert not gate.gate.trust(0.9)  # high risk not trusted


def test_accept_gate_fail_safe_uncalibrated(monkeypatch, tmp_path):
    import agora.wm as wm
    monkeypatch.setattr(wm, "_GATE_STATE", tmp_path / "missing.json")
    gate = wm.AcceptGate()
    assert not gate.calibrated
