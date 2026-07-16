import json

from majalis.beliefs import BeliefBoard, parse_date_ord
from majalis.wm import AcceptGate, rank_targets, risk_score


def test_weak_current_dominates_risk():
    from majalis.beliefs import parse_date_ord
    b = BeliefBoard()
    k = "e::ceo"
    b.assert_fact(k, "Chen", parse_date_ord("Jan 2025"), source="Filing")
    b.assert_fact(k, "Okafor", parse_date_ord("May 2025"), source="Rumor")
    assert b.weak_current(k)
    # A rumor-corrupted board must outscore an honest one decisively.
    assert risk_score(0.3, 0.0, 0.9, weak_current=True) > \
        risk_score(0.3, 0.0, 0.9, weak_current=False) + 0.3
    # A later authoritative filing clears the flag.
    b.assert_fact(k, "Larsson", parse_date_ord("Aug 2025"), source="Filing")
    assert not b.weak_current(k)


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
    import majalis.wm as wm
    monkeypatch.setattr(wm, "_GATE_STATE", tmp_path / "gate.json")
    monkeypatch.setattr(wm, "load_wm", lambda: None)  # isolate heuristic path
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
    import majalis.wm as wm
    monkeypatch.setattr(wm, "_GATE_STATE", tmp_path / "missing.json")
    monkeypatch.setattr(wm, "_GATE_STATE_LEARNED", tmp_path / "missing2.json")
    monkeypatch.setattr(wm, "load_wm", lambda: None)
    gate = wm.AcceptGate()
    assert not gate.calibrated


def test_majalis_session_wm_mode_controls_gate(monkeypatch):
    """MajalisSession(wm_mode=...) must deterministically pin heuristic vs
    learned scoring, independent of MAJALIS_WM — the arm-name-not-env-var
    fix bench/session.py relies on for reproducibility."""
    monkeypatch.delenv("MAJALIS_WM", raising=False)
    from majalis.society import MajalisSession
    heuristic = MajalisSession(wm_mode="heuristic")
    learned = MajalisSession(wm_mode="learned")
    assert heuristic.gate.wm is None
    assert learned.gate.wm is not None
    # An ambient MAJALIS_WM must still override (loudly) — the documented
    # escape hatch — even over an explicit wm_mode.
    monkeypatch.setenv("MAJALIS_WM", "heuristic")
    overridden = MajalisSession(wm_mode="learned")
    assert overridden.gate.wm is None


def test_learned_gate_loads_and_calibrates():
    """With trained weights + learned calibration present (repo state),
    the gate runs the learned path end to end."""
    import majalis.wm as wm
    gate = wm.AcceptGate()
    assert gate.wm is not None and gate.calibrated


def test_learned_wm_orders_risk_sensibly():
    from majalis.wmnet import LearnedWM
    m = LearnedWM()
    poisoned, clean = BeliefBoard(), BeliefBoard()
    poisoned.assert_fact("e::ceo", "chen", parse_date_ord("Jan 2025"), source="Filing")
    poisoned.assert_fact("e::ceo", "okafor", parse_date_ord("May 2025"), source="Rumor")
    clean.assert_fact("e::hq", "berlin", parse_date_ord("Jan 2025"), source="Filing")
    assert m.wrong_now(poisoned, "e::ceo") > m.wrong_now(clean, "e::hq") + 0.3
    assert m.wrong_now(clean, "missing::key") == 1.0
    # Stacker: risk rises with head risk and with the weak flag.
    assert m.commit_risk(0.9, 0.5, True) > m.commit_risk(0.05, 0.0, False)
    assert m.commit_risk(0.9, 0.0, False) > m.commit_risk(0.1, 0.0, False)
    assert m.commit_risk(0.5, 0.0, True) > m.commit_risk(0.5, 0.0, False)
    # Empirical finding baked into the gate: the stacker learned ZERO weight
    # on sampled disagreement (the head subsumes it on real episodes), which
    # is what licenses AcceptGate's sampler skip. If retraining ever makes
    # this coefficient meaningful, the gate resumes sampling automatically.
    if abs(m.stk_coef[1]) < 1e-9:
        assert m.commit_risk(0.2, 0.8, False) == m.commit_risk(0.2, 0.0, False)


def test_wmfeat_parse_and_replay_consistency():
    from majalis.bench.stream import make_session
    from majalis.wmfeat import FEATURES, key_features, parse_line, replay_stream
    f = parse_line("[Mar 2025] Filing: Acme Corp's ceo is Jane Doe.")
    assert f == {"entity": "Acme Corp", "attr": "ceo", "value": "Jane Doe",
                 "date": "Mar 2025", "source": "Filing"}
    assert parse_line("not an evidence line") is None
    rows = replay_stream(make_session(4242))
    assert rows and all(len(r["x"]) == len(FEATURES) for r in rows)
    # Labels are booleans over real keys; feature fn matches board state.
    b = BeliefBoard()
    b.assert_fact("x::y", "v", parse_date_ord("Jan 2025"), source="Filing")
    assert len(key_features(b, "x::y")) == len(FEATURES)
