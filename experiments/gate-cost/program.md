# Autoresearch program — gate cost

**Objective**: minimize cost_usd per question of the `agora` session arm.

**Metric** (runner-owned, fail-closed): `target.py` runs the agora arm on
HELD-OUT stream seeds 5,6 (steps 8; never used in eval seeds 0-4 or
calibration seeds 100-105), deletes any cached raw files first, and writes
cost/question to `$AUTORESEARCH_RESULTS`. HARD CONSTRAINT: if accuracy < 31/32
the metric is written as 999 (constraint violation = automatic revert).
(Amended after iteration 0: the baseline config itself scores 31/32 on the
held-out seeds — one non-weak residual error inside the calibrated 2.5%
band — so the constraint is "no worse than baseline", not perfection.)

**Editable surface**: `target.py` ONLY (it sets the `AGORA_GATE_K` /
`AGORA_GATE_SKIP_DOUBT` env knobs). The bench, grader, stream generator, and
calibration files are the judge and are out of bounds.

**Time-box**: 1800s per experiment (two live sessions ≈ 10-15 min).

**Baseline to beat**: current config (K=3, no skip): measured by iteration 0.

**Stop**: plateau over 2 iterations, 5-iteration cap, STOP file, or budget.
