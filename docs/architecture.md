# Agora — Architecture

## One paragraph

Agora is a multi-agent debate society in which a **shared, learned world model** —
a keyed belief board plus two trained predictive heads (`wrong_now`: is the
board's current value incorrect; `superseded_next`: will an authoritative
filing overturn it soon), a stacker fit on real logged episodes, and a
conformal accept gate on the learned score — controls the debate rather
than merely feeding it. The world model decides **whether a disagreement is worth
the tokens** (trigger), **which belief gets challenged** (targeting), **who argues**
(speaker/backbone selection), **when debate stops** (termination), **what gets
committed** (calibrated acceptance with a distribution-free error guarantee), and
**what is written back** (keyed supersession). Debate outcomes update the belief
state; the belief state steers the next debate.

## Why this design (evidence-driven)

Controlled 2025-26 evals show homogeneous multi-agent debate mostly re-buys
self-consistency at higher cost (arXiv:2502.08788), and a single agent at equal
token budget is a brutal baseline (arXiv:2604.02460). The levers that survive:

1. **Model heterogeneity** — the "universal antidote" (2502.08788). Agora's
   debaters run on different Qwen backbones (qwen3.7-max / qwen3.7-plus /
   qwen3.6-flash).
2. **Verification-shaped tasks** — where cross-checking demonstrably helps.
3. **Sparsity** — iMAD (2511.11306) showed gating debate per-query cuts up to 92%
   of tokens while *raising* accuracy. Agora generalizes the gate from a
   stateless per-query classifier to a persistent, calibrated world model that
   modulates the entire debate lifecycle and compounds across tasks.

### Positioning vs the mid-2026 frontier (checked 2026-07-10)

The sparse/gated-debate space moved fast in May–June 2026; per control point,
the closest neighbors are: **trigger** — iMAD (2511.11306) and SELENE (EACL
2026 Industry, 2026.eacl-industry.7: selective debate initiation bypasses
30–60% of cases, ~50% token cut *at a 0.8–1.5pp accuracy cost*); **topology**
— DySCo (2606.01828, −70% tokens via sparse edges) and PEAR (2606.20621,
equivariant sparse routing); **termination** — a Wald-SPRT compute governor
for debates (2605.19193: 3.7× call cut on GSM8K but 2.1× cost blowup on MMLU
where judge scores don't discriminate — stopping rules inherit their judge's
calibration, which is why Agora's stop signal reads the belief state rather
than a consensus judge); **acceptance** — Budgeted
Act-or-Defer deliberation (2606.29654, calibrated act/defer with a conditional
reliability bound, deferring to humans); **substrate** — Mesh Memory Protocol
(2604.19540) and ECON's belief-encoder coordination (ICML 2025); the
**belief-store-with-supersession itself is established art** — Tenure (2605.11325, typed
belief store with epistemic status + supersession), TOKI (2606.06240,
write-time contradiction control with audit-row provenance), and WorldDB
(2604.18478; 96.4% LongMemEval-s, author-reported) all ship one in 2026, so Agora claims no novelty for the
substrate — only for what it is wired to.

What none of them do, and Agora does: close the loop through ONE persistent,
inspectable belief state — the same board that amortizes perception also
carries doubt, provenance, and supersession outcomes; the gate reads it, the
debate writes back into it, and the next question starts smarter. Two
measured consequences the neighbors don't show: gating with **zero accuracy
cost** (the gate routes to correction instead of skipping — SELENE pays
0.8–1.5pp for its bypass), and a **flat cost-per-question curve in stream
length** (their evals are per-query, so amortization doesn't exist as an
axis). Each mechanism alone: incremental. The closed loop and the cost
regime: the contribution.

MAST (2503.13657) attributes multi-agent failures to structural causes —
inter-agent misalignment and weak verification/termination. Agora's answers are
structural too: star topology (no agent-to-agent chat), typed artifact handoffs,
author≠validator separation, and termination owned by the world model.

## Components

```
                        ┌────────────────────────────┐
                        │   ORCHESTRATOR (Python,     │
                        │   deterministic, sole       │
                        │   belief-board writer)      │
                        └──┬──────┬──────┬──────┬────┘
              typed        │      │      │      │      typed
              artifacts ┌──▼──┐ ┌─▼───┐ ┌▼────┐ ┌▼────┐ artifacts
                        │Extr.│ │Prop.│ │Skep.│ │Judge│
                        │fast │ │max  │ │plus │ │max  │   ← heterogeneous
                        └─────┘ └─────┘ └─────┘ └─────┘     Qwen backbones
                            │
                 ┌──────────▼───────────────┐
                 │  WORLD MODEL              │
                 │  BeliefBoard: keyed       │
                 │   supersession, stale-echo│
                 │   conflicts, Lomax        │
                 │   P(still valid)          │
                 │  AcceptGate: conformal    │
                 │   E[error|accepted] ≤ α   │
                 │  Disagreement sampler     │
                 │   (CSD frequency law)     │
                 └──────────────────────────┘
```

- **`beliefs.py` — BeliefBoard.** Keys are `entity::attribute`. Later-dated
  values supersede; older-dated re-assertions of retired values are recorded as
  *stale echoes* (churn evidence, doubt↑) but never win. P(still valid) is a
  Gamma-Lomax posterior predictive over each key's observed churn — the same
  form as tenet-memory's fact-dynamics model, which backs the board via MCP once
  embeddings are available (in-process board mirrors its semantics exactly).
- **`wm.py` + `wmnet.py` + `wmfeat.py` — the LEARNED world model.** Two
  decision-relevant heads (AAWM-style targets, arXiv:2606.09032) on a shared
  MLP trunk, trained on 115k logged episode rows replayed offline from the
  stream generator (labels from its own ground truth — zero LLM cost;
  train/train_wm.py, 18s on an RTX 3080; numpy-only inference at serve time):
  - `wrong_now(key)` — P(the board's current value is incorrect). Val AUROC
    **0.999 vs 0.79** for the old hand-set doubt blend; **0.937 on real
    LLM-built-board episodes** it never trained on (sim-to-real measured,
    not assumed). This is the same recipe Minority Sentinel
    (arXiv:2606.29270) validates for debate-outcome prediction: a small
    trained classifier on a fingerprint, which beats LLM-as-judge.
  - `superseded_next(key)` — P(an authoritative filing overturns the value
    within lookahead): learned fact dynamics, AUROC **0.657 vs 0.496 —
    chance — for the fixed-prior Lomax survival it replaced** (the learned
    per-key decay DAGE lists as open work).
  A logistic stacker fit on the 96 real calibration episodes (leave-one-
  seed-out AUROC 0.95) maps (head risk, sampled disagreement, weak flag) →
  P(committed answer wrong). It learned a ZERO weight on sampled
  disagreement — the head subsumes the K-sample sampler — so the gate skips
  those LLM calls entirely: the trigger decision costs **zero LLM calls**.
  The *guarantee* still never rests on the score being a true probability:
  the ACCEPT threshold is conformally calibrated on the learned score
  (preact-wm CalibratedGate, split CRC) so E[error | accepted-without-
  debate] ≤ α on exchangeable data; fail-safe when uncalibrated; the
  hand-set blend survives as fallback + ablation (`AGORA_WM=heuristic`).
  Targeting is expected-information-gain over the learned P(wrong):
  challenge the highest-entropy supporting belief.
- **`society.py` — the society.** extract → assert into board → propose →
  gate → (skeptic ↔ judge, ≤2 targets, always re-propose) → commit. Two rules
  learned from live failures, now encoded:
  1. **Doubt never reaches the reasoner.** Showing p_valid/doubt to the
     proposer poisoned it (measured 100%→33% on churn — the same regression
     Tenet recorded when doubt touched ranking). Dynamics are gate-only.
  2. **Debate always re-proposes.** A judge can uphold a belief that the
     original proposal contradicted; without a mandatory re-proposal the
     debate's work is silently discarded.
- **`bench/` — the harness.** Two evaluations, both seeded/offline-generated,
  Wilson 95% CIs, one shared token+USD ledger, calibration seeds (≥100)
  disjoint from eval seeds (0–99):
  1. **Session eval (headline)** — `bench/session.py`: dated evidence streams
     with questions interleaved, plus unreliable sources (rumors postdating
     filings, wrong by construction). Baselines re-read the stream-so-far per
     question — O(stream) input tokens each time; Agora ingests each batch
     once and answers from the board — O(board) per question, debating only
     doubted keys via per-key dockets. Measured (seed 0): accuracy parity at
     every length with **cost/question flat for Agora ($0.0053) and linear
     for single-agent ($0.0135 at 32 steps, 2.5× and growing)**. Every Qwen
     backbone ceilings on per-task synthetic families, so the honest gain is
     this structural one — the regime where multi-agent genuinely wins
     (context/amortization), not re-bought self-consistency.
  2. **Per-task eval (secondary + ablations)** — `bench/run.py`: `single`,
     `sc5` (the honest null), `mad` (vanilla 3×3), `agora`, and ablations
     (`agora-nogate` / `agora-nodebate` / `agora-noeig`) over `churn`,
     `compare`, `multihop` families.

## Guarantee fine print (honest limits)

- The conformal guarantee is *marginal* over exchangeable tasks and applies to
  the ACCEPT decision, not to debated answers.
- CSD-style naive coverage claims are **not** made: sample-frequency laws
  under-cover on small models (measured in wm-reasoner); frequencies are used
  as risk *features*, with the guarantee coming from threshold calibration.
- The closed-form Lomax P(still valid) is now a *feature* of the learned
  heads, not the model: measured at chance (AUROC 0.496) for forecasting
  supersession, which is precisely why it was replaced.
- The learned heads' near-ceiling wrong_now AUROC (0.999) is on synthetic
  validation streams where weak-source displacement is highly separable; the
  honest generalization numbers are the real-episode ones (0.937 head alone,
  0.95 LOSO stacker, 82.2% poisoned-board recall at 0.7% false-fire on 60
  unseen streams). World-model outputs *prioritize spend*, never decide truth
  — truth is decided by the judge against dated evidence.
