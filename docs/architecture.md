# Agora — Architecture

## One paragraph

Agora is a multi-agent debate society in which a **shared world model** — a keyed
belief board with a closed-form fact-dynamics survival model, plus sampling-based
disagreement estimates and a conformal accept gate — controls the debate rather
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
- **`wm.py` — control layer.** Risk feature = logistic blend of (max support
  doubt, K-sample disagreement, author confidence). The *guarantee* never rests
  on the feature being a true probability: the ACCEPT threshold is conformally
  calibrated (preact-wm CalibratedGate, split conformal risk control) so that
  E[error | accepted-without-debate] ≤ α on exchangeable data. Fail-safe when
  uncalibrated. Targeting is expected-information-gain: challenge the
  highest-entropy supporting belief; a belief with doubt ≈0 or ≈1 teaches
  nothing.
- **`society.py` — the society.** extract → assert into board → propose →
  gate → (skeptic ↔ judge, ≤2 targets, always re-propose) → commit. Two rules
  learned from live failures, now encoded:
  1. **Doubt never reaches the reasoner.** Showing p_valid/doubt to the
     proposer poisoned it (measured 100%→33% on churn — the same regression
     Tenet recorded when doubt touched ranking). Dynamics are gate-only.
  2. **Debate always re-proposes.** A judge can uphold a belief that the
     original proposal contradicted; without a mandatory re-proposal the
     debate's work is silently discarded.
- **`bench/` — the harness.** Arms: `single` (CoT), `sc5` (self-consistency,
  the honest null), `mad` (vanilla 3×3 debate), `agora`. Families: `churn`
  (claim verification under evolving evidence — debate's proven habitat) and
  `multihop` (control where the gate should stay closed). Seeded offline
  generators, Wilson 95% CIs, per-call token/latency ledger shared by every
  arm, one command (`make bench`). Calibration seeds (≥100) are disjoint from
  eval seeds (0–99).

## Guarantee fine print (honest limits)

- The conformal guarantee is *marginal* over exchangeable tasks and applies to
  the ACCEPT decision, not to debated answers.
- CSD-style naive coverage claims are **not** made: sample-frequency laws
  under-cover on small models (measured in wm-reasoner); frequencies are used
  as risk *features*, with the guarantee coming from threshold calibration.
- P(still valid) is a survival heuristic; it is calibrated in form, not
  validated per-domain. It is used to *prioritize spend*, never to decide truth
  — truth is decided by the judge against dated evidence.
