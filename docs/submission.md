# Devpost submission draft — Track 3: Agent Society

**Title**: Majalis — a world-model-modulated debate society

**Tagline**: Agents that know when a disagreement is worth the tokens.

## Description (paste into Devpost)

Multi-agent debate has a dirty secret: on a single backbone it mostly re-buys
self-consistency at 10× the cost (arXiv:2502.08788, 2604.02460). Majalis is a
society of heterogeneous Qwen agents (qwen3.7-max / plus / 3.6-flash) whose
debates are governed by a shared **world model**: a persistent belief board
with keyed supersession, per-belief survival estimates, provenance dockets,
and a conformally calibrated accept gate. The world model decides **whether**
to debate (calibrated risk + a deterministic rule for known policy
violations), **what** to challenge (expected information gain), **who**
argues (a skeptic on a different backbone than the proposer/judge —
author≠validator by construction), **when** to stop, **what** to commit
(E[error | committed] ≤ α, distribution-free), and **writes the resolution
back** so the next question starts smarter.

**Task division & roles** (track requirement a): a flash extractor structures
evidence into the board; a max proposer answers from beliefs; a plus skeptic
attacks doubted beliefs via binary sub-questions; a max judge adjudicates
from the key's docket; a deterministic Python orchestrator is the only
writer — star topology, typed artifact handoffs, no free agent-to-agent chat
(dodges MAST's inter-agent-misalignment failure cluster, 2503.13657).

**Disagreement resolution** (track requirement b): unreliable sources
(rumors postdating filings) poison the board by date-supersession; the board
flags the displacement, the gate mandates adjudication, and the judge
resolves the conflict from the docket under an explicit source-authority
policy — then supersedes the belief. Measured: the 3 residual errors in the
debate-ablated system are exactly the rumor-poisoned beliefs; gated debate
corrects all 3.

**Measurable efficiency gain** (track requirement c, one command, Wilson 95%
CIs, identical events + shared token/USD ledger per arm): on evidence streams
with interleaved questions, Majalis (heuristic gate) is within one question
of single-agent and vanilla-MAD accuracy (303/304, vs 272/272 and 32/32)
while its cost per question stays **flat** ($0.0056) as the stream grows —
the single agent re-reads the stream every question and grows linearly
($0.0137 at 32 steps, 2.4×); vanilla 3×3 MAD costs 12.6×. The shipped
default — a learned world-model gate — is both more accurate and cheaper
still: 240/240 correct at a flat $0.0049–0.0054/q across 8/16/32-step
streams (2.5× under the single agent at 32 steps), with zero LLM calls to
decide the gate. Gate parameters were tuned by a journaled, fail-closed
keep/revert optimization loop on held-out seeds (−12%, transfers across
seeds).

**A real world model, measured organ by organ** — not a branding exercise:
calibrated **state estimation** (`wrong_now`, 0.937 AUROC on real LLM-built
boards; live, it fires on and corrects every question the ungated board
misses), a **multi-horizon forward-dynamics hazard curve** you can roll out
(AUROC 0.63/0.66/0.70 at k=1/2/4, ECE < 0.01, 0% monotonicity violations),
an **action-conditioned outcome head** trained on 592 real mined
counterfactual (skip, debate) pairs — debate helps 4.6%, hurts 0/592 — and
**planning in imagination**: under a zero-latency serving constraint,
maintenance policies are auditioned entirely inside the model at $0
(no-maintenance 92.2% → learned-risk 99.5% vs oracle 99.9%, n=1,600),
so only the winning policy needs live spend. Two pre-registered-style
nulls (planned gate ⊁ reactive; hazard-planning ⊁ myopic) are reported at
the same prominence as the wins.

**Qwen Cloud usage**: three Qwen backbones via the OpenAI-compatible API
(model heterogeneity is the one debate lever with robust evidence —
2502.08788's "universal antidote"); backend deployed on Alibaba Cloud ECS
(`src/majalis/api.py`); real Qwen Cloud pricing in the cost ledger.

**Honest limits**: guarantees are marginal over exchangeable tasks and apply
to the ACCEPT decision; benchmark families are synthetic-but-property-tested
(gold provably consistent with the latest authoritative evidence); one
residual non-weak error class (~2.5%) sits inside the calibrated band.

## Form checklist
- Track: 3 (Agent Society)
- Repo: github.com/Nas01010101/majalis (public by submit; MIT LICENSE in About)
- Architecture diagram: docs/architecture.md (+ dashboard screenshot)
- Demo video: <3min YouTube (scripts/demo.sh flow + dashboard + bench table)
- Proof of AliCloud deploy: separate recording — ECS console + curl /healthz
  + /ingest + /ask from public IP; code file: src/majalis/api.py
- Testing access: instance stays up through judging — submission deadline
  Jul 20, 2026 2:00pm PT; judging period ends Aug 11, 2026 2:00pm PT
