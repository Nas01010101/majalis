# Devpost submission draft — Track 3: Agent Society

**Title**: Agora — a world-model-modulated debate society

**Tagline**: Agents that know when a disagreement is worth the tokens.

## Description (paste into Devpost)

Multi-agent debate has a dirty secret: on a single backbone it mostly re-buys
self-consistency at 10× the cost (arXiv:2502.08788, 2604.02460). Agora is a
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
with interleaved questions, Agora matches single-agent and vanilla-MAD
accuracy exactly (272/272) while its cost per question stays **flat**
($0.0056) as the stream grows — the single agent re-reads the stream every
question and grows linearly ($0.0137 at 32 steps, 2.4×); vanilla 3×3 MAD
costs 12.6×. Gate parameters were tuned by a journaled, fail-closed
keep/revert optimization loop on held-out seeds (−12%, transfers across
seeds).

**Qwen Cloud usage**: three Qwen backbones via the OpenAI-compatible API
(model heterogeneity is the one debate lever with robust evidence —
2502.08788's "universal antidote"); backend deployed on Alibaba Cloud ECS
(`src/agora/api.py`); real Qwen Cloud pricing in the cost ledger.

**Honest limits**: guarantees are marginal over exchangeable tasks and apply
to the ACCEPT decision; benchmark families are synthetic-but-property-tested
(gold provably consistent with the latest authoritative evidence); one
residual non-weak error class (~2.5%) sits inside the calibrated band.

## Form checklist
- Track: 3 (Agent Society)
- Repo: github.com/Nas01010101/agora (public by submit; MIT LICENSE in About)
- Architecture diagram: docs/architecture.md (+ dashboard screenshot)
- Demo video: <3min YouTube (scripts/demo.sh flow + dashboard + bench table)
- Proof of AliCloud deploy: separate recording — ECS console + curl /healthz
  + /ingest + /ask from public IP; code file: src/agora/api.py
- Testing access: instance stays up through judging (Jul 31)
