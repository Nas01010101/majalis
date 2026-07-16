---
title: "Majalis: A Learned World Model Decides When Multi-Agent Debate Is Worth the Tokens"
author: "Anas — Global AI Hackathon with Qwen Cloud, Track 3: Agent Society"
date: "July 11, 2026"
mainfont: "Times New Roman"
CJKmainfont: "Songti SC"
fontsize: 10pt
geometry: margin=2.6cm
numbersections: true
abstract: |
  Multi-agent debate (MAD) on a shared backbone largely re-purchases
  self-consistency at a premium, and a single agent at equal token budget is
  a strong baseline. We present Majalis, a small society of heterogeneous Qwen
  agents whose debate lifecycle is controlled by a world model **trained on
  the society's own logged episodes** rather than by hand-set heuristics. A
  shared belief board amortizes perception across an evidence stream; two
  trained heads over the board — `wrong_now` (is the board's current value
  incorrect?) and `superseded_next` (will an authoritative filing overturn
  it?) — feed a stacker fit on real logged episodes, and a split-conformal
  threshold on the learned score guarantees E[error | accepted without
  debate] ≤ α on exchangeable data. The debate trigger costs **zero LLM
  calls**: the stacker learned a zero weight on K-sample disagreement,
  making the sampler provably removable. On held-out streams the learned
  gate fires on 12.4% of questions and catches 86.2% of corrupted boards at
  a 0.9% false-fire rate, versus 23.8% / 78.8% / 15.1% for the hand-set
  gate it replaced; the fixed-prior survival model it displaced scores at
  chance (AUROC 0.496 vs 0.657 learned). End-to-end, the society matches a
  single agent's accuracy (single: 272/272; heuristic gate: 303/304, one
  miss; learned gate: 240/240) while cost per
  question stays flat in stream length ($0.0049–0.0054/q) against the
  single agent's linear growth ($0.0137/q at 32 steps), and vanilla 3×3
  MAD costs 12.6× more. All experiments run on Qwen Cloud backbones; the
  full benchmark, training pipeline, and a zero-API offline evaluation
  reproduce from seeds in the repository.

  **摘要** — 多智能体辩论在同构模型上大多以更高成本重复自洽采样。本文提出
  Majalis：一个由异构 Qwen 智能体组成的小型社会，其辩论全流程由一个**在
  社会自身运行日志上训练的世界模型**调控。共享信念板将感知成本在证据流上
  摊销；两个训练头（`wrong_now`：当前信念是否错误；`superseded_next`：
  是否将被权威信息推翻）与在真实日志上拟合的融合器共同给出风险评分，
  分裂共形阈值保证无辩论提交的错误率 E[error | accepted] ≤ α。辩论触发
  **零 LLM 调用**。在留出流上，学习门控以 12.4% 的触发率捕获 86.2% 的
  被污染信念（误触发 0.9%），全面优于其替代的手工门控（23.8% / 78.8% /
  15.1%）；被替换的固定参数生存先验仅达随机水平（AUROC 0.496 对 0.657）。
  端到端准确率与单智能体持平，而每问成本在流长度上保持平坦（$0.0049–
  0.0054/问），单智能体则线性增长（32 步时 $0.0137/问）；朴素 3×3 辩论
  成本高出 12.6 倍。全部实验基于 Qwen Cloud，代码与种子可完整复现。
---

# Introduction

Controlled evaluations of multi-agent debate systems repeatedly find the
same two results: homogeneous debate mostly re-buys self-consistency at
higher cost, with model heterogeneity the one robust lever
[1], and a single agent given the same token budget is a
brutal baseline [2]. The efficiency response of 2025–26 has
been *sparsity* — trigger debate per query with a trained classifier (iMAD
[3]; SELENE [4]), route by round-0 agreement (ARMOR-MAD
[5]), stop early with a sequential test [6], or sparsify the
communication topology (DySCo [7]; PEAR [8]). These controllers
are *stateless*: each query is gated in isolation, and whatever the gate
learned about the world is discarded.

Majalis moves the controller into a **persistent, learned world model**. A
society of heterogeneous Qwen agents maintains one shared belief board over
a stream of dated, contradictory evidence; the board is both the working
memory that amortizes perception and the substrate the world model reads.
Two small trained heads predict, per belief, the two quantities the debate
lifecycle actually needs — *is this value wrong now* and *will it be
overturned soon* — and a conformally calibrated threshold on the resulting
risk score decides, at zero LLM calls, whether committing without debate is
safe. Debate outcomes are written back into the board, so the next
question starts from a corrected state.

Our contributions:

1. **A learned world model for debate control.** Decision-relevant heads
   (in the sense of agent-authored world modeling [11]) trained on 115k
   offline episode rows minted from the benchmark generator's own ground
   truth at zero LLM cost, with the sim-to-real gap *measured* (0.937
   AUROC on real LLM-built boards the model never saw) rather than assumed.
2. **A zero-call calibrated gate.** A stacker fit on 96 real logged
   episodes maps (head risk, sampled disagreement, weak-source flag) to
   P(wrong); split conformal risk control on this learned score keeps
   E[error | accepted] ≤ α = 0.05 (empirically 2.1% on 1,600 held-out
   questions). The stacker assigns disagreement sampling a weight of
   exactly zero, so the gate runs without any LLM calls.
3. **An honest efficiency claim.** Accuracy parity with the single-agent
   baseline at every stream length with flat cost per question — the win is
   structural (amortization + selective spend), not a claim that debate
   improves ceiling accuracy on saturated tasks.
4. **Negative results with measurements.** The hand-set Lomax survival
   prior we shipped first scores at chance for forecasting supersession;
   exposing dynamics scores to the reasoner *reduces* accuracy from 100% to
   33% on churned facts; a judge that never re-proposes silently discards
   its own corrections.

# Related Work

**Sparse and gated debate.** iMAD [3] trains a 41-feature MLP on
self-critique text to trigger debate per query (up to 92% token reduction,
+13.5% accuracy); SELENE [4] reports ~50% token cuts at a 0.8–1.5pp
accuracy cost in an industrial setting; ARMOR-MAD [5] routes by
round-0 agreement, stops on convergence, and down-weights outliers,
training-free; a Wald-SPRT governor [6] cuts calls 3.7× on GSM8K but
inherits its judge's mis-calibration elsewhere. Majalis differs in *state*:
its gate reads a persistent belief board whose risk estimates compound
across a stream, catches corruption that per-query classifiers cannot see
(all samples read the same corrupted board), and costs zero LLM calls at
decision time.

**Learned debate-outcome prediction.** Minority Sentinel [9]
trains a LightGBM on a 22-dimensional debate fingerprint from 686 logged
episodes to decide when to overturn majority voting — and shows an
LLM-as-judge *degrades* performance. This validates small trained tabular
controllers on logged episodes; Majalis applies the recipe upstream of the
debate (commit-vs-debate) rather than after it.

**Conformal control for agent decisions.** Conformal Social Choice
[10] applies split conformal prediction over pooled agent probabilities
for act-versus-escalate decisions with a marginal coverage guarantee.
Majalis's guarantee has the same form (split CRC, exchangeable calibration
split) but calibrates a *learned* score over a persistent belief state, and
routes risk to a debate that can repair the state instead of escalating to
a human.

**Belief substrates.** Typed belief stores with supersession and epistemic
status are established 2026 art — Tenure [12], TOKI's write-time
contradiction control [13], WorldDB [14]. Majalis claims no novelty
for the substrate; the contribution is what it is wired to — the substrate
is the world model's feature source and the debate's write-back target.

**Multi-agent failure taxonomies.** MAST [15] attributes multi-agent
failures to inter-agent misalignment and weak verification/termination.
Majalis's answers are structural: star topology (no agent-to-agent chat),
typed artifact handoffs, author≠validator separation across backbones, and
termination owned by the world model.

# Method

## Belief board

The board is a keyed store (`entity::attribute` → value) with
bi-temporal-lite semantics: later-*dated* assertions supersede; older-dated
re-assertions of retired values are recorded as **stale echoes** (churn
evidence) but never win; same-date disagreements are conflicts. Sources
carry a two-tier authority (`Filing`/`debate` = authoritative; everything
else weak), and the sharpest single corruption signal is
`weak_current`: a weak source displacing an authoritative value by date —
a policy violation no sampling-based signal can see, because every sample
reads the same corrupted board. Debates read per-key **dockets** (the dated
assertion history of one key), never the raw stream, keeping debate cost
O(key history).

## Learned world model

**Features.** One feature function serves training and inference (12
per-key features): value age, key exposure, assertion count, supersession
count, stale-echo/conflict count, churn per month, distinct-value count,
current source tier, `weak_current`, weak-source share of history, and the
two displaced heuristics demoted to features — the closed-form Lomax
survival P(valid) and the hand-set doubt blend.

**Targets.** Following the decision-relevant-target principle [11] we
do not predict next observations. Head 1, `wrong_now`: is the board's
current value for this key incorrect under the generator's ground truth
(latest-dated arrived filing)? Head 2, `superseded_next`: does an
authoritative filing change this key's value within the next two evidence
batches?

**Data.** Training rows are minted **offline at zero LLM cost**: streams
from seeds 1000–2999 (train) and 3000–3199 (validation) are replayed
through a deterministic extractor into the real board code, and labels come
from the generator's own truth rule. This yields 115,483 train / 11,678
validation rows (label rates: `wrong_now` 10.8%, `superseded_next` 33.7%),
seed-disjoint from every evaluation and calibration split (§4.2).

**Architecture and training.** A shared trunk (two fully connected layers
of 64 units, ReLU) with two scalar heads, trained with BCE (positive-class
weight = neg/pos for `wrong_now`), Adam at $3\cdot 10^{-3}$, batch 8192, ≤60 epochs
with early stopping (patience 8) on validation AUROC — 18 seconds on one
RTX 3080. A HistGradientBoosting baseline on identical features ties the
MLP (0.9993 / 0.6596 AUROC), so nothing is lost by shipping the MLP, whose
weights export to JSON for dependency-free numpy inference at serve time.

**Stacker (sim-to-real).** A logistic regression fit on the **96 real
logged calibration episodes** (LLM-built boards, real disagreement samples,
harm = the committed answer was wrong) maps (head risk, K-sample
disagreement, `weak_current`) → P(wrong). Leave-one-seed-out AUROC is
0.9528 (head alone: 0.9372) — the measured transfer from offline replay to
real LLM perception. The fitted disagreement coefficient is **exactly
0.0**: on real episodes the trained head subsumes the sampler, licensing a
gate that makes zero LLM calls (§5.4).

## Conformal accept gate

Scores for the calibration episodes are recomputed offline with the learned
model; weak-flagged episodes are excluded from threshold fitting **by
flag** (they hard-fire as policy — a known violation, not a probabilistic
risk) and the remaining (score, harm) pairs calibrate a split-conformal
threshold at α = 0.05 (CalibratedGate, split conformal risk control), so
that E[error | accepted without debate] ≤ α on exchangeable data. The gate
is fail-safe: uncalibrated means a conservative floor debates everything
risky. The guarantee never depends on the learned score being a true
probability. A cost-breakeven clause additionally accepts when the
expected accuracy gain of a debate is below its token cost equivalent.

## The society

Star topology around a deterministic Python orchestrator (the only board
writer): an extractor (qwen3.6-flash) asserts dated facts once per
evidence batch; a proposer (qwen3.7-max) answers from the board summary
(never shown dynamics scores — §6); the gate decides commit vs debate; on
debate, a skeptic on a *different backbone* (qwen3.7-plus) attacks the
highest-risk supporting beliefs (expected-information-gain order: Bernoulli
entropy of learned P(wrong), ≤2 targets), and a judge (qwen3.7-max) rules
against the docket, writing corrections back as authoritative `debate`
assertions. After any debate the proposer **always re-proposes** from the
corrected board. Handoffs are typed artifacts; no agent messages another
directly.

# Experimental Setup

## Tasks

Session streams: dated evidence batches with interleaved true/false claim
questions, generated with per-seed ground truth. Streams contain three
noise processes — cross-batch supersession (arrival order ≠ date order),
stale echoes of retired values, and **rumors**: wrong values *postdating*
the latest filing, which a date-only reader absorbs but the stated policy
("filings are authoritative") excludes from gold. Questions are biased
(70%) toward churned keys. All arms see identical events and are graded
identically against generator truth, with one shared token/USD ledger
(ingest cost included and amortized).

## Seed hygiene

| Split | Seeds | Use |
|---|---|---|
| Evaluation | 0–99 | all reported end-to-end numbers |
| Gate calibration | 100–999 | 96 real logged episodes (stacker + conformal) |
| World-model training | 1000–2999 | 115,483 offline rows |
| World-model validation | 3000–3199 | 11,678 offline rows |
| Offline gate benchmark | 5000–5099 | 1,600 questions, never touched above |

## Baselines and ablations

**single** — one qwen3.7-max agent that re-reads the stream-so-far for
every question (O(stream) input tokens per question). **mad** — vanilla
3-agent × 3-round debate per question. **majalis (heuristic)** — identical
society with the original hand-set gate (logistic blend with hand-chosen
weights + fixed-prior Lomax survival), preserved as `MAJALIS_WM=heuristic`.
**majalis-nodebate** — board and gate, debates disabled (isolates what
debate adds). **majalis-wm** — the learned world model of §3.

## Metrics

Accuracy with Wilson 95% intervals; cost and tokens per question from the
shared ledger at DashScope list prices; gate decision quality on held-out
streams — fire rate, corrupted-board recall, false-fire rate on clean
boards, risk-score AUROC, and the empirical conformal coverage
E[board wrong | accepted] vs α.

# Results

## Cost stays flat; accuracy holds

Pooled session results (Wilson 95% CIs in the repository dashboard):

| Arm | Steps | Accuracy | $/question | tok/question |
|---|---|---|---|---|
| single | 8 | 80/80 | 0.00787 | 1,481 |
| mad (3×3) | 8 | 32/32 | **0.07086** | 15,614 |
| majalis (heuristic) | 8 | 111/112 | 0.00563 | 2,840 |
| majalis-nodebate | 8 | 77/80 (96.2%) | 0.00595 | 3,328 |
| **majalis-wm (learned)** | 8 | **48/48** | **0.00493** | **1,949** |
| single | 16 | 64/64 | 0.00974 | 2,141 |
| majalis (heuristic) | 16 | 64/64 | 0.00658 | 3,569 |
| **majalis-wm (learned)** | 16 | **64/64** | **0.00512** | 2,062 |
| single | 32 | 128/128 | 0.01370 | 3,525 |
| majalis (heuristic) | 32 | 128/128 | 0.00665 | 3,891 |
| **majalis-wm (learned)** | 32 | **128/128** | **0.00538** | 2,193 |

The single agent's cost grows linearly with stream length (re-reading);
Majalis's is flat, 2.1× cheaper at 32 steps and still growing apart. Vanilla
MAD pays 12.6× Majalis's cost for the same accuracy. The no-debate ablation
shows the honest value of debate here: its three errors are exactly the
rumor-poisoned beliefs the world model flags, and gated debate corrects all
three for +$0.0004 per question. The learned arm additionally cuts tokens
per question 31% versus the heuristic gate at 8 steps (1,949 vs 2,840) by
eliminating sampler calls, and is the cheapest and flattest arm at every
stream length ($0.0049–0.0054/q, 2.5× cheaper than the single agent at 32
steps). (Learned-arm cells now cover 3 seeds at 8 steps and 2 seeds at 16
and 32 steps — 240/240 correct, all on Qwen Cloud.)

## The learned heads versus the heuristics they replaced

Identical features, identical held-out data; the only change is trained
weights:

| Target | Learned (MLP) | GBDT | Hand-set baseline | ECE (learned) |
|---|---|---|---|---|
| `wrong_now` | **0.999** | 0.9993 | 0.792 (doubt blend) | 0.010 |
| `superseded_next` | **0.657** | 0.660 | 0.496 (Lomax prior) | 0.015 |

The fixed-prior survival model — a principled-looking closed form — is
**at chance** for forecasting supersession. Sim-to-real: on the 96 real
LLM-built-board episodes the `wrong_now` head alone reaches 0.937 AUROC;
the stacker 0.953 (leave-one-seed-out). The near-ceiling 0.999 is a
property of the synthetic validation domain (weak-source displacement is
highly separable there); we report the real-episode numbers as the honest
generalization figures.

## Gate decision quality, head to head

1,600 questions from 100 unseen streams (seeds 5000–5099), both modes run
through the identical `AcceptGate.decide()` code path with the sampler
stubbed to zero for both (the learned stacker's disagreement weight is 0;
the heuristic receives its skip-path value):

| Gate | Fire rate | Corrupted-board recall | False-fire | Score AUROC | E[err\|accept] (α=.05) |
|---|---|---|---|---|---|
| **Learned** | **12.4%** | **86.2%** | **0.9%** | **0.996** | **2.1%** (holds) |
| Hand-set | 23.8% | 78.8% | 15.1% | 0.921 | 3.8% (holds) |

The learned gate fires **half as often**, catches **more** corruption, and
false-fires **17× less**. Both hold the conformal coverage bound; the
learned gate holds it with a 2.3× margin. This benchmark costs zero LLM
calls and runs in under one second — it ships as a reproducible regression
test.

## The sampler is dead weight — measured, then removed

The stacker's fitted coefficient on K-sample disagreement is exactly 0.0
(coefficients: head risk 1.397, disagreement 0.000, weak flag 1.590). The
mechanism is structural: sampling-based disagreement re-reads the same
board state the head already scores, and cannot see board corruption at
all (every sample reads the same corrupted values — precisely the failure
Minority Sentinel documents for LLM-as-judge). Because p_wrong is
mathematically invariant to the sampler under a zero weight, removing it
cannot change any gate decision; it saves 2–3 flash calls per gated
question and makes the trigger decision **0-call**. If retraining ever
assigns the coefficient a non-zero value, the gate resumes sampling
automatically.

## Live deployment

The full system (FastAPI, learned gate, numpy-only inference) runs on an
Alibaba Cloud ECS instance; a recorded live session (seed 0, 8 steps,
$0.0911 total) achieves 16/16 with debates firing exactly on the two
rumor-displaced beliefs, both corrected by adjudication and written back.
The recording drives a public replay viewer showing the board's per-belief
risk, every gate decision with its reason, and the full skeptic/judge
exchanges.

# Analysis: three lessons the metrics forced

**Dynamics scores poison the reasoner.** Showing P(valid)/doubt to the
proposer dropped churn-family accuracy from 100% to 33% — the model
converts "this belief is probably stale" into "the claim is false"
regardless of the claim's content. Dynamics are consumed only by the gate;
prompts carry values, dates, and sources only.

**Debate must end in re-proposal.** A judge can uphold a belief that the
original proposal contradicted; without a mandatory re-proposal the
debate's work is silently discarded (observed live before the rule:
debates ran, answers didn't change).

**Weak-source displacement is policy, not probability.** Half of
weak-displacement episodes are harmless by luck; letting them shape the
conformal threshold drags it above the harmful half. They are excluded by
flag and hard-fire debate as a policy violation.

# Limitations

The evidence streams are synthetic with template-parsable structure; the
near-ceiling `wrong_now` AUROC reflects that separability, and the honest
transfer numbers are the real-episode ones (0.937/0.953). Real logged
episodes number only 96; the stacker has three coefficients partly for
this reason. Learned-arm live cells now cover 3 seeds at 8 steps and 2 seeds at 16/32
steps (240/240 correct, flat $0.0049–0.0054/q). The conformal guarantee is marginal over
exchangeable tasks and applies to the ACCEPT decision, not to debated
answers. Per-task families (churn/compare/multihop) saturate on every Qwen
backbone tested, so no ceiling-accuracy claim is made anywhere — the
contribution is the cost regime and the calibrated control.

# Reproducibility

All numbers reproduce from seeds with five commands against the released
repository: `make test` (35 tests), `python scripts/gen_wm_dataset.py`
(dataset, 1.4s, zero API), `python train/train_wm.py` (18s GPU or ~1min
CPU), `python scripts/offline_bench.py` (Table 3, zero API, <1s), and
`python -m majalis.bench.session --arms single,majalis,mad,majalis-wm` (paid
cells; finished cells resume free from raw logs). Calibration:
`python -m majalis.bench.calibrate --session-seeds 100,101,102` then
`python scripts/refit_gate_learned.py` (offline). These two reproduction
paths carry different determinism guarantees: `offline_bench.py` is
provably deterministic (zero LLM calls, numpy replay over fixed seeds),
while `bench.session`'s `--seeds` is seeded-but-LLM-dependent — the seed
is forwarded as DashScope's `seed` parameter, documented best-effort (as
with OpenAI's own `seed`), so a live re-run can differ from the committed
numbers by a question or two even at an identical seed. The heuristic and
learned arms are pinned by arm name (`majalis` vs `majalis-wm`) rather than
an ambient env var; `MAJALIS_WM=heuristic` remains available as an
explicit override for ad-hoc re-tuning.

# References

1. When more agents isn't better: debate as self-consistency re-purchase; model heterogeneity as the robust lever. arXiv:2502.08788.
2. Single-agent equal-budget baselines for multi-agent systems. arXiv:2604.02460.
3. iMAD: Instance-adaptive Multi-Agent Debate. arXiv:2511.11306 (AAAI-26).
4. SELENE: Selective debate initiation in production. EACL 2026 Industry Track, 2026.eacl-industry.7.
5. ARMOR-MAD: Adaptive Routing for Heterogeneous Multi-Agent Debate in LLM Reasoning. arXiv:2606.13197.
6. A sequential compute governor for multi-agent debate. arXiv:2605.19193.
7. DySCo: Dynamic sparse communication topologies for agent teams. arXiv:2606.01828.
8. PEAR: Equivariant sparse routing for multi-agent systems. arXiv:2606.20621.
9. Minority Sentinel: When to Overturn Majority Voting in Multi-Agent LLM Debates. arXiv:2606.29270.
10. From Debate to Decision: Conformal Social Choice for Safe Multi-Agent Deliberation. arXiv:2604.07667.
11. Agent-Authored World Modeling: decision-relevant training targets for agent world models. arXiv:2606.09032.
12. Tenure: Typed belief stores with epistemic status and supersession. arXiv:2605.11325.
13. TOKI: Write-time contradiction control with audit-row provenance. arXiv:2606.06240.
14. WorldDB: A database view of agent world state. arXiv:2604.18478.
15. MAST: Why do multi-agent LLM systems fail? arXiv:2503.13657.
