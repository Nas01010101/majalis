---
title: "Majalis: A Learned World Model for Agent Societies"
author: "Anas, Global AI Hackathon with Qwen Cloud, Track 3: Agent Society"
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
  trained heads over the board, `wrong_now` (is the board's current value
  incorrect?) and `superseded_next` (will an authoritative filing overturn
  it?), feed a stacker fit on real logged episodes, and a split-conformal
  threshold on the learned score guarantees E[error | accepted without
  debate] ≤ α on exchangeable data. The debate trigger costs **zero LLM
  calls**: the stacker learned a zero weight on K-sample disagreement,
  making the sampler provably removable. On held-out streams the learned
  gate fires on 12.4% of questions and catches 86.2% of corrupted boards at
  a 0.9% false-fire rate, versus 23.8% / 78.8% / 15.1% for the hand-set
  gate it replaced; the fixed-prior survival model it displaced scores at
  chance (AUROC 0.496 vs 0.657 learned). Counterfactual mining of 592 paired
  (skip, debate) outcomes shows debate flips 4.6% of answers right and none
  wrong, and a two-branch planned gate built on that model matches, but
  does not beat, the reactive threshold (an honest null; the reactive gate
  is 2× more debate-frugal at equal accuracy). A calibrated multi-horizon
  hazard curve gives the model a rollout, and maintenance policies audited
  **entirely in imagination** (zero API) close 95% of the
  no-maintenance→oracle gap under zero-latency serving. End-to-end, the society matches a
  single agent's accuracy (single: 480/480; heuristic gate: 303/304, one
  miss; learned gate: 448/448; planned gate: 320/320; zero-latency
  maintenance mode: 112/112 live with no ask-time debate) while cost per
  question stays flat in stream length ($0.0049–0.0054/q) against the
  single agent's linear growth ($0.0137/q at 32 steps), and vanilla 3×3
  MAD costs 12.6× more. All experiments run on Qwen Cloud backbones; the
  full benchmark, training pipeline, and a zero-API offline evaluation
  reproduce from seeds in the repository.

  **摘要**, 多智能体辩论在同构模型上大多以更高成本重复自洽采样。本文提出
  Majalis：一个由异构 Qwen 智能体组成的小型社会，其辩论全流程由一个**在
  社会自身运行日志上训练的世界模型**调控。共享信念板将感知成本在证据流上
  摊销；两个训练头（`wrong_now`：当前信念是否错误；`superseded_next`：
  是否将被权威信息推翻）与在真实日志上拟合的融合器共同给出风险评分，
  分裂共形阈值保证无辩论提交的错误率 E[error | accepted] ≤ α。辩论触发
  **零 LLM 调用**。在留出流上，学习门控以 12.4% 的触发率捕获 86.2% 的
  被污染信念（误触发 0.9%），全面优于其替代的手工门控（23.8% / 78.8% /
  15.1%）；被替换的固定参数生存先验仅达随机水平（AUROC 0.496 对 0.657）。
  对 592 组（跳过, 辩论）配对反事实的挖掘表明：辩论纠正 4.6% 的答案且从不帮倒忙；基于该模型的双分支规划门控与反应式阈值准确率持平（诚实的零结果，反应式门控辩论开销低 2 倍）。多时域风险曲线赋予模型可展开的前向动态；维护策略完全在想象中评估（零 API 成本），在零延迟服务约束下弥合 95% 的无维护→神谕差距；获胜策略已真实落地（majalis-maintain 模式）：7 个种子 112/112 全对，问答路径零辩论延迟。端到端准确率与单智能体持平，而每问成本在流长度上保持平坦（$0.0049–
  0.0054/问），单智能体则线性增长（32 步时 $0.0137/问）；朴素 3×3 辩论
  成本高出 12.6 倍。全部实验基于 Qwen Cloud，代码与种子可完整复现。
---

# Introduction

Controlled evaluations of multi-agent debate systems repeatedly find the
same two results: homogeneous debate mostly re-buys self-consistency at
higher cost, with model heterogeneity the one robust lever
[1], and a single agent given the same token budget is a
brutal baseline [2]. The efficiency response of 2025–26 has
been *sparsity*, trigger debate per query with a trained classifier (iMAD
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
lifecycle actually needs, *is this value wrong now* and *will it be
overturned soon*, and a conformally calibrated threshold on the resulting
risk score decides, at zero LLM calls, whether committing without debate is
safe. Debate outcomes are written back into the board, so the next
question starts from a corrected state.

Our contributions:

1. **A learned world model for debate control.** Decision-relevant heads
   (in the sense of agent-authored world modeling [11]) trained on 115k
   offline episode rows minted from the benchmark generator's own ground
   truth at zero LLM cost, with the sim-to-real gap *measured* (0.937
   AUROC on real LLM-built boards the model never saw) rather than assumed,
   extended to a calibrated multi-horizon hazard curve (rollout) and an
   action-conditioned outcome head trained on 592 mined counterfactual
   debate pairs, and exercised for **planning in imagination**: maintenance
   policies auditioned entirely inside the model at zero API cost (§5.6).
2. **A zero-call calibrated gate.** A stacker fit on 96 real logged
   episodes maps (head risk, sampled disagreement, weak-source flag) to
   P(wrong); split conformal risk control on this learned score keeps
   E[error | accepted] ≤ α = 0.05 (empirically 2.1% on 1,600 held-out
   questions). The stacker assigns disagreement sampling a weight of
   exactly zero, so the gate runs without any LLM calls.
3. **An honest efficiency claim.** Accuracy parity with the single-agent
   baseline at every stream length with flat cost per question, the win is
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
episodes to decide when to overturn majority voting, and shows an
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
status are established 2026 art, Tenure [12], TOKI's write-time
contradiction control [13], WorldDB [14]. Majalis claims no novelty
for the substrate; the contribution is what it is wired to, the substrate
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
`weak_current`: a weak source displacing an authoritative value by date,
a policy violation no sampling-based signal can see, because every sample
reads the same corrupted board. Debates read per-key **dockets** (the dated
assertion history of one key), never the raw stream, keeping debate cost
O(key history).

## Learned world model

**Features.** One feature function serves training and inference (12
per-key features): value age, key exposure, assertion count, supersession
count, stale-echo/conflict count, churn per month, distinct-value count,
current source tier, `weak_current`, weak-source share of history, and the
two displaced heuristics demoted to features, the closed-form Lomax
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
with early stopping (patience 8) on validation AUROC, 18 seconds on one
RTX 3080. A HistGradientBoosting baseline on identical features ties the
MLP (0.9993 / 0.6596 AUROC), so nothing is lost by shipping the MLP, whose
weights export to JSON for dependency-free numpy inference at serve time.

**Stacker (sim-to-real).** A logistic regression fit on the **96 real
logged calibration episodes** (LLM-built boards, real disagreement samples,
harm = the committed answer was wrong) maps (head risk, K-sample
disagreement, `weak_current`) → P(wrong). Leave-one-seed-out AUROC is
0.9528 (head alone: 0.9372), the measured transfer from offline replay to
real LLM perception. The fitted disagreement coefficient is **exactly
0.0**: on real episodes the trained head subsumes the sampler, licensing a
gate that makes zero LLM calls (§5.5).

## Conformal accept gate

Scores for the calibration episodes are recomputed offline with the learned
model; weak-flagged episodes are excluded from threshold fitting **by
flag** (they hard-fire as policy, a known violation, not a probabilistic
risk) and the remaining (score, harm) pairs calibrate a split-conformal
threshold at α = 0.05 (CalibratedGate, split conformal risk control), so
that E[error | accepted without debate] ≤ α on exchangeable data. The gate
is fail-safe: uncalibrated means a conservative floor debates everything
risky. The guarantee never depends on the learned score being a true
probability. A cost-breakeven clause additionally accepts when the
expected accuracy gain of a debate is below its token cost equivalent.

## Action-conditioned head and a planned gate

The gates above are *reactive*: they threshold a state-conditioned risk.
A world model in the operational sense [11] should instead predict the
**consequence of each action** and select by comparing them. We test that
distinction directly.

**Counterfactual mining.** For every question in seeds 100–124 (train) and
130–141 (held out) we run perception and proposal **once** on a shared
board, then grade both branches of the decision: `skip_correct` (the
pre-debate proposal, free) and `debate_correct` (a real
skeptic–adjudicate–repropose run on the same board, targeting the same
most-doubted key PlannedGate would score at serve time, no train/serve
skew). 592 paired rows, $10.55 of DashScope spend. The counterfactual
result is stark: **debate flips a wrong answer to right on 27/592
questions (4.6%) and never does the reverse (0/592)**;
P(correct | debate) ≈ 0.995–1.0 regardless of state.

**Head and calibration.** A second trunk+head (same 12 features) fits
`debate_correct`. The held-out band is degenerate, every debate
succeeded, so its AUROC is reported as **null**, not as a good number.
The class-rebalanced head is also *level*-biased (mean prediction 0.78 on
an all-correct band, ECE 0.219); since the planned gate compares
P(correct | debate) against P(correct | skip) in a utility, that bias
alone would silently turn it into never-fire. A Platt map fit on the train
band ships inside the weights artifact (ECE 0.219 → 0.003, ordering
preserved), a reminder that **decision-grade probabilities need
calibrated levels, not just good rankings**.

**PlannedGate.** U(skip) = p_skip; U(debate) = p_skip + (p_debate −
p_skip)·(1 + touch_rate·γ) − λ·cost, fire iff U(debate) > U(skip), a
genuine two-branch argmax with an explicit, non-oracle estimate of how
often the key will resurface (its own running touch rate).

## Multi-horizon hazards: giving the model a rollout

A one-step head can be queried; a world model should be *rolled out*. We
extend the offline labels to a hazard curve, `superseded_within(k)` for
k in {1, 2, 4} evidence batches (k = 2 reproduces the legacy
`superseded_next` exactly, asserted in tests), and train a three-head
HazardNet on the same 115k zero-cost replay rows (2.5 s on CPU). Validation:
AUROC 0.630 / 0.659 / 0.698 for k = 1/2/4, ECE < 0.01 at every horizon,
and 0% monotonicity violations (predicted h(1) ≤ h(2) ≤ h(4)), a calibrated,
internally consistent forward model of how the board's facts will churn,
exported to JSON for numpy-only inference like every other head.

## The society

Star topology around a deterministic Python orchestrator (the only board
writer): an extractor (qwen3.6-flash) asserts dated facts once per
evidence batch; a proposer (qwen3.7-max) answers from the board summary
(never shown dynamics scores, §6); the gate decides commit vs debate; on
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
noise processes, cross-batch supersession (arrival order ≠ date order),
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

**single**, one qwen3.7-max agent that re-reads the stream-so-far for
every question (O(stream) input tokens per question). **mad**, vanilla
3-agent × 3-round debate per question. **majalis (heuristic)**, identical
society with the original hand-set gate (logistic blend with hand-chosen
weights + fixed-prior Lomax survival), preserved as `MAJALIS_WM=heuristic`.
**majalis-nodebate**, board and gate, debates disabled (isolates what
debate adds). **majalis-wm**, the learned world model of §3.
**majalis-wm-plan**, same board, same debate mechanics, but the
two-branch argmax PlannedGate replaces the reactive threshold (isolates
the decision *rule* as the only variable).

## Metrics

Accuracy with Wilson 95% intervals; cost and tokens per question from the
shared ledger at DashScope list prices; gate decision quality on held-out
streams, fire rate, corrupted-board recall, false-fire rate on clean
boards, risk-score AUROC, and the empirical conformal coverage
E[board wrong | accepted] vs α.

# Results

## Cost stays flat; accuracy holds

Pooled session results (Wilson 95% CIs in the repository dashboard):

| Arm | Steps | Accuracy | $/question | tok/question |
|---|---|---|---|---|
| single | 8 | 288/288 | 0.00787 | 1,481 |
| mad (3×3) | 8 | 32/32 | **0.07086** | 15,614 |
| majalis (heuristic) | 8 | 111/112 | 0.00563 | 2,840 |
| majalis-nodebate | 8 | 107/112 (95.5%) | 0.00595 | 3,328 |
| **majalis-wm (learned)** | 8 | **256/256** | **0.00493** | **1,949** |
| single | 16 | 64/64 | 0.00974 | 2,141 |
| majalis (heuristic) | 16 | 64/64 | 0.00658 | 3,569 |
| **majalis-wm (learned)** | 16 | **64/64** | **0.00512** | 2,062 |
| single | 32 | 128/128 | 0.01370 | 3,525 |
| majalis (heuristic) | 32 | 128/128 | 0.00665 | 3,891 |
| **majalis-wm (learned)** | 32 | **128/128** | **0.00538** | 2,193 |

The single agent's cost grows linearly with stream length (re-reading);
Majalis's is flat, 2.1× cheaper at 32 steps and still growing apart. Vanilla
MAD pays 12.6× Majalis's cost for the same accuracy. The no-debate ablation
shows the honest value of debate here: its five errors (107/112 over 7
seeds) are exactly the rumor-poisoned beliefs the world model flags, and
every gated or maintained arm answers all five correctly (§5.4). The learned arm additionally cuts tokens
per question 31% versus the heuristic gate at 8 steps (1,949 vs 2,840) by
eliminating sampler calls, and is the cheapest and flattest arm at every
stream length ($0.0049–0.0054/q, 2.5× cheaper than the single agent at 32
steps). (Learned-arm cells now cover 16 seeds at 8 steps and 2 seeds at 16
and 32 steps, 448/448 correct, all on Qwen Cloud.)

## External benchmarks: does debate help off the belief board?

The synthetic streams above saturate near 100% for every arm, including the
single-agent baseline, so accuracy there cannot separate the society from a
single agent (the contribution is cost and calibration, not raw accuracy). To
test the gate on tasks that are *not* saturated, we run two recognized external
benchmarks with an honest single-turn adaptation of the gate. There is no belief
board for a bare question, so the learned world model is not used; the gate rides
on the proposer's verbalized confidence plus, only when that is ambiguous, a
cheap k=2 self-consistency sample. This is an uncalibrated gate, and no conformal
guarantee is claimed for these two benchmarks.

On **GSM8K** (full 1,319-question test set, qwen3.7-max) the gated arm scores
**98.0%** (1,292/1,319, Wilson 95% CI [97.0, 98.6]) at 1.05 calls per question
and $0.0047/q, matching a strong single agent while firing on under 1% of
questions.

On **MMLU** (150 questions across six reasoning-heavy subjects: high-school and
college mathematics, high-school physics, formal logic, logical fallacies, and
professional law) the single agent scores **96.0%** [91.5, 98.2], the gated arm
**96.7%** [92.4, 98.6] while firing on only **1.3%** of questions (1.05 calls/q,
$0.0094/q), and a vanilla homogeneous 3×2 multi-agent debate scores **96.7%**
[92.4, 98.6] at 6.0 calls per question ($0.0561/q). Vanilla debate buys **no
accuracy over the cheap gate**, at **6× the cost**; the gate reaches the same
accuracy while paying for debate on barely one question in eighty. Both external
benchmarks tell the same story as the synthetic streams: on a strong backbone
debate is rarely worth it, and a state-aware gate captures what little it buys
while almost never spending on it. (Reproduce: `python scripts/gsm8k_bench.py`
and `python scripts/mmlu_bench.py`.)

## The learned heads versus the heuristics they replaced

Identical features, identical held-out data; the only change is trained
weights:

| Target | Learned (MLP) | GBDT | Hand-set baseline | ECE (learned) |
|---|---|---|---|---|
| `wrong_now` | **0.999** | 0.9993 | 0.792 (doubt blend) | 0.010 |
| `superseded_next` | **0.657** | 0.660 | 0.496 (Lomax prior) | 0.015 |

The fixed-prior survival model, a principled-looking closed form, is
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
calls and runs in under one second, it ships as a reproducible regression
test.

## Does planning beat reacting? A measured null, and where the value really is

We ran the planned gate end-to-end against the reactive one on live
sessions (steps = 8): 20 seeds for majalis-wm-plan, 16 for majalis-wm,
plus two stress regimes (rumor rate 0.35 → 0.6; per-question debate budget
∞ → 1) and longer streams (16, 32 steps). **Accuracy never separates**:
both gates score 100% in every regime (majalis-wm 256/256 pooled baseline;
majalis-wm-plan 320/320; all stress cells 48/48). The regimes themselves
are not vacuous, the ungated board (majalis-nodebate) errs on 4.5% of
questions (107/112 over 7 seeds), matching the 4–6% skip-failure rate the
counterfactual mining measured on its disjoint seed band, and **every one
of those 5 board-error questions is answered correctly by every gated or
maintained arm covering the seed**: the reactive gate fired on all 3 it
saw, the planned gate on 4 of 5 (on the fifth its run's independently
built board happened to be correct, perception is LLM-nondeterministic
across runs), and the maintenance arm repairs in-window by construction.
Perfect recall of real board errors; the arms differ only in false-fire
economy.

That economy is where they part: the reactive gate fires on 7.8% of
questions (731 tokens/question); the planned gate fires on 14.1% (1,069
tokens/question) for identical accuracy. The mechanism is visible in the
mined counterfactuals: with P(correct | debate) ≈ 1 everywhere, predicted
uplift is positive almost everywhere, so an argmax with a small λ·cost
term overfires relative to a threshold tuned on conformal coverage. At
these task difficulties, **planning over predicted consequences adds cost,
not accuracy**, an honest null for the LeCun-style claim in this domain,
reported with the same prominence as our positive results.

The world model's measured value sits elsewhere: against always-debate
(mad, 100% accuracy at $7.09 per 100 questions), the learned reactive gate
delivers the **same accuracy at ~1/14 the cost** ($0.50 per 100
questions), because debate is a free win *only* on the ~5% of questions
where the board is actually wrong, and the model finds exactly those.
Full evidence artifact: `results/wm_action_eval.json`.

## The sampler is dead weight, measured, then removed

The stacker's fitted coefficient on K-sample disagreement is exactly 0.0
(coefficients: head risk 1.397, disagreement 0.000, weak flag 1.590). The
mechanism is structural: sampling-based disagreement re-reads the same
board state the head already scores, and cannot see board corruption at
all (every sample reads the same corrupted values, precisely the failure
Minority Sentinel documents for LLM-as-judge). Because p_wrong is
mathematically invariant to the sampler under a zero weight, removing it
cannot change any gate decision; it saves 2–3 flash calls per gated
question and makes the trigger decision **0-call**. If retraining ever
assigns the coefficient a non-zero value, the gate resumes sampling
automatically.

## Planning in imagination: the frontier the world model buys for $0

The strongest test of a world model is whether you can *use it instead of
the world*. We build a deployment regime where that is forced: **zero-latency
serving**, questions must be answered instantly from the board (no
question-time debate; interactive deployments cannot pay debate latency),
while repairs run only in maintenance windows between evidence batches, B
per step. The policy must predict which keys will be both wrong *and asked*
before the questions arrive. Simulating a debate as "repair the key" is
licensed by measurement, not assumption: the mined counterfactuals grade
real debates at P(correct | debate) ≈ 0.995–1.0 with zero harmful flips.
Every policy below is evaluated entirely inside the model's replay world,
zero LLM calls, on 100 held-out streams (seeds 5000–5099, n = 1,600):

| Policy (B = 1 repair/step) | Accuracy | Wilson 95% |
|---|---:|---|
| no maintenance | 92.2% | [90.8, 93.5] |
| random repair | 95.1% | [93.9, 96.0] |
| **learned-risk repair** (`wrong_now`) | **99.5%** | [99.0, 99.7] |
| hazard-discounted planned repair | 99.5% | [99.0, 99.7] |
| oracle (labels) | 99.9% | [99.5, 100] |

Learned-risk maintenance closes **95% of the no-maintenance→oracle gap**
using 2.8× more repairs than the oracle needs, and the entire policy
comparison cost nothing, which is precisely the world-model dividend:
candidate policies are auditioned in imagination, and only the winner needs
live verification (§5.4's live sessions confirm the transfer: the same
`wrong_now` head achieves perfect board-error recall in real LLM streams).

**Live transfer.** The winning policy is implemented for real
(`MajalisSession.maintain()`, arm `majalis-maintain`): between evidence
batches it runs one genuine skeptic→judge debate on the learned-risk top
key; at ask time the gate never fires, questions serve straight from the
board. Across 7 live seeds: **112/112 correct with zero ask-time debates**
at $0.0092/question. The trade is explicit: maintenance debates every batch
cost more than reactive gating (~$0.005/q) but remove debate latency from
the serving path entirely, the regime interactive deployments live in.

**Honest null #2.** The hazard-discounted ranking, deprioritize keys the
world is about to overwrite, p_wrong · (1 − h(1)) · (0.1 + touch_rate),
never beats plain myopic risk repair in any regime we tested (B in {1, 2};
rumor rate 0.35/0.6; 8/16 steps). Together with the planned-gate null
(§5.4) this is a consistent, twice-replicated finding: **in this
environment the world model's decision value concentrates in calibrated
state estimation; its forward dynamics are learnable and calibrated but
not yet decision-relevant at these horizons.** We report both nulls with
the same prominence as the wins because they carve the claim to exactly
what the evidence supports.

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
proposer dropped churn-family accuracy from 100% to 33%, the model
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
this reason. Learned-arm live cells now cover 16 seeds at 8 steps and 2 seeds at 16/32
steps (448/448 correct, flat $0.0049–0.0054/q), plus 20 planned-gate seeds
(320/320) and 7 zero-latency maintenance seeds (112/112). The conformal guarantee is marginal over
exchangeable tasks and applies to the ACCEPT decision, not to debated
answers. A finer point: the threshold scan selects the largest empirically
passing τ without the conformal-risk-control finite-sample (+1) correction
or learn-then-test multiplicity control, so at n = 96 calibration episodes
the α level is approximate (up to ~1pp anti-conservative); the load-bearing
evidence for the coverage claim is therefore the *empirical* held-out check
(2.1% ≤ α on 1,600 questions), and the vendored gate ships a Hoeffding-UCB
mode for deployments that need a high-probability rather than
in-expectation floor. Per-task families (churn/compare/multihop) saturate on every Qwen
backbone tested, so no ceiling-accuracy claim is made anywhere, the
contribution is the cost regime and the calibrated control.

**Is this a world model?** Under the operational, value-equivalent
definition we adopt (predict decision-relevant consequences of candidate
actions; select by comparing them, the MuZero/JEPA stance rather than a
generative simulator): yes, and the components are individually measured,
state estimation (`wrong_now`), calibrated multi-horizon dynamics (the
hazard curve, rollable with 0% monotonicity violations), an
action-conditioned outcome head trained on real counterfactual pairs, and
policy evaluation in imagination that transfers live. Under the strict
sense, a transition model over full board states, composable over long
action sequences, not yet: our rollouts are per-key risk curves, not
board-state simulation. Our own results keep the claim honest twice over:
both places where the *forward* components could have beaten purely
reactive state estimation (the planned gate, §5.4; hazard-discounted
maintenance, §5.6), they matched it instead. The world-model machinery
earns its keep today through zero-cost policy audition and measured
action outcomes, not yet through multi-step foresight.

# Reproducibility

All numbers reproduce from seeds with five commands against the released
repository: `make test` (118 tests), `python scripts/gen_wm_dataset.py`
(dataset, 1.4s, zero API), `python train/train_wm.py` (18s GPU or ~1min
CPU), `python scripts/offline_bench.py` (Table 3, zero API, <1s), and
`python -m majalis.bench.session --arms single,majalis,mad,majalis-wm` (paid
cells; finished cells resume free from raw logs). Calibration:
`python -m majalis.bench.calibrate --session-seeds 100,101,102` then
`python scripts/refit_gate_learned.py` (offline). These two reproduction
paths carry different determinism guarantees: `offline_bench.py` is
provably deterministic (zero LLM calls, numpy replay over fixed seeds),
while `bench.session`'s `--seeds` is seeded-but-LLM-dependent, the seed
is forwarded as DashScope's `seed` parameter, documented best-effort (as
with OpenAI's own `seed`), so a live re-run can differ from the committed
numbers by a question or two even at an identical seed. The heuristic and
learned arms are pinned by arm name (`majalis` vs `majalis-wm`) rather than
an ambient env var; `MAJALIS_WM=heuristic` remains available as an
explicit override for ad-hoc re-tuning.

The action-conditioned extension (§3.4, §5.4) adds three: `python
scripts/gen_action_wm_dataset.py --seeds 100:125 --out
data/action_wm_train.jsonl` (paid counterfactual mining, per-seed atomic
resume, budget-capped), the same for the 130:142 held-out band, and
`python train/train_action_wm.py` (offline, seconds; exports the
Platt-calibrated head to `data/wm_action_weights.json`). The planned-arm
cells are `python -m majalis.bench.session --arms majalis-wm-plan --seeds
0..19` plus the two stress regimes (`--rumor-rate 0.6`, `--max-debates 1`);
mined rows, weights, per-question raw logs, and the pooled evidence
artifact (`results/wm_action_eval.json`) all ship in the repository.
The rollout and imagination results (§3.5, §5.6) are fully zero-API:
`python scripts/gen_wm_dataset.py` (hazard labels), `python
train/train_wm_hazard.py` (2.5 s), and `python scripts/imagine_plan.py
--seeds 5000:5100 --budgets 1,2` regenerate `results/imagination_frontier.json`
deterministically.

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
