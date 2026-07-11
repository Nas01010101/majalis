# Agora

**A debate society steered by a learned world model.** Agents don't just argue in rounds — a shared world model, **trained on the society's own logged episodes**, decides *whether a disagreement is worth the tokens*, *which belief gets challenged*, *who speaks next*, *when debate terminates*, and *what gets committed back*. Debate outcomes update the belief state; the belief state is what the world model reads; every session's episodes become its next training data.

Two trained heads over the shared belief board (torch on GPU to train, numpy-only to serve; conformal threshold on top for a distribution-free accept guarantee):

| head | predicts | learned vs the hand-set baseline it replaced |
|---|---|---|
| `wrong_now` | P(the board's current value for a key is incorrect) | AUROC **0.999 vs 0.79** (synthetic val); **0.937 on real LLM-built boards** it never saw |
| `superseded_next` | P(an authoritative filing overturns it within lookahead) | AUROC **0.657 vs 0.496 (= chance)** for the fixed Lomax prior |

The stacker (fit on real logged episodes, LOSO AUROC 0.95) learned a **zero weight** on K-sample disagreement — the trained head subsumes the sampler — so the debate-trigger decision costs **zero LLM calls**. On 60 unseen streams (960 questions): fires on 9.8%, catches 82.2% of corrupted boards, 0.7% false-fire. Live: 16/16 accuracy at **25% fewer tokens/question** than the hand-set gate. Retrain from scratch: `python scripts/gen_wm_dataset.py && python train/train_wm.py` (~2 min end to end; the old heuristic survives as ablation via `AGORA_WM=heuristic`).

Built for the Global AI Hackathon with Qwen Cloud — Track 3: Agent Society.

## Measured results (session eval, 5 seeds, Wilson 95% CIs)

Evidence streams with interleaved questions and unreliable sources; every arm
sees identical events, graded identically, one shared token+USD ledger.

| arm | accuracy | cost/question | note |
|---|---|---|---|
| **Agora** | **272/272 across all stream lengths** | **$0.0056, flat in stream length** | gate debates 6–16% of questions; gate params tuned by a metric-driven keep/revert loop on held-out seeds (experiments/gate-cost) |
| single agent | 272/272 | $0.0079 → $0.0137, linear in stream length | re-reads the stream per question |
| vanilla MAD (3×3) | 32/32 | $0.0709 | 10.6× Agora's cost |
| Agora, debate ablated | 77/80 (96.2%) | $0.0060 | the 3 errors are exactly the rumor-poisoned beliefs the world model flagged; gated debate corrects all 3 for +$0.0004/q |

Reproduce: `make bench` (per-task families) and
`python -m agora.bench.session --arms single,agora,mad --seeds 0,1,2,3,4`.
Calibration: `python -m agora.bench.calibrate --session-seeds 100,101,102`
(seeds ≥100 are never evaluated).

## Why

Multi-agent debate on a single backbone is mostly self-consistency at higher cost (arXiv:2502.08788, 2604.02460). The levers that actually survive controlled evals are model heterogeneity, verification-shaped tasks, and *sparsity* — only debating when it's likely to change the answer (cf. iMAD, arXiv:2511.11306). Agora makes the sparsity decision with a persistent, calibrated world model instead of a stateless per-query classifier, and closes the loop by writing resolved debates back into the belief state.

## Components

- **Belief substrate**: [tenet-memory](https://github.com/Nas01010101/tenet) — bi-temporal keyed belief store with per-fact P(still valid).
- **Conformal control**: CalibratedGate / TrajectoryCRC / AnytimeAlarm (preact-wm) + conformalized symbolic dynamics (wm-reasoner), giving distribution-free guarantees on accepted claims.
- **Society**: star-topology orchestrator, heterogeneous Qwen backbones (qwen3.7-max / plus / 3.6-flash), author≠validator separation, typed artifact handoffs.

## License

MIT
