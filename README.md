# Agora

**A world-model-modulated debate society.** Agents don't just argue in rounds — a shared, calibrated belief-state world model decides *whether a disagreement is worth the tokens*, *which belief gets challenged*, *who speaks next*, *when debate terminates*, and *what gets committed back*. Debate outcomes update the world model; the world model steers the debate.

Built for the Global AI Hackathon with Qwen Cloud — Track 3: Agent Society.

## Measured results (session eval, 5 seeds, Wilson 95% CIs)

Evidence streams with interleaved questions and unreliable sources; every arm
sees identical events, graded identically, one shared token+USD ledger.

| arm | accuracy | cost/question | note |
|---|---|---|---|
| **Agora** | **272/272 across all stream lengths** | **$0.0063, flat in stream length** | gate debates 6–16% of questions |
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
