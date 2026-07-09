# Agora

**A world-model-modulated debate society.** Agents don't just argue in rounds — a shared, calibrated belief-state world model decides *whether a disagreement is worth the tokens*, *which belief gets challenged*, *who speaks next*, *when debate terminates*, and *what gets committed back*. Debate outcomes update the world model; the world model steers the debate.

Built for the Global AI Hackathon with Qwen Cloud — Track 3: Agent Society.

> Status: under construction. Architecture and benchmark docs land in `docs/`.

## Why

Multi-agent debate on a single backbone is mostly self-consistency at higher cost (arXiv:2502.08788, 2604.02460). The levers that actually survive controlled evals are model heterogeneity, verification-shaped tasks, and *sparsity* — only debating when it's likely to change the answer (cf. iMAD, arXiv:2511.11306). Agora makes the sparsity decision with a persistent, calibrated world model instead of a stateless per-query classifier, and closes the loop by writing resolved debates back into the belief state.

## Components

- **Belief substrate**: [tenet-memory](https://github.com/Nas01010101/tenet) — bi-temporal keyed belief store with per-fact P(still valid).
- **Conformal control**: CalibratedGate / TrajectoryCRC / AnytimeAlarm (preact-wm) + conformalized symbolic dynamics (wm-reasoner), giving distribution-free guarantees on accepted claims.
- **Society**: star-topology orchestrator, heterogeneous Qwen backbones (qwen3.7-max / plus / 3.6-flash), author≠validator separation, typed artifact handoffs.

## License

MIT
