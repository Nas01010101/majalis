<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
    <img alt="Agora" src="assets/logo-light.svg" width="300">
  </picture>

  <h3>Your agents debate too much. Agora's learned world model decides when it's worth it.</h3>

  <p>
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT">
    <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/tests-27%20passing-brightgreen" alt="Tests">
    <img src="https://img.shields.io/badge/gate%20decision-0%20LLM%20calls-blue" alt="Zero-call gate">
    <img src="https://img.shields.io/badge/Qwen%20Cloud-Track%203%3A%20Agent%20Society-8A2BE2" alt="Qwen Cloud Hackathon">
  </p>

  <p>
    <a href="http://47.237.187.157:8080/">Live dashboard</a> ·
    <a href="http://47.237.187.157:8080/docs">API playground</a> ·
    <a href="docs/architecture.md">Architecture</a> ·
    <a href="docs/submission.md">Hackathon submission</a>
  </p>
</div>

Multi-agent debate mostly re-buys self-consistency at higher cost. Agora gives an agent team a **shared belief board** and a **world model trained on the society's own logged episodes** that routes debate only where the state is likely corrupted — with a conformal guarantee on what gets committed without one (E[error | accepted] ≤ α), and **zero LLM calls per gate decision**.

```bash
git clone https://github.com/Nas01010101/agora && cd agora
pip install -e . && python examples/quickstart.py   # no API key needed
```

```python
from agora.beliefs import BeliefBoard, parse_date_ord
from agora.wmnet import load_wm

board = BeliefBoard()
board.assert_fact("acme::ceo", "Jane Doe", parse_date_ord("Jan 2026"), source="Filing")
board.assert_fact("acme::ceo", "John Roe", parse_date_ord("Mar 2026"), source="Rumor")

wm = load_wm()
wm.wrong_now(board, "acme::ceo")   # ~1.0 — a rumor displaced a filing: debate-worthy
```

## Why Agora?

- **Debate as a spend decision.** Two trained heads — `wrong_now` (is this belief incorrect?) and `superseded_next` (will it be overturned?) — plus a stacker fit on real logged episodes decide commit-vs-debate. No judge model, no extra samples: the trigger costs **0 LLM calls**.
- **A world model you retrain yourself.** `python scripts/gen_wm_dataset.py && python train/train_wm.py` — ~2 min end to end (torch to train, numpy-only to serve). The hand-set heuristic it replaced survives as an ablation (`AGORA_WM=heuristic`).
- **Calibrated, not vibes.** The ACCEPT threshold is split-conformal on the learned score; the coverage claim is checked empirically in the offline benchmark (accepted-error 2.1% ≤ α=0.05 on 1,600 held-out questions).
- **Provable without an API key.** `python scripts/offline_bench.py` replays 100 unseen evidence streams through the real gate in under a second: learned fires 12.4% / catches 86.2% of corrupted boards / 0.9% false-fire, vs 23.8% / 78.8% / 15.1% for the hand-set gate.
- **Flat cost in stream length.** Perception amortizes into the board (O(board) per question, not O(stream)); the single-agent baseline's cost grows linearly, Agora's doesn't.

| learned head | predicts | vs the hand-set baseline it replaced |
|---|---|---|
| `wrong_now` | P(board value is incorrect) | AUROC **0.999 vs 0.79** (synthetic val); **0.937 on real LLM-built boards** it never saw |
| `superseded_next` | P(fact overturned within lookahead) | AUROC **0.657 vs 0.496 (= chance)** for the fixed Lomax prior |

Built for the Global AI Hackathon with Qwen Cloud — Track 3: Agent Society (heterogeneous qwen3.7-max / plus / qwen3.6-flash backbones).

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
Zero-API offline benchmark (gate quality, learned vs hand-set, coverage,
reliability — 1,600 held-out questions in <1s):
`python scripts/offline_bench.py` → `results/offline_gate_eval.json`.

## Why

Multi-agent debate on a single backbone is mostly self-consistency at higher cost (arXiv:2502.08788, 2604.02460). The levers that actually survive controlled evals are model heterogeneity, verification-shaped tasks, and *sparsity* — only debating when it's likely to change the answer (cf. iMAD, arXiv:2511.11306). Agora makes the sparsity decision with a persistent, calibrated world model instead of a stateless per-query classifier, and closes the loop by writing resolved debates back into the belief state.

## Components

- **Belief substrate**: [tenet-memory](https://github.com/Nas01010101/tenet) — bi-temporal keyed belief store with per-fact P(still valid).
- **Conformal control**: CalibratedGate / TrajectoryCRC / AnytimeAlarm (preact-wm) + conformalized symbolic dynamics (wm-reasoner), giving distribution-free guarantees on accepted claims.
- **Society**: star-topology orchestrator, heterogeneous Qwen backbones (qwen3.7-max / plus / 3.6-flash), author≠validator separation, typed artifact handoffs.

## License

MIT
