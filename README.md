<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
    <img alt="Majalis" src="assets/logo-light.svg" width="300">
  </picture>

  <h3>Your agents debate too much. Majalis's learned world model decides when it's worth it.</h3>

  <p>
    <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
  </p>

  <p>
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT">
    <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/tests-35%20passing-brightgreen" alt="Tests">
    <img src="https://img.shields.io/badge/gate%20decision-0%20LLM%20calls-blue" alt="Zero-call gate">
    <img src="https://img.shields.io/badge/Qwen%20Cloud-Track%203%3A%20Agent%20Society-8A2BE2" alt="Qwen Cloud Hackathon">
  </p>

  <p>
    <a href="http://47.237.187.157:8080/">Live dashboard</a> ·
    <a href="http://47.237.187.157:8080/live">Society view — replay or live</a> ·
    <a href="http://47.237.187.157:8080/docs">API playground</a> ·
    <a href="docs/paper/majalis.pdf">Paper (PDF)</a> ·
    <a href="docs/architecture.md">Architecture</a>
  </p>
</div>

Multi-agent debate on one backbone mostly re-buys self-consistency at higher cost (arXiv:2502.08788, 2604.02460). Majalis gives a Qwen agent team a **shared belief board** and a **world model trained on the society's own logged episodes** that routes debate only where the belief state is likely corrupted — with a conformal guarantee on everything committed without one (E[error | accepted] ≤ α), at **zero LLM calls per gate decision**.

```bash
git clone https://github.com/Nas01010101/majalis && cd majalis
pip install -e . && python examples/quickstart.py   # no API key needed
```

```python
from majalis.beliefs import BeliefBoard, parse_date_ord
from majalis.wmnet import load_wm

board = BeliefBoard()
board.assert_fact("acme::ceo", "Jane Doe", parse_date_ord("Jan 2026"), source="Filing")
board.assert_fact("acme::ceo", "John Roe", parse_date_ord("Mar 2026"), source="Rumor")

wm = load_wm()
wm.wrong_now(board, "acme::ceo")   # ~1.0 — a rumor displaced a filing: debate-worthy
```

## Why Majalis

- **Debate is a spend decision.** Two trained heads — `wrong_now`, P(belief is incorrect) (AUROC **0.999 vs 0.79** for the hand-set gate it replaced; **0.937 on real LLM-built boards** it never saw) and `superseded_next`, P(overturned soon) (**0.657 vs 0.496 = chance** for the fixed prior) — decide commit-vs-debate with **0 LLM calls**.
- **Calibrated, not vibes.** The ACCEPT threshold is split-conformal on the learned score, and the coverage claim is checked empirically: accepted-error 2.1% ≤ α=0.05 on 1,600 held-out questions.
- **Retrain it yourself.** `python scripts/gen_wm_dataset.py && python train/train_wm.py` — ~2 min end to end (torch to train, numpy-only to serve). The replaced heuristic survives as an ablation (`MAJALIS_WM=heuristic`).
- **Feed it your own evidence, live.** In the [society view](http://47.237.187.157:8080/live), switch to *live — try it*: paste dated evidence lines and watch the extractor build beliefs, the world model re-score them, and the gate spend debate only where P(wrong) spikes — through the real deployed society. (Anonymous callers share a small daily budget.)

## Results

Session eval: evidence streams with interleaved questions and unreliable sources; every arm sees identical events, one shared token+USD ledger (5 seeds, Wilson 95% CIs).

| arm | accuracy | cost/question | note |
|---|---|---|---|
| **Majalis** (learned gate, default) | **240/240, all stream lengths** | **$0.0049–0.0054/q, flat** | 0 LLM calls to decide the gate; debates ~12% of questions |
| Majalis (heuristic gate, opt-in) | 303/304, all stream lengths | $0.0056, flat | `MAJALIS_WM=heuristic`; debates 6–16% of questions |
| single agent | 272/272 | $0.0079 → $0.0137, linear (2.5× at 32 steps) | re-reads the stream per question |
| vanilla MAD (3×3) | 32/32 | $0.0709 | 12.6× Majalis's cost |
| Majalis, debate ablated | 77/80 (96.2%) | $0.0060 | its 3 errors are exactly the rumor-poisoned beliefs the WM flagged; gated debate fixes all 3 for +$0.0004/q |

Gate quality, no API key (100 unseen streams, <1s): learned fires **12.4%** / catches **86.2%** of corrupted boards / **0.9%** false-fire, vs **23.8% / 78.8% / 15.1%** hand-set. Both hold the coverage bound.

```bash
python scripts/offline_bench.py                                    # gate quality + coverage, $0
python -m majalis.bench.session --arms single,majalis,mad --seeds 0,1,2,3,4
python scripts/e2e_live.py                                         # 14 invariants vs the deployed box (~$0.05)
```

## How it works

A star-topology society on heterogeneous Qwen backbones (qwen3.7-max proposer/judge · qwen3.7-plus skeptic · qwen3.6-flash extractor, author ≠ validator) writes to a bi-temporal belief board ([tenet-memory](https://github.com/Nas01010101/tenet)); the learned world model re-scores every belief as evidence lands, a conformal gate routes commit-vs-debate, and resolved debates are written back to the board. Full details: [paper](docs/paper/majalis.pdf) · [architecture](docs/architecture.md).

## License

MIT
