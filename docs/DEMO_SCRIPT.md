# Majalis — 3-minute demo video script

Built from what actually wins Devpost AI hackathons (judges watch the video first, back-to-back,
and score requirements + storytelling): **lead with a problem the judge feels, show it working
within ~90s, make the invisible mechanism visible, address every rubric axis in one line, and never
let the demo stall.** Screen + voice, uploaded **public** to YouTube, marked **"Not for Kids"**,
under 3:00.

Track 3 brief the demo must satisfy: task decomposition & roles · disagreement/conflict resolution ·
**measurable** efficiency gain vs a single agent. Rubric: Innovation 30 / Tech Depth 30 / Impact 25 /
Presentation 15 (ties → Innovation).

---

## The script (shot by shot)

**0:00–0:18 — Hook: a problem the judge feels**
> "Multi-agent debate has a dirty secret: on one model it mostly re-buys self-consistency at 10× the
> cost. And when agents share a memory, one bad source can quietly poison it. Majalis is a society
> of Qwen agents that debates *only when it's worth it* — and knows when a belief has been poisoned."

**0:18–1:20 — The wow, visible: a rumor poisons the board, the gate catches it**
Screen: `majalis replay examples/sample_trace.jsonl` (zero API key, deterministic) — or the live
society view on the deployed ECS box.
> "Watch the shared belief board. A filing says Meridian's ARR is $47M. Then a *rumor*, dated later,
> says $88M. A naive store takes the newest value. Majalis flags it — **superseded-by-weaker-source**
> — and the world model's risk on that belief spikes."
The gate line prints **FIRED → debate (reason=policy:weak-source, P(wrong)=0.92)**.
> "The gate fires — not on a vibe, on a policy violation. The skeptic — running on a *different* Qwen
> backbone than the proposer, so the author never validates its own claim — decomposes the challenge
> into binary sub-questions. The judge rules from the provenance docket: **overturned**, corrected
> back to $47M, written back to the board."
Then the repeat question commits instantly.
> "Ask again — it now commits for free. The poison is gone."

**1:20–1:55 — The measurable efficiency gain (the brief's hard requirement)**
Screen: README results table.
> "Every arm replays identical event streams under one shared token+USD ledger — the gain is
> measured, not claimed. Majalis stays **flat at ~$0.005 a question** as the stream grows, at
> 448/448 correct. The single agent re-reads the stream every question and grows linearly — 2.5×
> more at 32 steps. Vanilla 3×3 debate costs 12.6× the gated arm. And the gate decision itself costs
> **zero LLM calls** — it's world-model inference."

**1:55–2:30 — How it works: a real world model over shared memory (the technical edge)**
Screen: `docs/architecture.md` diagram.
> "This isn't per-query confidence gating like the recent literature. Majalis gates on the *state of
> a persistent shared memory*: a trained head estimates which beliefs are wrong now (0.94 AUROC on
> real boards), a hazard curve forecasts what gets overturned next, and maintenance policies are
> auditioned entirely *in imagination* at $0 before any live spend. It's the classic blackboard
> control problem — which knowledge source fires next — answered with a learned model."

**2:30–2:55 — Impact + honesty (Impact + confidence)**
> "Everything it commits without a debate carries a conformal guarantee — expected error under 5%,
> checked empirically. It runs three domains unchanged: due-diligence, GSM8K, and knowledge-base
> maintenance that repairs boards with **zero** ask-time debates. And we report two honest nulls at
> full prominence — planned gating never beats reactive here. That's the society."

**2:55–3:00 — Close**
> "Majalis: agents that know when a disagreement — or a source — is worth the tokens. Thanks."

---

## Exact commands per beat

| beat | time | on screen | how |
|---|---|---|---|
| Hook | 0:00–0:18 | title card / talking head | voiceover only |
| Poison → gate → overturn | 0:18–1:20 | `majalis replay examples/sample_trace.jsonl --speed 0` (zero key) **or** live society view on `http://47.237.187.157:8080/live` | replay is deterministic & safe; the live view is more impressive but pre-load it |
| Efficiency table | 1:20–1:55 | README results table | static scroll |
| Architecture | 1:55–2:30 | `docs/architecture.md` diagram | static |
| Impact + nulls | 2:30–2:55 | README "Why Majalis" / results | static |
| Close | 2:55–3:00 | logo / repo URL card | voiceover |

Full live alternative for the wow (needs `DASHSCOPE_API_KEY`, ~3 min, real Qwen calls ≈$0.05):
`python scripts/demo_company.py` — the investment-committee run (verified 10/10, NO-GO, ~3 min).
**Start recording at ACT 1** and keep the intro off-camera, or it runs over 3:00.

## Never-stall checklist (do before the real take)
- [ ] Default to `majalis replay examples/sample_trace.jsonl` for the wow — it's zero-key and can't
      stall on a live call. Use the live box/`demo_company.py` only if you've dry-run it that session.
- [ ] If using the live ECS view, hit `/healthz` and `/board` first (both 200) and pre-load the page.
- [ ] If running `demo_company.py`, `set -a; source .env; set +a` first; **never show the token or
      key on camera**; time it — it's right at 3:00.
- [ ] Have any typed evidence lines copied to clipboard.
- [ ] Dry-run once end-to-end and time it; trim the efficiency beat first if long.
- [ ] YouTube: **public** + **"Not for Kids"**; grab the link as upload starts.

## One-line-per-rubric-axis / brief-clause
- **Decomposition & roles (brief a / Innovation):** four typed roles across three Qwen backbones,
  author ≠ validator; skeptic decomposes challenges into binary sub-questions; planner decomposes the
  composite GO/NO-GO — all visible in the demo.
- **Conflict resolution (brief b):** date-supersession flags the rumor, the gate mandates
  adjudication, the judge resolves from the docket and writes back — the 5 debate-ablated errors are
  exactly the poisoned beliefs; gated debate corrects all 5.
- **Efficiency gain (brief c / Impact):** flat $0.005/q vs single-agent linear, 448/448, one shared
  ledger; gate = 0 LLM calls.
- **Technical Depth (30):** learned world model (state + hazards + imagination + conformal accept)
  over a persistent board; two honest nulls reported.
- **Presentation (15):** the poison → catch → overturn happens *on screen*.

## AliCloud proof (for the separate/optional deploy shot + the form field)
Live backend on Alibaba Cloud ECS: `http://47.237.187.157:8080` (`/healthz`, `/board`, `/ask`).
Code-file proof: `src/majalis/llm.py` (`dashscope-intl.aliyuncs.com` via DASHSCOPE_API_KEY) +
`src/majalis/api.py` (the ECS service).
