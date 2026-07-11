"""Build the society live viewer (dashboard/live.html) from a recorded run.

    python scripts/build_live.py [--replay results/replay_s0.json]

A MiroFish-style watch-the-society view at Agora's scale: belief board with
live learned-world-model meters (left), the agent feed — extract, propose,
gate, skeptic, judge — (right), terminal log (bottom), with play / pause /
scrub / speed. Self-contained single file (replay inlined); the FastAPI
server serves it at /live. Every event carries a full board snapshot, so
scrubbing is O(1) — no diff replay.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agora.config import MODEL_FAST, MODEL_MID, MODEL_STRONG  # noqa: E402

OUT = ROOT / "dashboard" / "live.html"

CSS = """
:root { --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --border:rgba(11,11,11,.10);
  --wm:#2a78d6; --extractor:#1baf7a; --skeptic:#e34948; --judge:#4a3aa7;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b; --term:#101010; --termink:#d6d5cd; }
@media (prefers-color-scheme: dark) {
  :root { --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7;
    --grid:#2c2c2a; --border:rgba(255,255,255,.10);
    --wm:#3987e5; --extractor:#199e70; --skeptic:#e66767; --judge:#9085e9;
    --term:#0a0a0a; --termink:#c3c2b7; } }
* { box-sizing:border-box; }
body { background:var(--plane); color:var(--ink); margin:0 auto; max-width:1240px;
  padding:20px 20px 32px; font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
a { color:var(--wm); text-decoration:none; font-weight:550; } a:hover { text-decoration:underline; }
.skip { position:absolute; left:-9999px; } .skip:focus { left:12px; top:8px;
  background:var(--surface); padding:8px 12px; border-radius:8px; z-index:9; }
:focus-visible { outline:2px solid var(--wm); outline-offset:2px; }
header { display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin-bottom:10px; }
.word { font-size:21px; font-weight:650; } .view { color:var(--muted); font-size:15px; }
.status { font-family:ui-monospace,Menlo,monospace; font-size:12px; color:var(--ink2);
  border:1px solid var(--border); border-radius:9px; padding:2px 9px; }
nav { margin-left:auto; display:flex; gap:14px; font-size:13px; }
.controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
  background:var(--surface); border:1px solid var(--border); border-radius:12px;
  padding:10px 14px; margin:10px 0; }
button, select { font:inherit; color:var(--ink); background:var(--surface);
  border:1px solid var(--border); border-radius:8px; padding:5px 12px; cursor:pointer; }
button:hover { border-color:var(--wm); }
input[type=range] { flex:1; min-width:160px; accent-color:var(--wm); }
.readout { font-family:ui-monospace,Menlo,monospace; font-size:12.5px; color:var(--ink2);
  min-width:130px; }
.stats { display:flex; gap:16px; font-family:ui-monospace,Menlo,monospace;
  font-size:12.5px; color:var(--ink2); flex-wrap:wrap; }
.stats b { color:var(--ink); font-weight:650; }
main { display:grid; grid-template-columns:minmax(340px,5fr) minmax(360px,7fr); gap:12px; }
@media (max-width:900px) { main { grid-template-columns:1fr; } }
section.panel { background:var(--surface); border:1px solid var(--border);
  border-radius:12px; padding:12px 14px; min-height:200px; }
.panel h2 { font-size:13.5px; margin:0 0 8px; color:var(--ink2);
  text-transform:uppercase; letter-spacing:.6px; }
.panel h2 .n { color:var(--muted); font-weight:550; }
#board { display:flex; flex-direction:column; gap:8px; max-height:66vh; overflow-y:auto; }
.belief { border:1px solid var(--grid); border-radius:10px; padding:8px 10px; }
.belief.hot { border-color:var(--warn); }
.belief .k { font-family:ui-monospace,Menlo,monospace; font-size:12px; color:var(--ink2); }
.belief .v { font-weight:650; margin:1px 0 4px; }
.meter { height:6px; border-radius:3px; background:color-mix(in srgb, var(--wm) 15%, var(--surface));
  overflow:hidden; } .meter i { display:block; height:100%; border-radius:3px; }
.belief .m { display:flex; gap:8px; align-items:center; font-size:11.5px; color:var(--muted);
  font-family:ui-monospace,Menlo,monospace; margin-top:3px; }
.chip { font-size:11px; font-weight:650; border-radius:8px; padding:0 7px;
  border:1px solid var(--border); white-space:nowrap; }
.chip.auth { background:color-mix(in srgb, var(--wm) 14%, var(--surface)); }
.chip.weakish { background:color-mix(in srgb, var(--warn) 22%, var(--surface)); }
.chip.deb { background:color-mix(in srgb, var(--judge) 16%, var(--surface)); }
#feed { display:flex; flex-direction:column; gap:8px; max-height:66vh; overflow-y:auto; }
.act { border:1px solid var(--grid); border-radius:10px; padding:8px 11px; }
.act .who { display:flex; gap:8px; align-items:baseline; margin-bottom:3px; }
.role { font-size:11px; font-weight:700; letter-spacing:.5px; }
.model { font-family:ui-monospace,Menlo,monospace; font-size:11px; color:var(--muted); }
.act .body { font-size:13px; } .act .body em { color:var(--ink2); }
.gatechip { font-weight:650; font-size:12px; padding:1px 8px; border-radius:9px; }
.gatechip.commit { background:color-mix(in srgb, var(--good) 15%, var(--surface)); }
.gatechip.debate { background:color-mix(in srgb, var(--warn) 25%, var(--surface)); }
.okmark { color:var(--good); font-weight:700; } .badmark { color:var(--crit); font-weight:700; }
#term { background:var(--term); color:var(--termink); border-radius:12px;
  font-family:ui-monospace,Menlo,monospace; font-size:12px; padding:12px 14px;
  margin-top:12px; max-height:180px; overflow-y:auto; }
#term .t { color:#6f6e68; margin-right:8px; }
footer { color:var(--muted); font-size:12px; margin-top:14px; }
@media (prefers-reduced-motion: no-preference) {
  .belief, .act { transition:border-color .4s; } }
"""

JS_TEMPLATE = """
const R = __REPLAY__;
const MODELS = __MODELS__;
const evs = R.events;
let i = -1, playing = null, speed = 1;
const $ = id => document.getElementById(id);
const board = $('board'), feed = $('feed'), term = $('term');

function meterColor(p) { return p >= 0.7 ? 'var(--crit)' : p >= 0.35 ? 'var(--warn)' : 'var(--wm)'; }
function el(tag, cls, text) { const e = document.createElement(tag);
  if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }

function renderBoard(snap, hotKeys) {
  board.replaceChildren();
  const rows = [...snap].sort((a, b) => b.wrong_now - a.wrong_now);
  for (const b of rows) {
    const card = el('div', 'belief' + (hotKeys.has(b.key) ? ' hot' : ''));
    card.appendChild(el('div', 'k', b.key));
    const v = el('div', 'v', b.value);
    if (b.weak) { const c = el('span', 'chip weakish', '▲ weak source'); c.style.marginLeft = '8px'; v.appendChild(c); }
    if (b.source === 'debate') { const c = el('span', 'chip deb', '⚖ adjudicated'); c.style.marginLeft = '8px'; v.appendChild(c); }
    card.appendChild(v);
    const meter = el('div', 'meter'); const fill = el('i');
    fill.style.width = Math.max(2, b.wrong_now * 100) + '%';
    fill.style.background = meterColor(b.wrong_now);
    meter.appendChild(fill); card.appendChild(meter);
    const m = el('div', 'm');
    m.appendChild(el('span', '', `P(wrong) ${b.wrong_now.toFixed(2)}`));
    m.appendChild(el('span', '', `P(overturned) ${b.superseded_next.toFixed(2)}`));
    if (b.churn) m.appendChild(el('span', '', `churn ${b.churn}`));
    const src = el('span', 'chip' + (/filing|debate/i.test(b.source) ? ' auth' : ''), b.source || 'unsourced');
    m.appendChild(src);
    card.appendChild(m);
    board.appendChild(card);
  }
}

function actCard(role, color, model, bodyNodes) {
  const card = el('div', 'act');
  const who = el('div', 'who');
  const r = el('span', 'role', role); r.style.color = color;
  who.appendChild(r);
  if (model) who.appendChild(el('span', 'model', model));
  card.appendChild(who);
  const body = el('div', 'body');
  for (const n of bodyNodes) body.appendChild(n);
  card.appendChild(body);
  return card;
}

function log(msg) { const d = el('div');
  d.appendChild(el('span', 't', `[e${String(i).padStart(2,'0')}]`));
  d.appendChild(document.createTextNode(msg));
  term.appendChild(d); term.scrollTop = term.scrollHeight; }

function feedFor(e) {
  const cards = [];
  if (e.type === 'evidence') {
    const lines = e.lines.map(l => { const d = el('div'); d.appendChild(el('em', '', l)); return d; });
    cards.push(actCard('EVIDENCE ARRIVES', 'var(--muted)', null, lines));
    const outcomes = e.asserts.map(a => `${a.key} = ${a.value} (${a.outcome})`).join(' · ');
    cards.push(actCard('EXTRACTOR', 'var(--extractor)', MODELS.fast,
      [el('span', '', `asserted ${e.asserts.length} facts — ${outcomes}`)]));
    log(`evidence batch: ${e.lines.length} lines → ${e.asserts.length} asserts ($${e.cost_usd})`);
    return cards;
  }
  cards.push(actCard('QUESTION', 'var(--muted)', null, [el('span', '', e.question.split(' Policy:')[0])]));
  for (const t of e.events) {
    if (t.kind === 'proposal') cards.push(actCard('PROPOSER', 'var(--wm)', MODELS.strong,
      [el('span', '', `answers “${t.answer}” (confidence ${t.confidence}) from ${t.support.length} beliefs`)]));
    if (t.kind === 'challenge') cards.push(actCard('SKEPTIC', 'var(--skeptic)', MODELS.mid,
      [el('span', '', `attacks ${t.key}: `), el('em', '', t.attack)]));
    if (t.kind === 'verdict') cards.push(actCard('JUDGE', 'var(--judge)', MODELS.strong,
      [el('span', '', t.upheld ? `upholds ${t.key}` :
        `overturns ${t.key} → corrected to “${t.corrected}” (written back to the board)`)]));
    if (t.kind === 'reproposal') cards.push(actCard('PROPOSER (re-proposal)', 'var(--wm)', MODELS.strong,
      [el('span', '', `now answers “${t.answer}” from the corrected board`)]));
  }
  const g = e.gate;
  const chip = el('span', 'gatechip ' + (g.fired ? 'debate' : 'commit'),
    g.fired ? '▲ DEBATE' : '✓ COMMIT');
  const gateCard = actCard('WORLD-MODEL GATE', 'var(--ink)', '0 LLM calls', [chip,
    el('span', '', `  p(wrong)=${g.p_wrong} — ${g.reason}`)]);
  cards.splice(2, 0, gateCard); // after question+proposal
  const res = el('span', e.correct ? 'okmark' : 'badmark',
    `${e.correct ? '✓' : '✗'} answer “${e.answer}” (gold: ${e.gold}) · ${e.tokens} tokens · $${e.cost_usd}`);
  cards.push(actCard('RESULT', e.correct ? 'var(--good)' : 'var(--crit)', null, [res]));
  log(`${e.task_id}: ${g.fired ? 'DEBATE' : 'commit'} p=${g.p_wrong} → ${e.correct ? '✓' : '✗'} ($${e.cost_usd})`);
  return cards;
}

function apply(n, rebuild) {
  n = Math.max(0, Math.min(evs.length - 1, n));
  if (rebuild) { feed.replaceChildren(); term.replaceChildren(); }
  const from = rebuild ? 0 : i + 1;
  for (let j = from; j <= n; j++) { i = j; for (const c of feedFor(evs[j])) feed.appendChild(c); }
  i = n;
  const e = evs[n];
  const hot = new Set(e.type === 'evidence' ? e.asserts.map(a => a.key)
    : (e.events || []).filter(t => t.key).map(t => t.key));
  renderBoard(e.board, hot);
  feed.scrollTop = feed.scrollHeight;
  $('scrub').value = n;
  $('readout').textContent = `event ${n + 1}/${evs.length}`;
  let q = 0, ok = 0, fired = 0, cost = 0;
  for (let j = 0; j <= n; j++) { const x = evs[j]; cost += x.cost_usd || 0;
    if (x.type === 'question') { q++; ok += x.correct; fired += x.gate.fired; } }
  $('stats').innerHTML = '';
  for (const [k, v] of [['answered', `${ok}/${q}`], ['debates', fired],
                        ['spent', `$${cost.toFixed(4)}`], ['beliefs', e.board.length]]) {
    const s = el('span'); s.appendChild(el('b', '', String(v)));
    s.appendChild(document.createTextNode(' ' + k)); $('stats').appendChild(s); }
  $('live-status').textContent = `event ${n + 1} of ${evs.length}` +
    (e.type === 'question' ? ` — ${e.task_id} ${e.gate.fired ? 'debated' : 'committed'}` : ' — evidence batch');
}

function stop() { clearInterval(playing); playing = null; $('play').textContent = '▶ play'; }
$('play').addEventListener('click', () => {
  if (playing) return stop();
  if (i >= evs.length - 1) apply(0, true);
  $('play').textContent = '⏸ pause';
  playing = setInterval(() => { if (i >= evs.length - 1) return stop(); apply(i + 1); },
    1800 / speed);
});
$('speed').addEventListener('change', ev => { speed = +ev.target.value;
  if (playing) { stop(); $('play').click(); } });
$('scrub').addEventListener('input', ev => { stop(); apply(+ev.target.value, true); });
document.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'SELECT') return;
  if (ev.code === 'Space') { ev.preventDefault(); $('play').click(); }
  if (ev.key === 'ArrowRight') { stop(); apply(i + 1); }
  if (ev.key === 'ArrowLeft') { stop(); apply(i - 1, true); }
});
apply(0, true);
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=str(ROOT / "results" / "replay_s0.json"))
    args = ap.parse_args()
    replay = json.loads(Path(args.replay).read_text())
    s = replay["summary"]
    models = {"strong": MODEL_STRONG, "mid": MODEL_MID, "fast": MODEL_FAST}
    js = (JS_TEMPLATE
          .replace("__REPLAY__", json.dumps(replay))
          .replace("__MODELS__", json.dumps(models)))
    mark = ('<svg viewBox="0 0 64 64" width="34" height="34" aria-hidden="true">'
            '<rect x="4" y="4" width="56" height="56" rx="14" fill="none" '
            'stroke="var(--wm)" stroke-width="4.5"/>'
            '<rect x="16" y="18" width="32" height="6" rx="3" fill="var(--wm)"/>'
            '<rect x="16" y="29" width="16" height="6" rx="3" fill="var(--warn)"/>'
            '<path d="M42 26 l6 6 -6 6 -6 -6 z" fill="var(--warn)"/>'
            '<rect x="16" y="40" width="32" height="6" rx="3" fill="var(--wm)"/></svg>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Watch Agora's agent society process a contradictory evidence stream, with the learned world model's risk meters live.">
<title>Agora — society view (live replay)</title><style>{CSS}</style></head><body>
<a class="skip" href="#main">Skip to content</a>
<header>
{mark}<span class="word">agora</span><span class="view">/ society view</span>
<span class="status">REPLAY · seed {replay['seed']} · {replay['steps']} steps ·
{s['correct']}/{s['questions']} correct · ${s['total_cost_usd']} total · recorded {date.today().isoformat()}</span>
<nav aria-label="Product links"><a href="/">Benchmarks</a><a href="/docs">API playground</a></nav>
</header>
<p style="color:var(--ink2);max-width:78ch;margin:2px 0 6px">A real recorded run
(not a mock): dated filings, stale echoes and rumors arrive on the left of the
feed; the belief board's <strong>learned world model</strong> re-scores every
belief as they land; the gate spends debate only where P(wrong) spikes.
Space = play/pause, arrows = step.</p>
<div class="controls" role="group" aria-label="Replay controls">
<button id="play" type="button">▶ play</button>
<label>speed <select id="speed" aria-label="Playback speed">
<option value="1">1×</option><option value="2">2×</option><option value="4">4×</option>
</select></label>
<input id="scrub" type="range" min="0" max="{len(replay['events']) - 1}" value="0"
 aria-label="Timeline scrubber">
<span class="readout" id="readout"></span>
<span class="stats" id="stats"></span>
</div>
<div id="live-status" class="sr" role="status" aria-live="polite"
 style="position:absolute;left:-9999px"></div>
<main id="main">
<section class="panel" aria-label="Belief board">
<h2>Belief board <span class="n">— learned P(wrong) per belief</span></h2>
<div id="board"></div></section>
<section class="panel" aria-label="Society activity feed">
<h2>Society feed <span class="n">— extract · propose · gate · skeptic · judge</span></h2>
<div id="feed" role="log" aria-label="Agent activity"></div></section>
</main>
<div id="term" role="log" aria-label="System log"></div>
<footer>Backbones: proposer/judge {MODEL_STRONG} · skeptic {MODEL_MID} · extractor
{MODEL_FAST}. Gate: learned world model + conformal threshold, 0 LLM calls per
decision. Generated by scripts/build_live.py from results/replay_s0.json.</footer>
<script>{js}</script>
</body></html>"""
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page) // 1024}KB, {len(replay['events'])} events)")


if __name__ == "__main__":
    main()
