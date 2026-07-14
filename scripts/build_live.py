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
# Astryx neutral design tokens (vendored, see scripts/extract_astryx_tokens.py)
# + the Agora skin overrides + Figtree, inlined: one self-contained file.
TOKENS = ((ROOT / "web" / "astryx-tokens.css").read_text()
          + (ROOT / "web" / "agora-skin.css").read_text())
FONT = (ROOT / "web" / "figtree.css").read_text()

CSS = """
/* MiroFish-style research console: light-locked editorial chrome, the belief
   world as a living graph (canvas), numbered activity on the right, black
   system-dashboard strip below. Colorful data, quiet chrome. */
:root { color-scheme:light;
  --plane:#f7f7f8; --surface:#ffffff; --ink:#17171a; --ink2:#6e6e76;
  --faint:#a0a0a8; --hairline:rgba(17,17,20,.08); --hairline2:rgba(17,17,20,.16);
  --crimson:#d6336c; --blue:#2563c9; --orange:#e8590c; --purple:#6f42c1;
  --green:#2f9e44; --risk:#d13438;
  --good:var(--green); --warn:var(--orange); --crit:var(--risk);
  --wm:var(--blue); --extractor:#0ca678; --skeptic:var(--crimson); --judge:var(--purple);
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace; }
* { box-sizing:border-box; }
body { background:var(--plane); color:var(--ink); margin:0; padding:0;
  font:13.5px/1.5 Figtree,-apple-system,"Segoe UI",sans-serif; }
.page { max-width:1440px; margin:0 auto; padding:0 20px 20px; }
a { color:var(--ink2); text-decoration:none; } a:hover { color:var(--ink); }
.skip { position:absolute; left:-9999px; } .skip:focus { left:12px; top:8px;
  background:var(--surface); padding:8px 12px; border-radius:8px; z-index:9; }
:focus-visible { outline:2px solid var(--blue); outline-offset:2px; }
header { display:flex; align-items:center; gap:14px; flex-wrap:wrap;
  padding:14px 0 12px; border-bottom:1px solid var(--hairline); }
.word { font:700 15px/1 var(--mono); letter-spacing:.14em; }
.view { color:var(--faint); font-size:12.5px; }
.modes { display:flex; background:#ececef; border-radius:9px; padding:3px; gap:2px; }
.modes button { border:0; border-radius:7px; padding:4px 14px; font-size:12.5px;
  color:var(--ink2); background:transparent; cursor:pointer; }
.modes button[aria-pressed=true] { background:var(--surface); color:var(--ink);
  font-weight:600; box-shadow:0 1px 3px rgba(17,17,20,.12); }
.status { font-family:var(--mono); font-size:11px; color:var(--ink2);
  font-variant-numeric:tabular-nums; display:flex; align-items:center; gap:7px; }
.status::before { content:""; width:7px; height:7px; border-radius:50%;
  background:var(--green); flex:0 0 auto; }
nav { margin-left:auto; display:flex; gap:16px; font-size:12.5px; }
.lede { color:var(--ink2); max-width:80ch; margin:10px 0 2px; font-size:13px; }
.lede strong { color:var(--ink); font-weight:600; }
#society { display:flex; gap:14px; flex-wrap:wrap; margin:8px 0 2px;
  font-size:11.5px; color:var(--ink2); }
.agent { display:flex; gap:6px; align-items:center; opacity:.75; transition:opacity .25s; }
.agent .dot { width:7px; height:7px; border-radius:50%; flex:0 0 auto; }
.agent .nm { font-weight:600; color:var(--ink2); }
.agent .md { font-family:var(--mono); font-size:10.5px; color:var(--faint); }
.agent.on { opacity:1; } .agent.on .nm { color:var(--ink); }
.controls { display:flex; gap:12px; align-items:center; flex-wrap:wrap;
  padding:10px 0 12px; }
button, select { font:inherit; font-size:12.5px; color:var(--ink);
  background:var(--surface); border:1px solid var(--hairline2); border-radius:8px;
  padding:4px 12px; cursor:pointer; }
button:hover { border-color:var(--ink2); }
#play { background:var(--ink); color:#fff; border-color:var(--ink);
  font-weight:600; padding:5px 18px; min-width:88px; }
#play:hover { opacity:.85; }
input[type=range] { flex:1; min-width:140px; accent-color:var(--ink); }
.readout { font-family:var(--mono); font-size:11.5px; color:var(--ink2);
  min-width:104px; font-variant-numeric:tabular-nums; }
/* stat tiles, MiroFish-style big numbers */
.stats { display:flex; gap:10px; flex-wrap:wrap; }
.stats span { background:var(--surface); border:1px solid var(--hairline);
  border-radius:10px; padding:7px 16px 8px; display:grid; justify-items:center;
  gap:1px; font-size:10.5px; color:var(--ink2); min-width:86px; }
.stats b { font:600 17px/1.2 var(--mono); color:var(--ink);
  font-variant-numeric:tabular-nums; }
main { display:grid; grid-template-columns:minmax(400px,11fr) minmax(360px,9fr);
  gap:14px; }
@media (max-width:960px) { main { grid-template-columns:1fr; } }
section.panel { background:var(--surface); border:1px solid var(--hairline);
  border-radius:12px; padding:14px 16px; min-height:200px; }
.panel h2 { font-size:12.5px; font-weight:600; margin:0 0 8px; color:var(--ink); }
.panel h2 .n { color:var(--faint); font-weight:400; }
/* the graph: dot-grid canvas, floating legend + node card */
.graphwrap { position:relative; height:46vh; min-height:330px; margin:0 -16px;
  border-top:1px solid var(--hairline); border-bottom:1px solid var(--hairline);
  background-image:radial-gradient(circle, #dededf 1px, transparent 1.2px);
  background-size:18px 18px; overflow:hidden; }
#graph { position:absolute; inset:0; width:100%; height:100%; cursor:default; }
.legend { position:absolute; left:12px; bottom:12px; background:var(--surface);
  border:1px solid var(--hairline); border-radius:10px; padding:9px 12px;
  box-shadow:0 6px 20px rgba(17,17,20,.08); font-size:11px; color:var(--ink2); }
.legend .lt { font:600 10px/1 var(--mono); letter-spacing:.1em; color:var(--faint);
  text-transform:uppercase; margin-bottom:7px; }
.legend div.li { display:flex; align-items:center; gap:7px; margin:3px 0; }
.legend i { width:8px; height:8px; border-radius:50%; flex:0 0 auto; }
#nodecard { position:absolute; right:12px; top:12px; width:250px;
  background:var(--surface); border:1px solid var(--hairline); border-radius:12px;
  padding:12px 14px; box-shadow:0 10px 30px rgba(17,17,20,.14); z-index:4; }
#nodecard .nchead { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
#nodecard .nctitle { font-weight:600; font-size:12.5px; }
#nodecard .ncchip { font-size:10px; font-weight:600; border-radius:999px;
  padding:2px 9px; color:#fff; }
#ncclose { margin-left:auto; border:0; background:transparent; color:var(--faint);
  font-size:15px; padding:0 2px; line-height:1; }
#ncclose:hover { color:var(--ink); }
#nodecard .ncrow { display:flex; justify-content:space-between; gap:10px;
  font-size:11.5px; color:var(--ink2); padding:3px 0;
  border-top:1px solid var(--hairline); }
#nodecard .ncrow b { color:var(--ink); font-weight:600; text-align:right;
  font-variant-numeric:tabular-nums; }
#nodecard .ncrow .mono { font-family:var(--mono); font-size:10.5px; }
/* belief ledger under the graph */
#board { display:flex; flex-direction:column; max-height:24vh; overflow-y:auto;
  margin-top:4px; }
.belief { padding:8px 2px 9px 12px; border-bottom:1px solid var(--hairline);
  position:relative; }
.belief.hot::before { content:""; position:absolute; left:0; top:9px; bottom:11px;
  width:2px; background:var(--orange); border-radius:1px; }
.belief .k { font-family:var(--mono); font-size:11px; color:var(--faint); }
.belief .v { font-weight:600; font-size:13px; margin:2px 0 5px; }
.meter { height:2px; border-radius:1px; background:#ececef; overflow:hidden; }
.meter i { display:block; height:100%; }
.belief .m { display:flex; gap:12px; align-items:center; font-size:10.5px;
  color:var(--faint); font-family:var(--mono); font-variant-numeric:tabular-nums;
  margin-top:5px; }
.chip { font-size:10px; font-weight:600; border-radius:999px; padding:1px 8px;
  white-space:nowrap; margin-left:auto; background:#ececef; color:var(--ink2); }
.chip.auth { background:rgba(37,99,201,.12); color:var(--blue); }
.chip.weakish { background:rgba(232,89,12,.12); color:var(--orange); margin-left:0; }
.chip.deb { background:rgba(111,66,193,.12); color:var(--purple); margin-left:0; }
.belief .v .chip { margin-left:8px; }
/* feed: activation-sequence cards */
#feed { display:flex; flex-direction:column; gap:10px; max-height:74vh;
  overflow-y:auto; padding-right:4px; }
.turn { border:1px solid var(--hairline); border-radius:10px; padding:2px 12px; }
.turn.debated { border-color:rgba(214,51,108,.45);
  box-shadow:0 0 0 1px rgba(214,51,108,.14); }
.evblock { border:1px dashed var(--hairline2); border-radius:10px; padding:2px 12px; }
.act { padding:8px 0; }
.act + .act { border-top:1px solid var(--hairline); }
.act .body .ev { display:block; font-family:var(--mono); font-size:11px;
  color:var(--ink2); line-height:1.7; }
.act .who { display:flex; gap:8px; align-items:baseline; margin-bottom:2px; }
.role { font:600 10px/1.5 var(--mono); letter-spacing:.1em; text-transform:uppercase; }
.model { font-family:var(--mono); font-size:10.5px; color:var(--faint); margin-left:auto; }
.act .body { font-size:12.5px; color:var(--ink); } .act .body em { color:var(--ink2); }
.gatechip { font-weight:600; font-size:10.5px; padding:2px 9px; border-radius:999px;
  font-family:var(--mono); background:#ececef; color:var(--ink2); }
.gatechip.commit { background:rgba(47,158,68,.13); color:var(--green); }
.gatechip.debate { background:rgba(214,51,108,.13); color:var(--crimson); }
.act.gatecard .body { font-family:var(--mono); font-size:11px; }
.okmark { color:var(--green); font-weight:600; } .badmark { color:var(--risk); font-weight:600; }
/* black system-dashboard strip */
#termwrap { background:#101013; border-radius:12px; margin-top:14px; overflow:hidden; }
.termhead { display:flex; justify-content:space-between; align-items:center;
  padding:8px 16px 0; font:600 10px/1 var(--mono); letter-spacing:.12em;
  color:#8f8f96; }
#term { color:#c9c9cf; font-family:var(--mono); font-size:11.5px; line-height:1.8;
  padding:8px 16px 12px; max-height:150px; overflow-y:auto; }
#term .t { color:#5f5f66; margin-right:8px; }
footer { color:var(--faint); font-size:11.5px; margin-top:10px; }
#livebar { display:none; flex-direction:column; gap:8px; padding:12px 0; }
#livebar textarea { font:12px/1.7 var(--mono); color:var(--ink);
  background:var(--surface); border:1px solid var(--hairline2); border-radius:8px;
  padding:8px 10px; width:100%; min-height:74px; resize:vertical; }
#livebar .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
#livebar input[type=text], #livebar input[type=password] { font:inherit;
  font-size:12.5px; color:var(--ink); background:var(--surface);
  border:1px solid var(--hairline2); border-radius:8px; padding:6px 10px; }
#question { flex:1; min-width:240px; }
#token { width:150px; font-family:var(--mono); font-size:11.5px; }
#ingest-btn, #ask-btn { background:var(--ink); color:#fff; border-color:var(--ink);
  font-weight:600; }
#ingest-btn:disabled, #ask-btn:disabled { opacity:.5; cursor:wait; }
#livemsg { font-family:var(--mono); font-size:11.5px; color:var(--faint); }
@keyframes hotflash { from { opacity:.35; } to { opacity:1; } }
@media (prefers-reduced-motion: no-preference) {
  .belief.hot { animation:hotflash 1.2s ease-out; } }
"""

JS_TEMPLATE = """
const R = __REPLAY__;
const MODELS = __MODELS__;
const evs = R.events;
let i = -1, playing = null, speed = 1;
const $ = id => document.getElementById(id);
const board = $('board'), feed = $('feed'), term = $('term');

function meterColor(p) { return p >= 0.7 ? 'var(--crit)' : p >= 0.35 ? 'var(--gate-rail)' : 'var(--color-border-emphasized)'; }
function el(tag, cls, text) { const e = document.createElement(tag);
  if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }

function renderBoard(snap, hotKeys) {
  board.replaceChildren();
  const rows = [...snap].sort((a, b) => b.wrong_now - a.wrong_now);
  for (const b of rows) {
    const card = el('div', 'belief' + (hotKeys.has(b.key) ? ' hot' : ''));
    card.appendChild(el('div', 'k', b.key));
    const v = el('div', 'v', b.value);
    if (b.weak) { const c = el('span', 'chip weakish', 'weak source'); c.style.marginLeft = '8px'; v.appendChild(c); }
    if (b.source === 'debate') { const c = el('span', 'chip deb', 'adjudicated'); c.style.marginLeft = '8px'; v.appendChild(c); }
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

const AGENT_IDS = { 'Evidence arrives': 'ag-extractor', 'Extractor': 'ag-extractor',
  'Proposer': 'ag-proposer', 'Proposer (re-proposal)': 'ag-proposer',
  'World-model gate': 'ag-gate', 'Skeptic': 'ag-skeptic', 'Judge': 'ag-judge' };
function pulse(role) {
  const n = document.getElementById(AGENT_IDS[role] || '');
  if (!n) return;
  n.classList.add('on'); clearTimeout(n._t);
  n._t = setTimeout(() => n.classList.remove('on'), 1400);
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
  pulse(role);
  return card;
}
const thread = c => (c.classList.add('thread'), c);

function log(msg) { const d = el('div');
  d.appendChild(el('span', 't', `[e${String(i).padStart(2,'0')}]`));
  d.appendChild(document.createTextNode(msg));
  term.appendChild(d); term.scrollTop = term.scrollHeight; }

function feedFor(e) {
  const cards = [];
  if (e.type === 'evidence') {
    const lines = e.lines.map(l => { const d = el('div'); d.appendChild(el('span', 'ev', l)); return d; });
    cards.push(actCard('Evidence arrives', 'var(--muted)', null, lines));
    const outcomes = e.asserts.map(a => `${a.key} = ${a.value} (${a.outcome})`).join(' · ');
    cards.push(actCard('Extractor', 'var(--extractor)', MODELS.fast,
      [el('span', '', `asserted ${e.asserts.length} facts — ${outcomes}`)]));
    log(`evidence batch: ${e.lines.length} lines → ${e.asserts.length} asserts ($${e.cost_usd})`);
    const block = el('div', 'evblock');
    for (const c of cards) block.appendChild(c);
    return [block];
  }
  cards.push(actCard('Question', 'var(--muted)', null, [el('span', '', e.question.split(' Policy:')[0])]));
  for (const t of e.events) {
    if (t.kind === 'proposal') cards.push(actCard('Proposer', 'var(--wm)', MODELS.strong,
      [el('span', '', `answers “${t.answer}” (confidence ${t.confidence}) from ${t.support.length} beliefs`)]));
    if (t.kind === 'challenge') cards.push(thread(actCard('Skeptic', 'var(--skeptic)', MODELS.mid,
      [el('span', '', `attacks ${t.key}: `), el('em', '', t.attack)])));
    if (t.kind === 'verdict') cards.push(thread(actCard('Judge', 'var(--judge)', MODELS.strong,
      [el('span', '', t.upheld ? `upholds ${t.key}` :
        `overturns ${t.key} → corrected to “${t.corrected}” (written back to the board)`)])));
    if (t.kind === 'reproposal') cards.push(thread(actCard('Proposer (re-proposal)', 'var(--wm)', MODELS.strong,
      [el('span', '', `now answers “${t.answer}” from the corrected board`)])));
  }
  const g = e.gate;
  const chip = el('span', 'gatechip ' + (g.fired ? 'debate' : 'commit'),
    g.fired ? 'debate' : 'commit');
  const gateCard = actCard('World-model gate', 'var(--ink)', '0 LLM calls', [chip,
    el('span', '', `  p(wrong)=${g.p_wrong} — ${g.reason}`)]);
  gateCard.classList.add('gatecard');
  cards.splice(2, 0, gateCard); // after question+proposal
  const graded = e.correct !== null && e.correct !== undefined; // live asks have no gold
  const res = graded
    ? el('span', e.correct ? 'okmark' : 'badmark',
        `${e.correct ? '✓' : '✗'} answer “${e.answer}” (gold: ${e.gold}) · ${e.tokens} tokens · $${e.cost_usd}`)
    : el('span', '', `answer “${e.answer}” · ${e.tokens} tokens · $${e.cost_usd}`);
  cards.push(actCard('Result',
    graded ? (e.correct ? 'var(--good)' : 'var(--crit)') : 'var(--ink)', null, [res]));
  log(`${e.task_id}: ${g.fired ? 'DEBATE' : 'commit'} p=${g.p_wrong}` +
      (graded ? ` → ${e.correct ? '✓' : '✗'}` : '') + ` ($${e.cost_usd})`);
  const turn = el('div', 'turn' + (g.fired ? ' debated' : ''));
  for (const c of cards) turn.appendChild(c);
  return [turn];
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
  updateGraph(e);
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

function stop() { clearInterval(playing); playing = null; $('play').textContent = 'Play'; }
$('play').addEventListener('click', () => {
  if (playing) return stop();
  if (i >= evs.length - 1) apply(0, true);
  $('play').textContent = 'Pause';
  playing = setInterval(() => { if (i >= evs.length - 1) return stop(); apply(i + 1); },
    1800 / speed);
});
$('speed').addEventListener('change', ev => { speed = +ev.target.value;
  if (playing) { stop(); $('play').click(); } });
$('scrub').addEventListener('input', ev => { stop(); apply(+ev.target.value, true); });
document.addEventListener('keydown', ev => {
  if (mode !== 'replay') return;
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(ev.target.tagName)) return;
  if (ev.code === 'Space') { ev.preventDefault(); $('play').click(); }
  if (ev.key === 'ArrowRight') { stop(); apply(i + 1); }
  if (ev.key === 'ArrowLeft') { stop(); apply(i - 1, true); }
});


// ---- society graph: beliefs + agents on a force-laid canvas ----
const gc = $('graph'), gctx = gc.getContext('2d');
const G = { nodes: new Map(), fx: [], alpha: 0, w: 0, h: 0, sel: null };
const AGENTS = [
  { id: 'EXTRACTOR', color: '#0ca678' }, { id: 'PROPOSER', color: '#2563c9' },
  { id: 'GATE', color: '#e8a33d' }, { id: 'SKEPTIC', color: '#d6336c' },
  { id: 'JUDGE', color: '#6f42c1' }];
function srcColor(src, weak) {
  if (/debate/i.test(src || '')) return '#6f42c1';
  if (weak || /rumor|blog|forum/i.test(src || '')) return '#e8590c';
  if (/filing/i.test(src || '')) return '#2563c9';
  return '#8a8a92';
}
function sizeGraph() {
  const r = gc.parentElement.getBoundingClientRect();
  G.w = r.width; G.h = r.height;
  const dpr = devicePixelRatio || 1;
  gc.width = G.w * dpr; gc.height = G.h * dpr;
  gctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  AGENTS.forEach((a, k) => {   // pinned pentagon, slightly high
    const th = -Math.PI / 2 + k * 2 * Math.PI / AGENTS.length;
    a.x = G.w / 2 + Math.cos(th) * Math.min(G.w, G.h) * 0.40;
    a.y = G.h / 2 + Math.sin(th) * Math.min(G.h, G.w) * 0.38; });
  G.alpha = 1;
}
new ResizeObserver(sizeGraph).observe(gc.parentElement);
function updateGraph(e) {
  const seen = new Set();
  let k = 0;
  for (const b of e.board) {
    seen.add(b.key);
    let n = G.nodes.get(b.key);
    if (!n) {                                  // enter near the extractor
      n = { x: AGENTS[0].x + (Math.random() - 0.5) * 60,
            y: AGENTS[0].y + (Math.random() - 0.5) * 60, vx: 0, vy: 0, flash: 0 };
      G.nodes.set(b.key, n);
    }
    n.b = b; n.ent = b.key.split('::')[0]; n.ord = k++;
  }
  for (const key of [...G.nodes.keys()]) if (!seen.has(key)) G.nodes.delete(key);
  const hit = key => { const n = G.nodes.get(key); if (n) n.flash = 90; return n; };
  if (e.type === 'evidence') {
    for (const a of e.asserts) { hit(a.key);
      G.fx.push({ from: AGENTS[0], key: a.key, color: '#0ca678', ttl: 110 }); }
  } else {
    for (const t of e.events || []) {
      if (t.kind === 'proposal') for (const sk of (t.support || []))
        G.fx.push({ from: AGENTS[1], key: sk, color: '#2563c9', ttl: 110 });
      if (t.kind === 'challenge' && t.key)
        G.fx.push({ from: AGENTS[3], key: t.key, color: '#d6336c', ttl: 150 }), hit(t.key);
      if (t.kind === 'verdict' && t.key)
        G.fx.push({ from: AGENTS[4], key: t.key, color: '#6f42c1', ttl: 150 }), hit(t.key);
    }
    if (e.gate && e.gate.fired) G.fx.push({ from: AGENTS[2], key: null,
      color: '#e8a33d', ttl: 130 });
  }
  if (G.fx.length > 40) G.fx.splice(0, G.fx.length - 40);
  G.alpha = 1;
}
function stepPhysics() {
  const ns = [...G.nodes.values()];
  const cents = {};                             // entity centroids
  for (const n of ns) { (cents[n.ent] ||= { x: 0, y: 0, c: 0 });
    cents[n.ent].x += n.x; cents[n.ent].y += n.y; cents[n.ent].c++; }
  for (const c of Object.values(cents)) { c.x /= c.c; c.y /= c.c; }
  for (const n of ns) {
    let fx = (G.w / 2 - n.x) * 0.0060, fy = (G.h / 2 - n.y) * 0.0060;
    for (const m of ns) { if (m === n) continue;
      const dx = n.x - m.x, dy = n.y - m.y, d2 = dx * dx + dy * dy + 40;
      const f = (n.ent === m.ent ? 900 : 1700) / d2;
      fx += dx * f / Math.sqrt(d2); fy += dy * f / Math.sqrt(d2); }
    const c = cents[n.ent];
    fx += (c.x - n.x) * 0.012; fy += (c.y - n.y) * 0.012;
    for (const a of AGENTS) {                   // keep clear of the agent ring
      const dx = n.x - a.x, dy = n.y - a.y, d2 = dx * dx + dy * dy + 40;
      const af = 5200 / d2;
      fx += dx * af / Math.sqrt(d2); fy += dy * af / Math.sqrt(d2); }
    n.vx = (n.vx + fx) * 0.82; n.vy = (n.vy + fy) * 0.82;
    n.x += n.vx * G.alpha * 2.2; n.y += n.vy * G.alpha * 2.2;
    n.x = Math.max(46, Math.min(G.w - 46, n.x));
    n.y = Math.max(40, Math.min(G.h - 34, n.y));
  }
  G.alpha = Math.max(0, G.alpha - 0.004);
}
function drawGraph() {
  gctx.clearRect(0, 0, G.w, G.h);
  const ns = [...G.nodes.values()];
  gctx.lineWidth = 1; gctx.strokeStyle = 'rgba(17,17,20,.10)';
  const byEnt = {};
  for (const n of ns) (byEnt[n.ent] ||= []).push(n);
  for (const g of Object.values(byEnt))         // same-entity hairlines
    for (let a = 0; a < g.length; a++) for (let b = a + 1; b < g.length; b++) {
      gctx.beginPath(); gctx.moveTo(g[a].x, g[a].y); gctx.lineTo(g[b].x, g[b].y);
      gctx.stroke(); }
  for (const f of G.fx) {                       // agent -> belief action edges
    const to = f.key ? G.nodes.get(f.key) : { x: G.w / 2, y: G.h / 2 };
    if (!to) continue;
    const a = Math.min(1, f.ttl / 90);
    gctx.strokeStyle = f.color + Math.round(a * 200).toString(16).padStart(2, '0');
    gctx.lineWidth = 1.6;
    const mx = (f.from.x + to.x) / 2, my = (f.from.y + to.y) / 2 - 26;
    gctx.beginPath(); gctx.moveTo(f.from.x, f.from.y);
    gctx.quadraticCurveTo(mx, my, to.x, to.y); gctx.stroke();
    f.ttl -= 1;
  }
  G.fx = G.fx.filter(f => f.ttl > 0);
  gctx.font = '9.5px ui-monospace,Menlo,monospace'; gctx.textAlign = 'center';
  for (const [ent, g] of Object.entries(byEnt)) {  // entity label
    const cx = Math.max(50, Math.min(G.w - 50,
      g.reduce((s, n) => s + n.x, 0) / g.length));
    const cy = Math.max(12, Math.min(...g.map(n => n.y)) - 16);
    gctx.fillStyle = 'rgba(17,17,20,.42)';
    gctx.fillText(ent, cx, cy);
  }
  for (const n of ns) {
    const b = n.b, r = 6 + Math.min(b.churn || 0, 4) * 1.6;
    if (n.flash > 0) { gctx.beginPath();       // amber halo on touch
      gctx.arc(n.x, n.y, r + 7 + (n.flash % 30) / 6, 0, 7);
      gctx.strokeStyle = 'rgba(232,163,61,' + (n.flash / 160) + ')';
      gctx.lineWidth = 2; gctx.stroke(); n.flash -= 1; }
    if (b.wrong_now > 0.05) { gctx.beginPath(); // the risk ring
      gctx.arc(n.x, n.y, r + 3.5, 0, 7);
      gctx.strokeStyle = 'rgba(209,52,56,' + (0.25 + b.wrong_now * 0.75) + ')';
      gctx.lineWidth = 1.5 + b.wrong_now * 2.5; gctx.stroke(); }
    gctx.beginPath(); gctx.arc(n.x, n.y, r, 0, 7);
    gctx.fillStyle = srcColor(b.source, b.weak); gctx.fill();
    if (G.sel === b.key) { gctx.beginPath(); gctx.arc(n.x, n.y, r + 6, 0, 7);
      gctx.strokeStyle = '#17171a'; gctx.lineWidth = 1.2; gctx.stroke(); }
    gctx.fillStyle = 'rgba(17,17,20,.55)';
    gctx.fillText(b.key.split('::')[1] || '',
      Math.max(44, Math.min(G.w - 44, n.x)), n.y + r + 11);
  }
  gctx.font = '600 9px ui-monospace,Menlo,monospace';
  for (const a of AGENTS) {                     // pinned agents
    gctx.beginPath(); gctx.arc(a.x, a.y, 9, 0, 7);
    gctx.fillStyle = '#ffffff'; gctx.fill();
    gctx.strokeStyle = a.color; gctx.lineWidth = 2; gctx.stroke();
    gctx.fillStyle = a.color; gctx.fillText(a.id, a.x, a.y - 15);
  }
}
(function loop() { stepPhysics(); drawGraph(); requestAnimationFrame(loop); })();
gc.addEventListener('click', ev => {
  const r = gc.getBoundingClientRect();
  const x = ev.clientX - r.left, y = ev.clientY - r.top;
  let best = null, bd = 18 * 18;
  for (const [key, n] of G.nodes) { const d = (n.x - x) ** 2 + (n.y - y) ** 2;
    if (d < bd) { bd = d; best = key; } }
  G.sel = best;
  const card = $('nodecard');
  if (!best) { card.hidden = true; return; }
  const b = G.nodes.get(best).b;
  $('nckey').textContent = b.key;
  $('ncsrc').textContent = b.source || 'unsourced';
  $('ncsrc').style.background = srcColor(b.source, b.weak);
  $('ncval').textContent = b.value;
  $('ncw').textContent = b.wrong_now.toFixed(2);
  $('ncs').textContent = b.superseded_next.toFixed(2);
  $('ncc').textContent = String(b.churn || 0);
  card.hidden = false;
});
$('ncclose').addEventListener('click', () => { G.sel = null; $('nodecard').hidden = true; });
sizeGraph();

// ---- live mode: your own evidence through the real society ----
const liveSid = 'live-' + Math.random().toString(36).slice(2, 10);
let mode = 'replay', liveN = 0, liveCost = 0, liveQ = 0, liveFired = 0, busy = false;
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
$('token').value = localStorage.getItem('agora-token') || '';
$('token').addEventListener('change', ev => localStorage.setItem('agora-token', ev.target.value));

function liveMsg(s) { $('livemsg').textContent = s; }
function setBusy(b) { busy = b; $('ingest-btn').disabled = b; $('ask-btn').disabled = b; }

function setMode(m) {
  mode = m; stop();
  $('mode-replay').setAttribute('aria-pressed', String(m === 'replay'));
  $('mode-live').setAttribute('aria-pressed', String(m === 'live'));
  document.querySelector('.controls').style.display = m === 'replay' ? 'flex' : 'none';
  $('livebar').style.display = m === 'live' ? 'flex' : 'none';
  feed.replaceChildren(); term.replaceChildren(); board.replaceChildren();
  G.nodes.clear(); G.fx = []; G.sel = null; $('nodecard').hidden = true;
  if (m === 'replay') { i = -1; apply(0, true); }
  else { i = liveN;
    liveMsg('your own evidence, the real society, the real learned gate — every call spends real tokens');
    log('live session started — feed evidence, then ask'); }
}

async function liveCall(path, payload) {
  setBusy(true); liveMsg('society working — real Qwen calls in flight…');
  try {
    const headers = { 'Content-Type': 'application/json' };
    const tok = $('token').value.trim();
    if (tok) headers['X-Agora-Token'] = tok;
    const r = await fetch(path, { method: 'POST', headers, body: JSON.stringify(payload) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    return data;
  } finally { setBusy(false); }
}

function liveRender(e) {
  liveN += 1; i = liveN; liveCost += e.cost_usd || 0;
  if (e.type === 'question') { liveQ += 1; liveFired += e.gate.fired ? 1 : 0; }
  const cards = feedFor(e);
  const step = REDUCED ? 0 : 380;
  cards.forEach((c, k) => setTimeout(() => {
    feed.appendChild(c); feed.scrollTop = feed.scrollHeight; }, k * step));
  setTimeout(() => {
    const hot = new Set(e.type === 'evidence' ? e.asserts.map(a => a.key)
      : (e.events || []).filter(t => t.key).map(t => t.key));
    renderBoard(e.board, hot);
    updateGraph(e);
    liveMsg(`${e.board.length} beliefs · ${liveQ} asked · ${liveFired} debates · $${liveCost.toFixed(4)} this session`);
  }, cards.length * step);
}

$('ingest-btn').addEventListener('click', async () => {
  if (busy) return;
  const lines = $('evidence').value.split('\\n').map(s => s.trim()).filter(Boolean);
  if (!lines.length) return liveMsg('paste at least one evidence line');
  try { liveRender((await liveCall('/ingest', { session_id: liveSid, lines })).event); }
  catch (err) { liveMsg(String(err.message || err)); log(`ERROR ${err.message || err}`); }
});
$('ask-btn').addEventListener('click', async () => {
  if (busy) return;
  const q = $('question').value.trim();
  if (!q) return liveMsg('type a claim or question first');
  try { liveRender((await liveCall('/ask', { session_id: liveSid, question: q })).event); }
  catch (err) { liveMsg(String(err.message || err)); log(`ERROR ${err.message || err}`); }
});
$('mode-replay').addEventListener('click', () => setMode('replay'));
$('mode-live').addEventListener('click', () => setMode('live'));
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
    mark = ('<svg viewBox="0 0 64 64" width="32" height="32" aria-hidden="true">'
            '<rect x="4" y="4" width="56" height="56" rx="14" fill="none" '
            'stroke="currentColor" stroke-width="4.5"/>'
            '<rect x="16" y="18" width="32" height="6" rx="3" fill="currentColor"/>'
            '<rect x="16" y="29" width="16" height="6" rx="3" fill="currentColor" opacity=".45"/>'
            '<path d="M42 26 l6 6 -6 6 -6 -6 z" fill="currentColor" opacity=".45"/>'
            '<rect x="16" y="40" width="32" height="6" rx="3" fill="currentColor"/></svg>')
    favicon = ('data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 '
               'viewBox=%220 0 64 64%22%3E%3Crect x=%224%22 y=%224%22 width=%2256%22 '
               'height=%2256%22 rx=%2214%22 fill=%22none%22 stroke=%22%23171717%22 '
               'stroke-width=%224.5%22/%3E%3Crect x=%2216%22 y=%2218%22 width=%2232%22 '
               'height=%226%22 rx=%223%22 fill=%22%23171717%22/%3E%3Crect x=%2216%22 '
               'y=%2229%22 width=%2216%22 height=%226%22 rx=%223%22 fill=%22%23737373%22/%3E'
               '%3Cpath d=%22M42 26 l6 6 -6 6 -6 -6 z%22 fill=%22%23737373%22/%3E%3Crect '
               'x=%2216%22 y=%2240%22 width=%2232%22 height=%226%22 rx=%223%22 '
               'fill=%22%23171717%22/%3E%3C/svg%3E')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Watch Agora's agent society process a contradictory evidence stream, with the learned world model's risk meters live.">
<link rel="icon" href="{favicon}">
<title>Agora — society view (live replay)</title>
<style>{TOKENS}{FONT}{CSS}</style></head><body>
<a class="skip" href="#main">Skip to content</a>
<div class="page">
<header>
{mark}<span class="word">AGORA</span><span class="view">/ society view</span>
<div class="modes" role="group" aria-label="Viewer mode">
<button id="mode-replay" type="button" aria-pressed="true">recorded run</button>
<button id="mode-live" type="button" aria-pressed="false">live — try it</button>
</div>
<span class="status">seed {replay['seed']} · {replay['steps']} steps ·
{s['correct']}/{s['questions']} correct · ${s['total_cost_usd']} total · recorded {date.today().isoformat()}</span>
<nav aria-label="Product links"><a href="/">Benchmarks</a><a href="/docs">API playground</a><a href="/zh/live" hreflang="zh-CN" lang="zh-CN">中文</a></nav>
</header>
<p class="lede">A real recorded run
(not a mock): dated filings, stale echoes and rumors arrive on the left of the
feed; the belief board's <strong>learned world model</strong> re-scores every
belief as they land; the gate spends debate only where P(wrong) spikes.
Space = play/pause, arrows = step. Or switch to <strong>live</strong> and feed
the society your own evidence.</p>
<div id="society" aria-label="The society">
<span class="agent" id="ag-extractor" style="--c:var(--extractor)"><i class="dot" style="background:var(--extractor)"></i><span class="nm">Extractor</span><span class="md">{MODEL_FAST}</span></span>
<span class="agent" id="ag-proposer" style="--c:var(--wm)"><i class="dot" style="background:var(--wm)"></i><span class="nm">Proposer</span><span class="md">{MODEL_STRONG}</span></span>
<span class="agent" id="ag-gate" style="--c:var(--warn)"><i class="dot" style="background:var(--warn)"></i><span class="nm">WM gate</span><span class="md">0 LLM calls</span></span>
<span class="agent" id="ag-skeptic" style="--c:var(--skeptic)"><i class="dot" style="background:var(--skeptic)"></i><span class="nm">Skeptic</span><span class="md">{MODEL_MID}</span></span>
<span class="agent" id="ag-judge" style="--c:var(--judge)"><i class="dot" style="background:var(--judge)"></i><span class="nm">Judge</span><span class="md">{MODEL_STRONG}</span></span>
</div>
<div class="controls" role="group" aria-label="Replay controls">
<button id="play" type="button">Play</button>
<label>speed <select id="speed" aria-label="Playback speed">
<option value="1">1×</option><option value="2">2×</option><option value="4">4×</option>
</select></label>
<input id="scrub" type="range" min="0" max="{len(replay['events']) - 1}" value="0"
 aria-label="Timeline scrubber">
<span class="readout" id="readout"></span>
<span class="stats" id="stats"></span>
</div>
<div id="livebar" role="group" aria-label="Live session controls">
<label for="evidence" style="font-size:var(--font-size-sm);font-weight:var(--font-weight-semibold);color:var(--muted)">Evidence — one dated line each, filings beat rumors</label>
<textarea id="evidence" spellcheck="false">[Jan 2026] Filing: Acme Robotics's CEO is Jane Doe.
[Mar 2026] Rumor: acme robotics's ceo is John Roe.</textarea>
<div class="row">
<button id="ingest-btn" type="button">Feed the society</button>
<input id="question" type="text" value="Claim: &quot;Acme Robotics's CEO is currently John Roe.&quot; Policy: filings are authoritative; rumors are unreliable and never override a filing. True or false?" aria-label="Question or claim">
<button id="ask-btn" type="button">Ask</button>
<input id="token" type="password" placeholder="token (optional)" aria-label="Access token" autocomplete="off">
</div>
<span id="livemsg">anonymous callers share a small daily budget — real Qwen calls, ~$0.01 per question</span>
</div>
<div id="live-status" class="sr" role="status" aria-live="polite"
 style="position:absolute;left:-9999px"></div>
<main id="main">
<section class="panel" aria-label="Belief board">
<h2>Belief board <span class="n">— learned P(wrong) per belief</span></h2>
<div class="graphwrap">
<canvas id="graph" aria-label="Belief graph"></canvas>
<div class="legend"><div class="lt">Belief graph</div>
<div class="li"><i style="background:var(--blue)"></i>filing</div>
<div class="li"><i style="background:var(--orange)"></i>rumor / weak</div>
<div class="li"><i style="background:var(--purple)"></i>adjudicated</div>
<div class="li"><i style="background:var(--risk)"></i>ring = P(wrong)</div>
<div class="li"><i style="background:var(--crimson)"></i>debate edge</div></div>
<div id="nodecard" hidden>
<div class="nchead"><span class="nctitle" id="nckey">—</span>
<span class="ncchip" id="ncsrc"></span>
<button id="ncclose" type="button" aria-label="Close">×</button></div>
<div class="ncrow"><span>value</span><b id="ncval"></b></div>
<div class="ncrow"><span>P(wrong)</span><b id="ncw"></b></div>
<div class="ncrow"><span>P(overturned)</span><b id="ncs"></b></div>
<div class="ncrow"><span>churn</span><b id="ncc"></b></div>
</div>
</div>
<div id="board"></div></section>
<section class="panel" aria-label="Society activity feed">
<h2>Society feed <span class="n">— extract · propose · gate · skeptic · judge</span></h2>
<div id="feed" role="log" aria-label="Agent activity"></div></section>
</main>
<div id="termwrap"><div class="termhead"><span>SYSTEM DASHBOARD</span><span>agora society runtime</span></div>
<div id="term" role="log" aria-label="System log"></div></div>
<footer>Backbones: proposer/judge {MODEL_STRONG} · skeptic {MODEL_MID} · extractor
{MODEL_FAST}. Gate: learned world model + conformal threshold, 0 LLM calls per
decision. Generated by scripts/build_live.py from results/replay_s0.json.</footer>
</div>
<script>{js}</script>
</body></html>"""
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page) // 1024}KB, {len(replay['events'])} events)")


if __name__ == "__main__":
    main()
