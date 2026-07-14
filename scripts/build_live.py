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
/* Trace-room skin: near-black canvas, surface ladder, 7% hairlines, mono for
   data only. One accent: gate amber (the society spending money). Belief
   board = ledger rows; feed = threaded turns with a cause rail. */
:root { --surface:var(--color-background-surface); --plane:var(--color-background-body);
  --ink:var(--color-text-primary); --ink2:var(--color-text-secondary);
  --muted:var(--color-text-secondary); --faint:var(--color-text-disabled);
  --grid:var(--color-border); --border:var(--color-border);
  --hairline:var(--color-border); --accent:var(--color-accent);
  --good:var(--color-success); --warn:var(--gate); --crit:var(--color-error);
  --wm:light-dark(#3b6ea8,#7da7d9); --extractor:light-dark(#2e7d5b,#5fb894);
  --skeptic:light-dark(#a34d4d,#cc7a7a); --judge:light-dark(#5c4fa3,#9d92d0);
  --mono:var(--font-family-code); }
* { box-sizing:border-box; }
body { background:var(--plane); color:var(--ink); margin:0 auto; max-width:1280px;
  padding:18px 22px 30px; font:13.5px/1.5 var(--font-family-body); }
a { color:var(--ink2); text-decoration:none; }
a:hover { color:var(--ink); }
.skip { position:absolute; left:-9999px; } .skip:focus { left:12px; top:8px;
  background:var(--surface); padding:8px 12px; border-radius:6px; z-index:9; }
:focus-visible { outline:2px solid var(--color-border-emphasized); outline-offset:2px; }
header { display:flex; align-items:center; gap:14px; flex-wrap:wrap;
  padding-bottom:12px; border-bottom:1px solid var(--hairline); }
.word { font-size:17px; font-weight:var(--font-weight-semibold); letter-spacing:-.02em; }
.view { color:var(--faint); font-size:13px; }
.status { font-family:var(--mono); font-size:11px; color:var(--muted);
  font-variant-numeric:tabular-nums; letter-spacing:.01em; }
nav { margin-left:auto; display:flex; gap:16px; font-size:12.5px; }
.lede { color:var(--ink2); max-width:74ch; margin:12px 0 4px; }
.lede strong { color:var(--ink); font-weight:var(--font-weight-semibold); }
#society { display:flex; gap:14px; flex-wrap:wrap; margin:10px 0 4px;
  font-size:11.5px; color:var(--muted); }
.agent { display:flex; gap:6px; align-items:center; opacity:.65;
  transition:opacity .25s; }
.agent .dot { width:7px; height:7px; border-radius:50%; flex:0 0 auto; }
.agent .nm { font-weight:var(--font-weight-semibold); color:var(--ink2); }
.agent .md { font-family:var(--mono); font-size:10.5px; color:var(--faint); }
.agent.on { opacity:1; } .agent.on .nm { color:var(--ink); }
.controls { display:flex; gap:14px; align-items:center; flex-wrap:wrap;
  padding:10px 0; margin:6px 0 10px; border-bottom:1px solid var(--hairline); }
button, select { font:inherit; font-size:12.5px; color:var(--ink);
  background:transparent; border:1px solid var(--color-border-emphasized);
  border-radius:6px; padding:4px 12px; cursor:pointer; }
button:hover { background:var(--color-background-muted); }
#play { background:var(--ink); color:var(--plane); border-color:var(--ink);
  font-weight:var(--font-weight-semibold); padding:5px 18px; min-width:88px; }
#play:hover { opacity:.85; background:var(--ink); }
input[type=range] { flex:1; min-width:160px; accent-color:var(--gate-rail); }
.readout { font-family:var(--mono); font-size:11.5px; color:var(--muted);
  min-width:110px; font-variant-numeric:tabular-nums; }
.stats { display:flex; gap:16px; font-family:var(--mono); font-variant-numeric:tabular-nums;
  font-size:11.5px; color:var(--muted); flex-wrap:wrap; }
.stats b { color:var(--ink); font-weight:var(--font-weight-semibold); }
main { display:grid; grid-template-columns:minmax(330px,5fr) minmax(380px,7fr); gap:22px; }
@media (max-width:900px) { main { grid-template-columns:1fr; } }
section.panel { min-height:200px; }
.panel h2 { font-size:12px; font-weight:var(--font-weight-semibold); margin:0 0 2px;
  color:var(--ink); }
.panel h2 .n { color:var(--faint); font-weight:var(--font-weight-normal); }
/* belief ledger: hairline rows, risk bar, no boxes */
#board { display:flex; flex-direction:column; max-height:66vh; overflow-y:auto; }
.belief { padding:9px 2px 10px 12px; border-bottom:1px solid var(--hairline);
  position:relative; }
.belief.hot::before { content:""; position:absolute; left:0; top:10px;
  bottom:12px; width:2px; background:var(--gate-rail); border-radius:1px; }
.belief .k { font-family:var(--mono); font-size:11px; color:var(--faint); }
.belief .v { font-weight:var(--font-weight-semibold); font-size:13.5px; margin:2px 0 6px; }
.meter { height:2px; border-radius:1px; background:var(--color-background-muted);
  overflow:hidden; }
.meter i { display:block; height:100%; }
.belief .m { display:flex; gap:12px; align-items:center; font-size:10.5px;
  color:var(--faint); font-family:var(--mono); font-variant-numeric:tabular-nums;
  margin-top:5px; }
.chip { font-size:10px; font-weight:var(--font-weight-semibold); border-radius:4px;
  padding:1px 6px; white-space:nowrap; margin-left:auto;
  background:var(--color-background-muted); color:var(--ink2); }
.chip.auth { color:var(--ink); }
.chip.weakish { background:var(--gate-bg); color:var(--gate); margin-left:0; }
.chip.deb { background:var(--color-background-muted); color:var(--judge); margin-left:0; }
.belief .v .chip { margin-left:8px; }
/* feed: threaded turns; the rail is the cause (gray commit, amber debate) */
#feed { display:flex; flex-direction:column; gap:14px; max-height:66vh;
  overflow-y:auto; padding-right:4px; }
.turn { border-left:2px solid var(--color-border-emphasized); padding:2px 0 2px 14px; }
.turn.debated { border-left-color:var(--gate-rail); }
.evblock { padding:2px 0 2px 14px; border-left:2px solid transparent; }
.act { padding:6px 0; }
.act + .act { border-top:1px solid var(--hairline); }
.act.thread {}
.act.gatecard { background:transparent; }
.turn.debated .act.gatecard .gatechip.debate { background:var(--gate-bg);
  color:var(--gate); }
.act .body .ev { display:block; font-family:var(--mono); font-size:11.5px;
  color:var(--ink2); line-height:1.7; }
.act .who { display:flex; gap:8px; align-items:baseline; margin-bottom:2px; }
.role { font-size:11px; font-weight:var(--font-weight-semibold); }
.model { font-family:var(--mono); font-size:10.5px; color:var(--faint); margin-left:auto; }
.act .body { font-size:13px; color:var(--ink2); } .act .body em { color:var(--ink2); }
.gatechip { font-weight:var(--font-weight-semibold); font-size:11px; padding:1px 7px;
  border-radius:4px; font-family:var(--mono);
  background:var(--color-background-muted); color:var(--ink2); }
.gatechip.commit { background:var(--color-success-muted); color:var(--good); }
.gatechip.debate { background:var(--gate-bg); color:var(--gate); }
.act.gatecard .body { font-family:var(--mono); font-size:11.5px; }
.okmark { color:var(--good); font-weight:var(--font-weight-semibold); }
.badmark { color:var(--crit); font-weight:var(--font-weight-semibold); }
#term { background:var(--surface); color:var(--muted);
  border:1px solid var(--hairline); border-radius:8px;
  font-family:var(--mono); font-size:11.5px; line-height:1.7; padding:10px 14px;
  margin-top:18px; max-height:160px; overflow-y:auto; }
#term .t { color:var(--faint); margin-right:8px; }
footer { color:var(--faint); font-size:11.5px; margin-top:12px; }
.modes { display:flex; border:1px solid var(--color-border-emphasized);
  border-radius:6px; overflow:hidden; }
.modes button { border:0; border-radius:0; padding:4px 13px; color:var(--ink2); }
.modes button[aria-pressed=true] { background:var(--ink); color:var(--plane);
  font-weight:var(--font-weight-semibold); }
#livebar { display:none; flex-direction:column; gap:8px;
  padding:12px 0; margin:6px 0 10px; border-bottom:1px solid var(--hairline); }
#livebar textarea { font:12px/1.7 var(--mono); color:var(--ink);
  background:var(--surface); border:1px solid var(--hairline); border-radius:6px;
  padding:8px 10px; width:100%; min-height:74px; resize:vertical; }
#livebar .row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
#livebar input[type=text], #livebar input[type=password] { font:inherit;
  font-size:12.5px; color:var(--ink); background:var(--surface);
  border:1px solid var(--hairline); border-radius:6px; padding:6px 10px; }
#question { flex:1; min-width:240px; }
#token { width:150px; font-family:var(--mono); font-size:11.5px; }
#ingest-btn, #ask-btn { background:var(--ink); color:var(--plane);
  border-color:var(--ink); font-weight:var(--font-weight-semibold); }
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
<header>
{mark}<span class="word">agora</span><span class="view">/ society view</span>
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
