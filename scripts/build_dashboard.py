"""Render results/ + data/ into a self-contained dashboard (dashboard/index.html).

    python scripts/build_dashboard.py

Design per the dataviz contract: palette slots validated with the six-checks
script on both surfaces (light WARNs on aqua/yellow contrast — relief rule
satisfied by direct end labels + the table views), light/dark both SELECTED
(not auto-flipped), hover layer on every chart, every chart twinned by a
table, text in ink tokens with identity carried by colored keys.
"""
from __future__ import annotations

import html
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dashboard_panels import auroc_bars, reliability_panel, scaling_chart  # noqa: E402

RESULTS, DATA = ROOT / "results", ROOT / "data"
OUT = ROOT / "dashboard" / "index.html"

# Categorical slots in fixed order (validated); color follows the entity.
LIGHT = {"agora-wm": "#2a78d6", "agora": "#1baf7a", "single": "#eda100",
         "learned": "#2a78d6", "baseline": "#898781"}
DARK = {"agora-wm": "#3987e5", "agora": "#199e70", "single": "#c98500",
        "learned": "#3987e5", "baseline": "#898781"}

CSS = """
:root { --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --good:#0ca30c;
  --warn:#fab219; --crit:#d03b3b; --border:rgba(11,11,11,.10);
  --wm:#2a78d6; --ag:#1baf7a; --si:#eda100; --base:#898781; }
@media (prefers-color-scheme: dark) {
  :root { --surface:#1a1a19; --plane:#0d0d0d; --ink:#ffffff; --ink2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835; --good:#0ca30c;
    --warn:#fab219; --crit:#d03b3b; --border:rgba(255,255,255,.10);
    --wm:#3987e5; --ag:#199e70; --si:#c98500; } }
* { box-sizing:border-box; }
body { background:var(--plane); color:var(--ink); margin:0 auto;
  padding:28px 24px 40px; max-width:1120px;
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:32px 0 8px; }
.sub { color:var(--ink2); margin:0 0 8px; max-width:72ch; }
.card { background:var(--surface); border:1px solid var(--border);
  border-radius:12px; padding:16px 18px; margin:10px 0; }
.tiles { display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }
.tile { background:var(--surface); border:1px solid var(--border);
  border-radius:12px; padding:14px 18px; min-width:180px; flex:1; }
.tile .v { font-size:28px; font-weight:650; }
.tile .k { color:var(--ink2); font-size:13px; margin-top:2px; }
.tile .d { font-size:12px; margin-top:4px; color:var(--ink2); }
.tile .d.up { color:var(--good); }
.legend { display:flex; gap:18px; font-size:13px; color:var(--ink2);
  margin:2px 0 6px; flex-wrap:wrap; }
.key { display:inline-block; width:14px; height:3px; border-radius:2px;
  vertical-align:3px; margin-right:6px; }
.swatch { display:inline-block; width:10px; height:10px; border-radius:2px;
  vertical-align:-1px; margin-right:6px; }
figure.chart { margin:0; } figure.chart.row { display:flex; gap:16px; flex-wrap:wrap; }
svg { max-width:100%; height:auto; }
svg.reliability { width:280px; flex:0 0 auto; }
svg .grid { stroke:var(--grid); stroke-width:1; }
svg .axis { stroke:var(--axis); stroke-width:1; }
svg .ref { stroke:var(--muted); stroke-width:1; }
svg .xhair { stroke:var(--muted); stroke-width:1; }
svg text { fill:var(--ink2); font:11px system-ui,sans-serif; }
svg .tick { fill:var(--muted); font-variant-numeric:tabular-nums; }
svg .cap { fill:var(--ink2); font-weight:600; }
svg .endlabel { fill:var(--ink2); font-weight:600; font-size:12px; }
svg .subtitle { fill:var(--ink2); font-weight:600; font-size:12px; }
svg .bar, svg .dot { cursor:default; } svg .bar:hover, svg .bar:focus,
svg .dot:hover, svg .dot:focus { opacity:.85; outline:none; }
.wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th,td { text-align:left; padding:5px 10px; border-bottom:1px solid var(--grid);
  font-variant-numeric:tabular-nums; }
th { color:var(--ink2); font-weight:550; }
.chip { font-size:12px; font-weight:650; padding:1px 8px; border-radius:9px;
  border:1px solid var(--border); white-space:nowrap; }
.chip.fire { color:var(--ink); background:color-mix(in srgb, var(--warn) 22%, var(--surface)); }
.chip.ok { color:var(--ink); background:color-mix(in srgb, var(--good) 16%, var(--surface)); }
.good { color:var(--good); } .crit { color:var(--crit); }
details { margin:8px 0; } summary { color:var(--ink2); cursor:pointer; font-size:13px; }
#tip { position:fixed; display:none; background:var(--surface); color:var(--ink);
  border:1px solid var(--border); border-radius:8px; padding:8px 10px;
  font-size:12.5px; pointer-events:none; box-shadow:0 4px 14px rgba(0,0,0,.15);
  max-width:320px; z-index:9; }
#tip .val { font-weight:650; } #tip .lbl { color:var(--ink2); }
footer { color:var(--muted); font-size:12.5px; margin-top:28px; max-width:80ch; }
footer li { margin:3px 0; }
"""

JS = """
const tip = document.getElementById('tip');
function showTip(evt, rows) {
  tip.replaceChildren(...rows.map(([k, v, col]) => {
    const d = document.createElement('div');
    if (col) { const key = document.createElement('span');
      key.className = 'key'; key.style.background = col; d.appendChild(key); }
    const val = document.createElement('span'); val.className = 'val';
    val.textContent = v;                       // untrusted-data rule: textContent
    const lbl = document.createElement('span'); lbl.className = 'lbl';
    lbl.textContent = ' ' + k;
    d.append(val, lbl); return d;
  }));
  tip.style.display = 'block';
  const pad = 14, vw = innerWidth, r = tip.getBoundingClientRect();
  tip.style.left = Math.min(evt.clientX + pad, vw - r.width - 8) + 'px';
  tip.style.top = (evt.clientY + pad) + 'px';
}
function hideTip() { tip.style.display = 'none'; }
// Per-mark tooltips (bars, reliability dots)
for (const el of document.querySelectorAll('[data-tip]')) {
  const show = e => showTip(e, [['', el.dataset.tip, null]]);
  el.addEventListener('pointermove', show);
  el.addEventListener('focus', e => showTip({clientX: 40, clientY: 40}, [['', el.dataset.tip, null]]));
  el.addEventListener('pointerleave', hideTip);
  el.addEventListener('blur', hideTip);
}
// Line chart: crosshair snaps to nearest step; one tooltip, every series.
for (const fig of document.querySelectorAll('[data-chart="line"]')) {
  const cfg = JSON.parse(fig.dataset.payload);
  const svg = fig.querySelector('svg'), hair = svg.querySelector('.xhair');
  const X = s => cfg.pad + (cfg.w - 2 * cfg.pad) * (s - cfg.xs[0]) /
                 Math.max(1, cfg.xs[cfg.xs.length - 1] - cfg.xs[0]);
  svg.addEventListener('pointermove', e => {
    const pt = new DOMPoint(e.clientX, e.clientY)
      .matrixTransform(svg.getScreenCTM().inverse());
    const step = cfg.xs.reduce((a, b) =>
      Math.abs(X(b) - pt.x) < Math.abs(X(a) - pt.x) ? b : a);
    hair.setAttribute('x1', X(step)); hair.setAttribute('x2', X(step));
    hair.style.display = '';
    const rows = [['', step + '-step stream', null]];
    for (const [arm, pts] of Object.entries(cfg.series)) {
      const c = pts[String(step)];
      if (c !== undefined)
        rows.push([arm, '$' + c.toFixed(4) + '/q', cfg.colors[arm]]);
    }
    showTip(e, rows);
  });
  svg.addEventListener('pointerleave', () => { hair.style.display = 'none'; hideTip(); });
}
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def pooled(sessions: list[dict]) -> dict[tuple, dict]:
    cells: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "c": 0, "cost": 0.0})
    for s in sessions:
        a = cells[(s["arm"], s.get("steps", 8))]
        a["n"] += s["n"]; a["c"] += s["correct"]; a["cost"] += s["cost_usd"]
    return cells


def tiles(cells, gate_eval, wm_metrics) -> str:
    wm_rows = [(k, a) for k, a in cells.items() if k[0] == "agora-wm"]
    wm_c = sum(a["c"] for _, a in wm_rows); wm_n = sum(a["n"] for _, a in wm_rows)
    big_si = cells.get(("single", 32)); big_ag = cells.get(("agora", 32))
    ratio = ((big_si["cost"] / big_si["n"]) / (big_ag["cost"] / big_ag["n"])
             if big_si and big_ag else None)
    gq = gate_eval["gate_quality"]; heur = gate_eval["learned_vs_heuristic"]["heuristic"]
    d_recall = (gq["recall_poisoned"] - heur["recall_poisoned"]) * 100
    t = "<div class='tiles'>"
    t += (f"<div class='tile'><div class='v'>{wm_c}/{wm_n}</div>"
          f"<div class='k'>learned-gate accuracy, live session eval (agora-wm)</div></div>")
    t += (f"<div class='tile'><div class='v'>0</div>"
          f"<div class='k'>LLM calls per gate decision</div>"
          f"<div class='d'>the stacker learned the sampler adds nothing</div></div>")
    if ratio:
        t += (f"<div class='tile'><div class='v'>{ratio:.1f}×</div>"
              f"<div class='k'>single-agent cost per question at 32-step "
              f"streams, vs Agora</div></div>")
    t += (f"<div class='tile'><div class='v'>{gq['recall_poisoned']:.0%}</div>"
          f"<div class='k'>corrupted boards caught, {gate_eval['n_questions']} "
          f"held-out questions</div>"
          f"<div class='d up'>+{d_recall:.0f}pp vs the hand-set gate, at "
          f"{gq['false_fire_rate']:.1%} false-fire</div></div>")
    return t + "</div>"


def arms_table(cells) -> str:
    rows = ""
    for (arm, steps), a in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rows += (f"<tr><td>{esc(arm)}</td><td>{steps}</td><td>{a['c']}/{a['n']}</td>"
                 f"<td>${a['cost'] / max(1, a['n']):.5f}</td></tr>")
    return ("<div class='wrap'><table><tr><th>arm</th><th>stream steps</th>"
            f"<th>accuracy</th><th>cost / question</th></tr>{rows}</table></div>")


def gate_table(path: Path) -> str:
    if not path.exists():
        return "<p class='sub'>(no learned-arm trace yet)</p>"
    rows = ""
    for line in path.read_text().splitlines():
        r = json.loads(line)
        g = r.get("gate") or {}
        fired = g.get("fired")
        chip = ("<span class='chip fire'>▲ DEBATE</span>" if fired
                else "<span class='chip ok'>✓ commit</span>")
        okmark = ("<span class='good'>✓</span>" if r["correct"]
                  else "<span class='crit'>✗</span>")
        rows += (f"<tr><td>{esc(r['task_id'])}</td><td>{chip}</td>"
                 f"<td>{g.get('p_wrong', '—')}</td><td>{esc(g.get('reason', ''))}</td>"
                 f"<td>{okmark}</td><td>{r['total_tokens']}</td></tr>")
    return ("<div class='wrap'><table><tr><th>question</th><th>gate</th>"
            "<th>p(wrong)</th><th>reason</th><th>correct</th><th>tokens</th>"
            f"</tr>{rows}</table></div>")


def auroc_table(groups) -> str:
    rows = "".join(
        f"<tr><td>{esc(g['label'])}</td><td>{g['learned']:.3f}</td>"
        f"<td>{g['baseline']:.3f}</td><td>{esc(g['baseline_name'])}</td>"
        f"<td>{esc(g['note'])}</td></tr>" for g in groups)
    return ("<details><summary>table view — exact AUROC values</summary>"
            "<div class='wrap'><table><tr><th>target</th><th>learned</th>"
            f"<th>baseline</th><th>baseline model</th><th>data</th></tr>{rows}"
            "</table></div></details>")


def reliability_table(rel) -> str:
    rows = "".join(
        f"<tr><td>{esc(b['bin'])}</td><td>{b['n']}</td>"
        f"<td>{b['mean_pred']:.3f}</td><td>{b['empirical']:.3f}</td></tr>"
        for b in rel["wrong_now_head"])
    return (f"<details><summary>table view — wrong_now reliability bins "
            f"({rel['n_step_rows']} held-out rows)</summary>"
            "<div class='wrap'><table><tr><th>score bin</th><th>n</th>"
            f"<th>mean predicted</th><th>observed wrong</th></tr>{rows}"
            "</table></div></details>")


def main() -> None:
    sessions = load(RESULTS / "session_summary.json") or []
    gate_eval = load(RESULTS / "offline_gate_eval.json")
    wm = load(DATA / "wm_weights.json")
    cells = pooled(sessions)

    series: dict[str, list[tuple[int, float]]] = {}
    for arm in ("agora-wm", "agora", "single"):  # fixed slot order
        pts = sorted((steps, a["cost"] / a["n"]) for (nm, steps), a in cells.items()
                     if nm == arm)
        if len(pts) > 1:
            series[arm] = pts

    au = wm["metrics"]["auroc"]
    lvh = gate_eval["learned_vs_heuristic"]
    groups = [
        {"label": "wrong_now (board value incorrect)", "note": "11.7k held-out rows",
         "learned": au["wrong_now/learned"], "baseline": au["wrong_now/hand_doubt"],
         "baseline_name": "hand-set doubt blend"},
        {"label": "superseded_next (fact overturned)", "note": "11.7k held-out rows",
         "learned": au["superseded/learned"], "baseline": au["superseded/hand_lomax"],
         "baseline_name": "fixed Lomax prior"},
        {"label": "commit_risk (gate score)", "note": f"{gate_eval['n_questions']} held-out questions",
         "learned": lvh["learned"]["score_auroc"],
         "baseline": lvh["heuristic"]["score_auroc"],
         "baseline_name": "hand-set risk blend"},
    ]
    curves = [
        {"title": "wrong_now head", "bins": gate_eval["reliability"]["wrong_now_head"]},
        {"title": "commit_risk (gate)", "bins": gate_eval["reliability"]["commit_risk"]},
    ]
    gq, heur = lvh["learned"], lvh["heuristic"]
    today = date.today().isoformat()

    def legend_line(items):
        return ("<div class='legend'>" + "".join(
            f"<span><span class='key' style='background:var(--{v})'></span>"
            f"{esc(k)}</span>" for k, v in items) + "</div>")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agora — a debate society steered by a learned world model</title>
<style>{CSS}</style></head><body>
<h1>Agora</h1>
<p class="sub">A multi-agent debate society whose <strong>learned world
model</strong> — trained on the society's own logged episodes — decides when
disagreement is worth the tokens. Baselines re-read the evidence stream per
question; Agora ingests once into a shared belief board, and two trained heads
(<code>wrong_now</code>, <code>superseded_next</code>) plus a conformal
threshold route debate only where the board is likely corrupted.</p>
{tiles(cells, gate_eval, wm["metrics"])}

<div class="card"><h2 style="margin-top:0">Cost per question vs stream length</h2>
<p class="sub">Perception is amortized into the board, so Agora's cost stays
flat while the single agent re-reads a growing stream. (vanilla 3×3 debate:
$0.0709/q at 8 steps — off this chart's scale; see the table.)</p>
{legend_line([("agora-wm (learned gate)", "wm"), ("agora (hand-set gate)", "ag"), ("single agent", "si")])}
{scaling_chart(series, {"agora-wm": "var(--wm)", "agora": "var(--ag)", "single": "var(--si)"})}
<details><summary>table view — all arms, pooled across seeds</summary>{arms_table(cells)}</details></div>

<div class="card"><h2 style="margin-top:0">The learned world model vs the
hand-set heuristics it replaced</h2>
<p class="sub">Same features, same held-out data — the only change is that the
weights are trained instead of typed. The fixed Lomax survival prior sits at
chance; the learned dynamics head does not.</p>
{legend_line([("learned (trained on logged episodes)", "wm"), ("hand-set baseline", "base")])}
{auroc_bars(groups, {"learned": "var(--wm)", "baseline": "var(--base)"})}
{auroc_table(groups)}
<p class="sub">Gate decision quality on {gate_eval['n_questions']} questions from
100 unseen streams — learned: fires {gq['fire_rate']:.1%}, catches
{gq['recall_poisoned']:.1%} of corrupted boards, {gq['false_fire_rate']:.1%}
false-fire, accepted-error {gq['accepted_error_rate']:.1%} ≤ α={gq['alpha']}.
Hand-set: fires {heur['fire_rate']:.1%} (2× the debates) for
{heur['recall_poisoned']:.1%} recall at {heur['false_fire_rate']:.1%}
false-fire.</p></div>

<div class="card"><h2 style="margin-top:0">Calibration (held-out)</h2>
<p class="sub">Predicted P(wrong) vs observed frequency, 10 bins; the diagonal
is perfect calibration. The conformal ACCEPT threshold is calibrated on top of
these scores, so the guarantee never rests on the model being exactly
calibrated.</p>
{reliability_panel(curves, "var(--wm)")}
{reliability_table(gate_eval["reliability"])}</div>

<div class="card"><h2 style="margin-top:0">Gate decisions — live learned-gate
session (seed 0, 16 steps)</h2>
<p class="sub">Every question the deployed society answered, with the gate's
decision and reason. It debated exactly the weak-source displacements —
rumor-poisoned beliefs — and nothing else.</p>
{gate_table(RESULTS / "raw" / "session_agora-wm_s0_t16.jsonl")}</div>

<footer><strong>Honesty notes.</strong><ul>
<li>wrong_now's 0.999 AUROC is on synthetic validation streams where
weak-source displacement is highly separable; the honest generalization
numbers are 0.937 AUROC on real LLM-built boards and the gate-quality figures
above.</li>
<li>The offline benchmark stubs the disagreement sampler to 0 for both modes
(the learned stacker measured its weight at zero; the heuristic gets its
skip-path value).</li>
<li>agora-wm live cells measured at 8- and 16-step streams; 32-step pending.
Hand-set-gate and single-agent numbers are frozen from the pre-learned-WM
benchmark.</li>
<li>Generated {today} by scripts/build_dashboard.py from results/ + data/
(offline eval: seeds 5000–5099, disjoint from eval 0–99, calibration 100–999,
world-model training 1000–3199).</li></ul></footer>
<div id="tip" role="status"></div>
<script>{JS}</script>
</body></html>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page)//1024}KB)")


if __name__ == "__main__":
    main()
