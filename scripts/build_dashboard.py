"""Render results/ into a self-contained dashboard (dashboard/index.html).

    python scripts/build_dashboard.py

No external assets; light/dark via prefers-color-scheme; data inlined at
build time so the file is portable (demo video, AliCloud static hosting).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "dashboard" / "index.html"

CSS = """
:root { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e5e4e0;
  --s1:#2a78d6; --s2:#eb6834; --good:#008300; --warn:#eda100; }
@media (prefers-color-scheme: dark) {
  :root { --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --grid:#3a3936;
    --s1:#3987e5; --s2:#d95926; --good:#00a300; --warn:#c98500; } }
* { box-sizing:border-box; }
body { background:var(--surface); color:var(--ink); margin:0 auto; padding:24px;
  max-width:1080px; font:15px/1.5 -apple-system,'Segoe UI',sans-serif; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 8px; }
.sub { color:var(--ink2); margin:0 0 20px; }
.tiles { display:flex; gap:12px; flex-wrap:wrap; }
.tile { border:1px solid var(--grid); border-radius:10px; padding:14px 18px;
  min-width:170px; flex:1; }
.tile .v { font-size:26px; font-weight:650; } .tile .k { color:var(--ink2); font-size:13px; }
svg text { fill:var(--ink2); font-size:11px; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th,td { text-align:left; padding:5px 10px; border-bottom:1px solid var(--grid); }
th { color:var(--ink2); font-weight:550; }
.fired { color:var(--warn); font-weight:600; } .ok { color:var(--good); }
.legend { display:flex; gap:16px; font-size:13px; color:var(--ink2); margin:4px 0; }
.dot { display:inline-block; width:10px; height:10px; border-radius:5px;
  margin-right:5px; vertical-align:-1px; }
.wrap { overflow-x:auto; }
"""


def load_sessions() -> list[dict]:
    p = RESULTS / "session_summary.json"
    return json.loads(p.read_text()) if p.exists() else []


def scaling_chart(sessions: list[dict]) -> str:
    """$/question vs stream length — single grows linearly, agora stays flat."""
    cells: dict[tuple[str, int], list[dict]] = {}
    for s in sessions:
        if s["arm"] in ("single", "agora"):
            cells.setdefault((s["arm"], s.get("steps", 8)), []).append(s)
    series = {}
    for (arm, steps), rows in sorted(cells.items(), key=lambda kv: kv[0][1]):
        cost_q = sum(r["cost_usd"] for r in rows) / max(1, sum(r["n"] for r in rows))
        series.setdefault(arm, []).append((steps, cost_q))
    if not series or not any(len(v) > 1 for v in series.values()):
        return "<p class='sub'>(scaling data pending)</p>"

    w, h, pad = 640, 260, 44
    xs = sorted({s for pts in series.values() for s, _ in pts})
    ymax = max(c for pts in series.values() for _, c in pts) * 1.15
    def X(s): return pad + (w - 2 * pad) * (s - xs[0]) / max(1, xs[-1] - xs[0])
    def Y(c): return h - pad - (h - 2 * pad) * c / ymax
    colors = {"agora": "var(--s1)", "single": "var(--s2)"}
    grid = "".join(
        f'<line x1="{pad}" y1="{Y(ymax*f):.0f}" x2="{w-pad}" y2="{Y(ymax*f):.0f}" '
        f'stroke="var(--grid)" stroke-width="1"/>'
        f'<text x="{pad-6}" y="{Y(ymax*f):.0f}" text-anchor="end" dy="4">'
        f'${ymax*f:.3f}</text>' for f in (0, 0.5, 1))
    marks = ""
    for arm, pts in series.items():
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(s):.0f},{Y(c):.0f}"
                     for i, (s, c) in enumerate(pts))
        marks += f'<path d="{d}" fill="none" stroke="{colors[arm]}" stroke-width="2"/>'
        for s, c in pts:
            marks += (f'<circle cx="{X(s):.0f}" cy="{Y(c):.0f}" r="4" '
                      f'fill="{colors[arm]}"><title>{arm} · {s} steps · '
                      f'${c:.4f}/question</title></circle>')
        lx, lc = pts[-1]
        marks += (f'<text x="{X(lx)+8:.0f}" y="{Y(lc):.0f}" dy="4" '
                  f'style="fill:{colors[arm]};font-weight:600">{arm}</text>')
    xticks = "".join(
        f'<text x="{X(s):.0f}" y="{h-pad+16}" text-anchor="middle">{s}</text>'
        for s in xs)
    return (f'<svg viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="Cost per question vs stream length">{grid}{marks}{xticks}'
            f'<text x="{w/2:.0f}" y="{h-8}" text-anchor="middle">stream length '
            f'(steps)</text></svg>')


def gate_table(seed: int = 0, steps: int = 8) -> str:
    p = RESULTS / "raw" / f"session_agora_s{seed}_t{steps}.jsonl"
    if not p.exists():
        return "<p class='sub'>(no session trace yet)</p>"
    rows = []
    for line in p.read_text().splitlines():
        r = json.loads(line)
        g = r.get("gate") or {}
        fired = g.get("fired")
        rows.append(
            f"<tr><td>{html.escape(r['task_id'])}</td>"
            f"<td class='{'fired' if fired else 'ok'}'>"
            f"{'DEBATE' if fired else 'commit'}</td>"
            f"<td>{g.get('p_wrong', '—')}</td>"
            f"<td>{g.get('max_doubt', '—')}</td>"
            f"<td>{g.get('disagreement', '—')}</td>"
            f"<td>{'✓' if r['correct'] else '✗'}</td>"
            f"<td>{r['total_tokens']}</td></tr>")
    return ("<div class='wrap'><table><tr><th>question</th><th>gate</th>"
            "<th>p(wrong)</th><th>max doubt</th><th>disagreement</th>"
            "<th>correct</th><th>tokens</th></tr>" + "".join(rows)
            + "</table></div>")


def tiles(sessions: list[dict]) -> str:
    ag = [s for s in sessions if s["arm"] == "agora"]
    si = [s for s in sessions if s["arm"] == "single"]
    if not ag or not si:
        return ""
    def acc(rows): return sum(r["correct"] for r in rows), sum(r["n"] for r in rows)
    def cpq(rows): return sum(r["cost_usd"] for r in rows) / max(1, sum(r["n"] for r in rows))
    a_c, a_n = acc(ag)
    s_c, s_n = acc(si)
    big_ag = [s for s in ag if s.get("steps", 8) >= 32] or ag
    big_si = [s for s in si if s.get("steps", 8) >= 32] or si
    ratio = cpq(big_si) / max(1e-9, cpq(big_ag))
    return (
        f"<div class='tiles'>"
        f"<div class='tile'><div class='v'>{a_c}/{a_n}</div>"
        f"<div class='k'>Agora accuracy (all sessions)</div></div>"
        f"<div class='tile'><div class='v'>{s_c}/{s_n}</div>"
        f"<div class='k'>Single-agent accuracy</div></div>"
        f"<div class='tile'><div class='v'>{ratio:.1f}×</div>"
        f"<div class='k'>Single-agent cost per question at the longest "
        f"stream, relative to Agora</div></div></div>")


def arms_table(sessions: list[dict]) -> str:
    from collections import defaultdict
    cells: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "c": 0, "cost": 0.0})
    for s in sessions:
        a = cells[(s["arm"], s.get("steps", 8))]
        a["n"] += s["n"]
        a["c"] += s["correct"]
        a["cost"] += s["cost_usd"]
    rows = ""
    for (arm, steps), a in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rows += (f"<tr><td>{html.escape(arm)}</td><td>{steps}</td>"
                 f"<td>{a['c']}/{a['n']}</td>"
                 f"<td>${a['cost'] / max(1, a['n']):.5f}</td></tr>")
    return ("<div class='wrap'><table><tr><th>arm</th><th>stream steps</th>"
            "<th>accuracy</th><th>cost / question</th></tr>"
            + rows + "</table></div>")


def main() -> None:
    sessions = load_sessions()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agora — world-model-modulated debate</title><style>{CSS}</style></head><body>
<h1>Agora</h1>
<p class="sub">A shared world model decides when debate is worth the tokens.
Baselines re-read the evidence stream per question; Agora ingests once into a
belief board and debates only doubted keys.</p>
{tiles(sessions)}
<h2>Cost per question vs stream length</h2>
<div class="legend"><span><span class="dot" style="background:var(--s1)"></span>
agora (flat — perception amortized)</span>
<span><span class="dot" style="background:var(--s2)"></span>
single agent (linear — re-reads the stream)</span></div>
{scaling_chart(sessions)}
<h2>All arms (pooled across seeds)</h2>
{arms_table(sessions)}
<h2>Gate decisions (session seed 0)</h2>
{gate_table()}
<p class="sub">Generated by scripts/build_dashboard.py from results/.</p>
</body></html>""")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
