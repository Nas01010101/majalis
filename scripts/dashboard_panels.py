"""SVG panel builders for the Agora dashboard (imported by build_dashboard.py).

Mark specs follow the dataviz contract: 2px lines with round caps, >=8px
markers with a 2px surface ring, bars <=24px with 4px rounded data-ends and a
2px surface gap between neighbors, hairline solid grid, text in ink tokens
(never the series color) with identity carried by a colored key beside it.
"""
from __future__ import annotations

import html
import json


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def scaling_chart(series: dict[str, list[tuple[int, float]]],
                  colors: dict[str, str]) -> str:
    """$/question vs stream length. Crosshair + all-series tooltip via JS
    (the svg carries its data in a JSON island)."""
    w, h, pad = 680, 300, 52
    xs = sorted({s for pts in series.values() for s, _ in pts})
    ymax = max(c for pts in series.values() for _, c in pts) * 1.2
    x0, x1 = xs[0], xs[-1]

    def X(s): return pad + (w - 2 * pad) * (s - x0) / max(1, x1 - x0)
    def Y(c): return h - pad - (h - 2 * pad) * c / ymax

    grid, yticks = "", [0, 0.005, 0.010, 0.015]
    for v in yticks:
        if v > ymax:
            continue
        grid += (f'<line x1="{pad}" y1="{Y(v):.1f}" x2="{w-pad}" y2="{Y(v):.1f}" '
                 f'class="grid"/>'
                 f'<text x="{pad-8}" y="{Y(v):.1f}" text-anchor="end" dy="4" '
                 f'class="tick">${v:.3f}</text>')
    marks = ""
    for arm, pts in series.items():
        col = colors[arm]
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(s):.1f},{Y(c):.1f}"
                     for i, (s, c) in enumerate(pts))
        marks += (f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" '
                  f'stroke-linecap="round" stroke-linejoin="round"/>')
        for s, c in pts:
            # 2px surface ring on every marker (legible across crossings)
            marks += (f'<circle cx="{X(s):.1f}" cy="{Y(c):.1f}" r="4" fill="{col}" '
                      f'stroke="var(--surface)" stroke-width="2"/>')
        # Direct end label: ink text keyed by a colored dot, never colored text.
        ex, ey = X(pts[-1][0]), Y(pts[-1][1])
        marks += (f'<circle cx="{ex+10:.1f}" cy="{ey-1:.1f}" r="4" fill="{col}"/>'
                  f'<text x="{ex+18:.1f}" y="{ey:.1f}" dy="3" class="endlabel">'
                  f'{_esc(arm)}</text>')
    xticks = "".join(f'<text x="{X(s):.1f}" y="{h-pad+18}" text-anchor="middle" '
                     f'class="tick">{s}</text>' for s in xs)
    payload = _esc(json.dumps({
        "xs": xs, "pad": pad, "w": w, "h": h, "ymax": ymax,
        "series": {a: {str(s): c for s, c in pts} for a, pts in series.items()},
        "colors": colors}))
    return (
        f'<figure class="chart" data-chart="line" data-payload="{payload}">'
        f'<svg viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="Cost per question vs stream length, by arm">'
        f'{grid}'
        f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" class="axis"/>'
        f'{marks}{xticks}'
        f'<text x="{w/2:.0f}" y="{h-10}" text-anchor="middle" class="tick">'
        f'stream length (evidence steps)</text>'
        f'<line class="xhair" x1="0" x2="0" y1="{pad-10}" y2="{h-pad}" '
        f'style="display:none"/>'
        f'</svg></figure>')


def auroc_bars(groups: list[dict], colors: dict[str, str]) -> str:
    """Grouped columns: learned vs the hand-set baseline it replaced, per
    predictive target. Chance line at 0.5. Values on the cap (bars are the
    no-hover channel), tooltip adds the definition."""
    w, h, pad = 680, 300, 52
    bw, gap = 22, 2  # bar width; 2px surface gap between the pair
    def Y(v): return h - pad - (h - 2 * pad) * v
    marks, xlabels = "", ""
    n = len(groups)
    for gi, g in enumerate(groups):
        cx = pad + (w - 2 * pad) * (gi + 0.5) / n
        for bi, (name, v) in enumerate([("learned", g["learned"]),
                                        (g["baseline_name"], g["baseline"])]):
            col = colors["learned"] if bi == 0 else colors["baseline"]
            x = cx - bw - gap / 2 + bi * (bw + gap)
            y, y0 = Y(v), Y(0)
            tip = f"{g['label']} — {name}: AUROC {v:.3f} ({g['note']})"
            # 4px rounded data-end, square at the baseline.
            marks += (
                f'<path d="M{x:.1f},{y0:.1f} V{y+4:.1f} Q{x:.1f},{y:.1f} '
                f'{x+4:.1f},{y:.1f} H{x+bw-4:.1f} Q{x+bw:.1f},{y:.1f} '
                f'{x+bw:.1f},{y+4:.1f} V{y0:.1f} Z" fill="{col}" class="bar" '
                f'data-tip="{_esc(tip)}" tabindex="0"/>')
            if bi == 0:
                # Selective labeling: cap value on the learned bar only (3dp —
                # "1.00" would overstate 0.999); the baseline's exact value
                # lives in the tooltip and the table view.
                marks += (f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" '
                          f'text-anchor="middle" class="cap">{v:.3f}</text>')
        xlabels += (f'<text x="{cx:.1f}" y="{h-pad+18}" text-anchor="middle" '
                    f'class="tick">{_esc(g["label"])}</text>')
    grid = "".join(
        f'<line x1="{pad}" y1="{Y(v):.1f}" x2="{w-pad}" y2="{Y(v):.1f}" class="grid"/>'
        f'<text x="{pad-8}" y="{Y(v):.1f}" text-anchor="end" dy="4" class="tick">'
        f'{v:.1f}</text>' for v in (0.0, 0.5, 1.0))
    chance = (f'<line x1="{pad}" y1="{Y(0.5):.1f}" x2="{w-pad}" y2="{Y(0.5):.1f}" '
              f'class="ref"/><text x="{w-pad+4}" y="{Y(0.5):.1f}" dy="4" '
              f'class="tick">chance</text>')
    return (
        f'<figure class="chart"><svg viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="AUROC, learned heads vs hand-set baselines">'
        f'{grid}{chance}'
        f'<line x1="{pad}" y1="{Y(0):.1f}" x2="{w-pad}" y2="{Y(0):.1f}" class="axis"/>'
        f'{marks}{xlabels}</svg></figure>')


def reliability_panel(curves: list[dict], color: str) -> str:
    """Small-multiple reliability squares: mean predicted vs empirical per
    bin, diagonal = perfect calibration. Per-dot tooltip carries bin + n."""
    panels = ""
    w = h = 260
    pad = 44
    def P(v): return pad + (w - 2 * pad) * v
    for c in curves:
        dots, path = "", ""
        pts = [(b["mean_pred"], b["empirical"], b["n"], b["bin"])
               for b in c["bins"]]
        for i, (p, e, nn, label) in enumerate(pts):
            x, y = P(p), h - P(e)
            path += f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
            tip = (f"{c['title']} bin {label}: predicted {p:.2f} vs observed "
                   f"{e:.2f} (n={nn})")
            dots += (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" '
                     f'stroke="var(--surface)" stroke-width="2" class="dot" '
                     f'data-tip="{_esc(tip)}" tabindex="0"/>')
        ticks = "".join(
            f'<text x="{P(v):.1f}" y="{h-pad+16}" text-anchor="middle" '
            f'class="tick">{v:.1f}</text>'
            f'<text x="{pad-8}" y="{h-P(v):.1f}" dy="3" text-anchor="end" '
            f'class="tick">{v:.1f}</text>' for v in (0.0, 0.5, 1.0))
        panels += (
            f'<svg viewBox="0 0 {w} {h}" role="img" class="reliability" '
            f'aria-label="Reliability of {_esc(c["title"])}">'
            f'<line x1="{P(0)}" y1="{h-P(0)}" x2="{P(1)}" y2="{h-P(1)}" class="ref"/>'
            f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" class="axis"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" class="axis"/>'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linecap="round"/>{dots}{ticks}'
            f'<text x="{w/2}" y="{h-6}" text-anchor="middle" class="tick">'
            f'predicted P(wrong)</text>'
            f'<text x="12" y="{h/2}" class="tick" transform="rotate(-90 12 {h/2})" '
            f'text-anchor="middle">observed</text>'
            f'<text x="{w/2}" y="16" text-anchor="middle" class="subtitle">'
            f'{_esc(c["title"])}</text></svg>')
    return f'<figure class="chart row">{panels}</figure>'
