"""Flatten the vendored Astryx neutral theme into web/astryx-tokens.css.

    python scripts/extract_astryx_tokens.py

Source: web/vendor/astryxdesign-theme-neutral-*/dist/theme.css (facebook/astryx,
MIT). web/vendor/ is gitignored; refetch with
    cd web/vendor && npm pack @astryxdesign/theme-neutral \
      && tar xzf astryxdesign-theme-neutral-*.tgz -C <dir> --strip-components=1
The published theme is @scope-wrapped for React apps; the static
dashboard only needs the design tokens, so this pulls the custom properties
out of the astryx-theme layer into a plain :root block (light-dark() values
kept — one declaration serves both themes).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = sorted((ROOT / "web" / "vendor").glob(
    "astryxdesign-theme-neutral-*/dist/theme.css"))[-1]
OUT = ROOT / "web" / "astryx-tokens.css"

KEEP = ("--color-", "--radius-", "--font-", "--text-", "--shadow-")
DROP = ("--color-syntax", "--color-chart")


def main() -> None:
    css = SRC.read_text()
    layer = re.search(r"@layer astryx-theme \{(.*)", css, re.S).group(1)
    scope = re.search(r":scope \{(.*?)\n  \}", layer, re.S).group(1)
    props = re.findall(r"(--[a-z0-9-]+):\s*([^;]+);", scope)
    lines = [
        "/* Astryx neutral theme tokens — extracted from "
        f"@astryxdesign/theme-neutral ({SRC.parent.parent.name})",
        " * (facebook/astryx, MIT). Flattened @scope -> :root for the static dashboard.",
        " * Regenerate: python scripts/extract_astryx_tokens.py */",
        ":root { color-scheme: light dark;",
    ]
    seen: set[str] = set()
    for p, v in props:
        if p.startswith(KEEP) and not p.startswith(DROP) and p not in seen:
            seen.add(p)
            lines.append(f"  {p}: {v};")
    lines.append("}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(seen)} tokens)")


if __name__ == "__main__":
    main()
