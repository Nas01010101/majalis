"""Chinese pages: generated, translated, and data-safe."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_translate_zh_generates_pages():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "translate_zh.py")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    zh_index = (ROOT / "dashboard" / "index.zh.html").read_text()
    zh_live = (ROOT / "dashboard" / "live.zh.html").read_text()

    assert 'lang="zh-CN"' in zh_index and 'lang="zh-CN"' in zh_live
    for needle in ("世界模型", "每问成本", "信念板", "诚实性说明"):
        assert needle in zh_index, needle
    for needle in ("社会实况", "质疑者", "裁决者", "'辩论'"):
        assert needle in zh_live, needle
    # Code identifiers survive untranslated (the chart keys series by them).
    assert "agora-wm" in zh_index and "wrong_now" in zh_index
    # Each page links back to its English twin.
    assert 'href="/"' in zh_index and 'href="/live"' in zh_live


def test_replay_json_survives_translation():
    pat = re.compile(r"const R = (\{.*?\});\nconst MODELS", re.S)
    en = pat.search((ROOT / "dashboard" / "live.html").read_text())
    zh = pat.search((ROOT / "dashboard" / "live.zh.html").read_text())
    assert en and zh
    assert json.loads(en.group(1)) == json.loads(zh.group(1))
