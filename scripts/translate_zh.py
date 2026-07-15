"""Generate the Chinese pages (index.zh.html, live.zh.html) from the built
English ones — a deterministic, ordered string-substitution pass.

    python scripts/translate_zh.py   # run AFTER build_dashboard.py / build_live.py

Least-code i18n: the two pages are generated artifacts with a fixed string
set we control, so a longest-first replacement table beats an i18n
framework. Code identifiers (arm names, task ids, JSON keys, model names)
are deliberately NOT translated — they are data, and the line chart's JS
keys its series by arm name. A test asserts key zh strings land and the
replay JSON survives byte-identical.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "dashboard"

COMMON = [
    ('<html lang="en">', '<html lang="zh-CN">'),
    ('Skip to content', '跳转到正文'),
    ('<a href="/">Benchmarks</a>', '<a href="/zh">基准结果</a>'),
    ('API playground (try /ingest and /ask)', 'API 演练场（试试 /ingest 和 /ask）'),
    ('>API playground<', '>API 演练场<'),
]

INDEX = [
    ('<title>Majalis — a debate society steered by a learned world model</title>',
     '<title>Majalis — 由学习世界模型调控的辩论社会</title>'),
    ('content="Majalis — a multi-agent debate society whose learned world model '
     'decides when debate is worth the tokens."',
     'content="Majalis — 一个多智能体辩论社会：学习世界模型决定何时值得为辩论花费 token。"'),
    ('Qwen Cloud · Track 3 — agent\nsociety · learned world model',
     '<b>Qwen Cloud · 赛道 3</b> — 智能体社会 · 学习世界模型'),
    ('Your agents debate too much. The world model decides when it\'s worth it.',
     '你的智能体辩论太多了。世界模型来决定何时值得辩论。'),
    ('A society of Qwen agents shares one belief board; two heads\n<strong>trained on its own logged episodes</strong>',
     '一群 Qwen 智能体共享一块信念板；两个<strong>在社会自身运行日志上训练的</strong>预测头'),
    (') plus a conformal threshold route debate only\nwhere the board is likely corrupted — at <strong>zero LLM calls per gate\ndecision</strong>.',
     '）加上共形阈值，只在信念板可能被污染的地方触发辩论——<strong>每次门控决策零 LLM 调用</strong>。'),
    ('Society view — watch a live run', '社会实况 — 观看真实运行回放'),
    ('<a href="/healthz">health</a>', '<a href="/healthz">健康检查</a>'),
    ('<a href="/zh" hreflang="zh-CN" lang="zh-CN">中文</a>', '<a href="/" hreflang="en" lang="en">English</a>'),
    ('# try it — one command, no API key', '# 一条命令即可体验 — 无需 API key'),
    ('learned-gate accuracy, live session eval (majalis-wm)', '学习门控准确率（majalis-wm，线上会话评测）'),
    ('LLM calls per gate decision', '每次门控决策的 LLM 调用数'),
    ('the stacker learned the sampler adds nothing', '融合器学到采样器毫无贡献'),
    ('single-agent cost per question at 32-step streams, vs Majalis (learned gate)',
     '32 步流上单智能体每问成本（相对 Majalis 学习门控）'),
    ('corrupted boards caught, 1600 held-out questions', '被污染信念捕获率（1600 个留出问题）'),
    ('vs the hand-set gate, at 0.9% false-fire', '优于手工门控，误触发仅 0.9%'),
    ('Benchmark — session eval, live Qwen runs', '基准 — 会话评测 · 真实 Qwen 运行'),
    ('Cost per question vs stream length', '每问成本 vs 证据流长度'),
    ('Perception is amortized into the board, so Majalis\'s cost stays\nflat while the single agent re-reads a growing stream. (vanilla 3×3 debate:\n$0.0709/q at 8 steps — off this chart\'s scale; see the table.)',
     '感知成本摊销进信念板，因此 Majalis 的成本保持平坦，而单智能体每问都要重读不断增长的流。（朴素 3×3 辩论：8 步时 $0.0709/问——超出本图量程，见表格。）'),
    ('majalis-wm (learned gate)', 'majalis-wm（学习门控）'),
    ('majalis (hand-set gate)', 'majalis（手工门控）'),
    ('single agent</span>', '单智能体</span>'),
    ('table view — all arms, pooled across seeds', '表格视图 — 全部对比臂（跨种子汇总）'),
    ('World model — trained vs hand-set, held out', '世界模型 — 学习 vs 手工 · 留出数据'),
    ('The learned heads vs the heuristics they replaced', '学习头 vs 被它们替换的手工启发式'),
    ('Same features, same held-out data — the only change is that the\nweights are trained instead of typed. The fixed Lomax survival prior sits at\nchance; the learned dynamics head does not.',
     '相同特征、相同留出数据——唯一的区别是权重来自训练而非手写。固定 Lomax 生存先验停留在随机水平；学习动态头则不然。'),
    ('learned (trained on logged episodes)', '学习（在运行日志上训练）'),
    ('hand-set baseline', '手工基线'),
    ('table view — exact AUROC values', '表格视图 — 精确 AUROC 数值'),
    ('Gate decision quality on 1600 questions from\n100 unseen streams — learned: fires',
     '门控决策质量（100 条未见过的流、1600 个问题）— 学习门控：触发'),
    ('catches\n', '捕获 '),
    ('of corrupted boards,', '的被污染信念板，'),
    ('false-fire, accepted-error', '误触发，接受后错误率'),
    ('Hand-set: fires', '手工门控：触发'),
    ('(2× the debates) for', '（2 倍辩论量），捕获'),
    ('recall at', '，误触发'),
    ('false-fire.</p></div>', '。</p></div>'),
    ('World model — reliability, held out', '世界模型 — 可靠性 · 留出数据'),
    ('<h2>Calibration</h2>', '<h2>校准曲线</h2>'),
    ('Predicted P(wrong) vs observed frequency, 10 bins; the diagonal\nis perfect calibration. The conformal ACCEPT threshold is calibrated on top of\nthese scores, so the guarantee never rests on the model being exactly\ncalibrated.',
     '预测 P(wrong) 对比实际频率（10 个分箱）；对角线为完美校准。共形 ACCEPT 阈值在这些分数之上再做校准，因此保证不依赖模型本身完全校准。'),
    ('table view — wrong_now reliability bins', '表格视图 — wrong_now 可靠性分箱'),
    ('held-out rows)', '条留出样本)'),
    ('Live run — seed 0 · 16 steps · learned gate', '真实运行 — 种子 0 · 16 步 · 学习门控'),
    ('Gate decisions, question by question', '逐问门控决策'),
    ('Every question the deployed society answered, with the gate\'s\ndecision and reason. It debated exactly the weak-source displacements —\nrumor-poisoned beliefs — and nothing else.',
     '已部署社会回答的每个问题，附门控决策与理由。它只对弱信源顶替（谣言污染的信念）发起辩论，其余一律直接提交。'),
    ('Watch this run\nplay out in the society view →', '在社会实况页观看这次运行 →'),
    ('<th>question</th><th>gate</th>', '<th>问题</th><th>门控</th>'),
    ('<th>p(wrong)</th><th>reason</th><th>correct</th><th>tokens</th>',
     '<th>p(wrong)</th><th>理由</th><th>正确</th><th>tokens</th>'),
    ('<th>arm</th><th>stream steps</th>', '<th>对比臂</th><th>流步数</th>'),
    ('<th>accuracy</th><th>cost / question</th>', '<th>准确率</th><th>每问成本</th>'),
    ('<th>target</th><th>learned</th>', '<th>预测目标</th><th>学习</th>'),
    ('<th>baseline</th><th>baseline model</th><th>data</th>', '<th>基线</th><th>基线模型</th><th>数据</th>'),
    ('<th>score bin</th><th>n</th>', '<th>分数分箱</th><th>n</th>'),
    ('<th>mean predicted</th><th>observed wrong</th>', '<th>平均预测</th><th>实际错误率</th>'),
    ('>debate<', '>辩论<'),
    ('>commit<', '>提交<'),
    ('Honesty notes', '诚实性说明'),
    ('stream length\n(evidence steps)', '流长度（证据步数）'),
    ('stream length (evidence steps)', '流长度（证据步数）'),
    ('>chance</text>', '>随机</text>'),
]

LIVE = [
    ('<title>Majalis — society view (live replay)</title>', '<title>Majalis — 社会实况（真实运行回放）</title>'),
    ('content="Watch Majalis\'s agent society process a contradictory evidence stream, with the learned world model\'s risk meters live."',
     'content="观看 Majalis 智能体社会处理相互矛盾的证据流，学习世界模型的风险仪表实时可见。"'),
    ('/ society view', '/ 社会实况'),
    ('"status">seed', '"status">种子'),
    ('steps ·\n', '步 · '),
    ('correct ·', '正确 ·'),
    ('total · recorded', '总成本 · 录制于'),
    ('<a href="/zh/live" hreflang="zh-CN" lang="zh-CN">中文</a>', '<a href="/live" hreflang="en" lang="en">English</a>'),
    ('A real recorded run\n(not a mock): dated filings, stale echoes and rumors arrive on the left of the\nfeed; the belief board\'s <strong>learned world model</strong> re-scores every\nbelief as they land; the gate spends debate only where P(wrong) spikes.\nSpace = play/pause, arrows = step.',
     '一次真实录制的运行（并非模拟数据）：带日期的公告、过期回声与谣言依次到达；信念板的<strong>学习世界模型</strong>随之为每条信念实时重新打分；门控只在 P(wrong) 飙升处发起辩论。空格 = 播放/暂停，方向键 = 逐帧。'),
    ('>Play<', '>播放<'),
    ('label>speed', 'label>速度'),
    ('aria-label="Playback speed"', 'aria-label="播放速度"'),
    ('aria-label="Timeline scrubber"', 'aria-label="时间轴"'),
    ('Belief board <span class="n">— learned P(wrong) per belief</span>',
     '信念板 <span class="n">— 每条信念的学习 P(wrong)</span>'),
    ('Society feed <span class="n">— extract · propose · gate · skeptic · judge</span>',
     '社会动态 <span class="n">— 抽取 · 提议 · 门控 · 质疑 · 裁决</span>'),
    ("'Pause'", "'暂停'"),
    ("'Play'", "'播放'"),
    ("'Evidence arrives'", "'证据到达'"),
    ("'Extractor'", "'抽取器'"),
    ("'Question'", "'问题'"),
    ("'Proposer'", "'提议者'"),
    ("'World-model gate'", "'世界模型门控'"),
    ("'Skeptic'", "'质疑者'"),
    ("'Judge'", "'裁决者'"),
    ("'Proposer (re-proposal)'", "'提议者（重新提议）'"),
    ("'Result'", "'结果'"),
    ("'debate' : 'commit'", "'辩论' : '提交'"),
    ("['answered',", "['已答',"), ("['debates',", "['辩论',"),
    ("['spent',", "['已花费',"), ("['beliefs',", "['信念',"),
    ("'weak source'", "'弱信源'"),
    ("'adjudicated'", "'已裁决'"),
    ("'0 LLM calls'", "'0 次 LLM 调用'"),
    ("`asserted ${e.asserts.length} facts — ${outcomes}`", "`断言了 ${e.asserts.length} 条事实 — ${outcomes}`"),
    ("`answers “${t.answer}” (confidence ${t.confidence}) from ${t.support.length} beliefs`",
     "`基于 ${t.support.length} 条信念回答 “${t.answer}”（置信度 ${t.confidence}）`"),
    ("`attacks ${t.key}: `", "`质疑 ${t.key}：`"),
    ("`upholds ${t.key}`", "`维持 ${t.key}`"),
    ("`overturns ${t.key} → corrected to “${t.corrected}” (written back to the board)`",
     "`推翻 ${t.key} → 更正为 “${t.corrected}”（已写回信念板）`"),
    ("`now answers “${t.answer}” from the corrected board`", "`基于更正后的信念板改答 “${t.answer}”`"),
    ("`  p(wrong)=${g.p_wrong} — ${g.reason}`", "`  p(wrong)=${g.p_wrong} — ${g.reason}`"),
    ("(gold: ", "(标准答案: "),
    ("} answer \u201c", "} 回答 \u201c"),
    ("`event ${n + 1}/${evs.length}`", "`事件 ${n + 1}/${evs.length}`"),
    ('aria-label="Belief board"', 'aria-label="信念板"'),
    ('aria-label="Society activity feed"', 'aria-label="社会动态"'),
    ('aria-label="Agent activity"', 'aria-label="智能体动态"'),
    ('aria-label="System log"', 'aria-label="系统日志"'),
    ('aria-label="Replay controls"', 'aria-label="回放控制"'),
    ('Backbones: proposer/judge', '骨干模型：提议者/裁决者'),
    ('· skeptic', '· 质疑者'),
    ('· extractor\n', '· 抽取器 '),
    ('Gate: learned world model + conformal threshold, 0 LLM calls per\ndecision.',
     '门控：学习世界模型 + 共形阈值，每次决策 0 次 LLM 调用。'),
    ('Or switch to <strong>live</strong> and feed\nthe society your own evidence.',
     '也可以切换到<strong>实况</strong>模式，把你自己的证据喂给这个社会。'),
    ('aria-label="Viewer mode"', 'aria-label="查看模式"'),
    ('>recorded run<', '>录制回放<'),
    ('>live — try it<', '>实况 — 试一试<'),
    ('aria-label="Live session controls"', 'aria-label="实况会话控制"'),
    ('Evidence — one dated line each, filings beat rumors',
     '证据 — 每行一条、带日期；公告优先于谣言（示例保留英文，抽取器按英文模板解析）'),
    ('>Feed the society<', '>喂给社会<'),
    ('>Ask<', '>提问<'),
    ('placeholder="token (optional)"', 'placeholder="令牌（可选）"'),
    ('aria-label="Access token"', 'aria-label="访问令牌"'),
    ('aria-label="Question or claim"', 'aria-label="问题或断言"'),
    ('anonymous callers share a small daily budget — real Qwen calls, ~$0.01 per question',
     '匿名访问共享每日小额预算 — 真实 Qwen 调用，每问约 $0.01'),
    ("'your own evidence, the real society, the real learned gate — every call spends real tokens'",
     "'你自己的证据、真实的社会、真实的学习门控 — 每次调用都消耗真实 token'"),
    ("'live session started — feed evidence, then ask'", "'实况会话已开始 — 先喂证据，再提问'"),
    ("'society working — real Qwen calls in flight…'", "'社会运转中 — 真实 Qwen 调用进行中…'"),
    ("'paste at least one evidence line'", "'请至少粘贴一行证据'"),
    ("'type a claim or question first'", "'请先输入断言或问题'"),
    ("`${e.board.length} beliefs · ${liveQ} asked · ${liveFired} debates · $${liveCost.toFixed(4)} this session`",
     "`${e.board.length} 条信念 · 已问 ${liveQ} · 辩论 ${liveFired} · 本会话已花 $${liveCost.toFixed(4)}`"),
    ("`answer \u201c${e.answer}\u201d · ${e.tokens} tokens · $${e.cost_usd}`",
     "`回答 \u201c${e.answer}\u201d · ${e.tokens} tokens · $${e.cost_usd}`"),
    ('aria-label="The society"', 'aria-label="智能体社会"'),
    ('>Extractor<', '>抽取器<'),
    ('>Proposer<', '>提议者<'),
    ('>WM gate<', '>WM 门控<'),
    ('>Skeptic<', '>质疑者<'),
    ('>Judge<', '>裁决者<'),
    ('>0 LLM calls<', '>0 次 LLM 调用<'),
    ('<div class="lt">Belief graph</div>', '<div class="lt">信念图谱</div>'),
    ('</i>filing</div>', '</i>公告</div>'),
    ('</i>rumor / weak</div>', '</i>谣言 / 弱信源</div>'),
    ('</i>adjudicated</div>', '</i>已裁决</div>'),
    ('</i>ring = P(wrong)</div>', '</i>红圈 = P(wrong)</div>'),
    ('</i>debate edge</div>', '</i>辩论边</div>'),
    ('<span>value</span>', '<span>取值</span>'),
    ('<span>churn</span>', '<span>更替次数</span>'),
    ('<span>majalis society runtime</span>', '<span>majalis 社会运行时</span>'),
    ('aria-label="Belief graph"', 'aria-label="信念图谱"'),
]


def translate(src: Path, dst: Path, table: list[tuple[str, str]]) -> None:
    s = src.read_text()
    misses = []
    for en, zh in COMMON + table:
        if en in s:
            s = s.replace(en, zh)
        else:
            misses.append(en[:60])
    dst.write_text(s)
    print(f"{dst.name}: {len(COMMON + table) - len(misses)} applied, "
          f"{len(misses)} missed" + (f" — first miss: {misses[0]!r}" if misses else ""))


def main() -> None:
    translate(DASH / "index.html", DASH / "index.zh.html", INDEX)
    translate(DASH / "live.html", DASH / "live.zh.html", LIVE)
    # The replay payload must survive translation byte-identical.
    m_en = re.search(r"const R = (\{.*?\});\nconst MODELS", (DASH / "live.html").read_text(), re.S)
    m_zh = re.search(r"const R = (\{.*?\});\nconst MODELS", (DASH / "live.zh.html").read_text(), re.S)
    assert m_en and m_zh and json.loads(m_en.group(1)) == json.loads(m_zh.group(1)), \
        "replay JSON altered by translation"
    print("replay JSON intact")


if __name__ == "__main__":
    main()
