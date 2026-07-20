<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
    <img alt="Majalis" src="assets/logo-light.svg" width="300">
  </picture>

  <h3>你的智能体辩论太多了。Majalis 的学习世界模型来决定何时值得辩论。</h3>

  <p>
    <a href="README.md">English</a> · <strong>简体中文</strong>
  </p>

  <p>
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT">
    <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/tests-128%20passing-brightgreen" alt="Tests">
    <img src="https://img.shields.io/badge/%E9%97%A8%E6%8E%A7%E5%86%B3%E7%AD%96-0%20LLM%20%E8%B0%83%E7%94%A8-blue" alt="零调用门控">
    <img src="https://img.shields.io/badge/Qwen%20Cloud-%E8%B5%9B%E9%81%93%203%EF%BC%9A%E6%99%BA%E8%83%BD%E4%BD%93%E7%A4%BE%E4%BC%9A-8A2BE2" alt="Qwen Cloud Hackathon">
  </p>

  <p>
    <a href="http://47.237.187.157:8080/zh">在线仪表盘</a> ·
    <a href="http://47.237.187.157:8080/zh/live">社会实况 — 回放或实时体验</a> ·
    <a href="http://47.237.187.157:8080/docs">API 演练场</a> ·
    <a href="docs/paper/majalis.pdf">论文（双语摘要）</a> ·
    <a href="docs/architecture.md">架构文档</a>
  </p>
</div>

单一骨干上的多智能体辩论大多是在用更高成本重复自洽采样（arXiv:2502.08788、2604.02460）。Majalis 为 Qwen 智能体团队提供一块**共享信念板**，以及一个**在社会自身运行日志上训练的世界模型**——只在信念状态可能被污染处触发辩论，对无辩论提交的答案给出共形保证（E[error | accepted] ≤ α），且**每次门控决策零 LLM 调用**。

```bash
git clone https://github.com/Nas01010101/majalis && cd majalis
pip install -e . && python examples/quickstart.py   # 无需 API key
```

```python
from majalis.beliefs import BeliefBoard, parse_date_ord
from majalis.wmnet import load_wm

board = BeliefBoard()
board.assert_fact("acme::ceo", "Jane Doe", parse_date_ord("Jan 2026"), source="Filing")
board.assert_fact("acme::ceo", "John Roe", parse_date_ord("Mar 2026"), source="Rumor")

wm = load_wm()
wm.wrong_now(board, "acme::ceo")   # ~1.0 —— 谣言顶掉了权威公告：值得辩论
```

## 命令行工具

`pip install -e .` 会同时安装 `majalis` 命令。`majalis replay` 把一段记录下来的社会运行轨迹渲染成
"冲突 → 共识"时间线——证据到达、信念被替换（或被弱来源污染）、门控触发并给出原因、怀疑者把挑战拆解成
可二元判定的子问题、裁判做出裁决、修正结果写回信念板——**全程无需 API key**，使用一个确定性的示例
文件（`examples/sample_trace.jsonl`，由 `scripts/gen_sample_trace.py` 基于真实的信念板/门控代码生成，
不发起任何 LLM 调用）：

```bash
majalis replay examples/sample_trace.jsonl        # 带样式的终端渲染
majalis replay examples/sample_trace.jsonl --json # 原始 JSON 记录，逐行输出
majalis --version
majalis demo                                      # 执行 scripts/demo_company.py —— 需要 DASHSCOPE_API_KEY
```

## 为什么选 Majalis

- **辩论是一笔开销决策。** 两个训练头——`wrong_now`：P(信念当前是错的)（AUROC **0.999 对 0.79**（被替换的手工门控）；在从未见过的**真实 LLM 信念板上 0.937**）与 `superseded_next`：P(即将被推翻)（**0.657 对 0.496 = 随机水平**的固定先验）——以 **0 次 LLM 调用**决定"直接提交还是辩论"。
- **有校准，不靠感觉。** ACCEPT 阈值在学习分数上做分裂共形校准，覆盖率实测验证：1,600 个留出问题上 accepted-error 2.1% ≤ α=0.05。
- **世界模型你可以自己重训。** `python scripts/gen_wm_dataset.py && python train/train_wm.py`——端到端约 2 分钟（torch 训练，numpy 推理）。被替换的手工启发式保留为消融项（`MAJALIS_WM=heuristic`）。
- **实时喂入你自己的证据。** 在[社会实况](http://47.237.187.157:8080/zh/live)切换到"实况 — 试一试"：粘贴带日期的证据行，观看抽取器构建信念、世界模型实时重新打分、门控只在 P(wrong) 飙升处发起辩论——全部经由真实部署的社会。（匿名访问共享每日小额预算。）
- **一个社会，三个领域。** 同一套角色 + 信念板 + 世界模型运行：(1) 合成证据流基准族；(2) 投委会尽调场景（[`scripts/demo_company.py`](scripts/demo_company.py)：门控捕获谣言投毒、规划器分解 GO/NO-GO）；(3) 带零额外调用单轮门控的 GSM8K——按领域重新参数化，从不分叉。
- **不是又一个"按题门控辩论"。** 2025–26 的门控文献——DOWN（arXiv:2504.05047）、iMAD（arXiv:2511.11306）、SELENE（EACL 2026）、ARMOR-MAD（arXiv:2606.13197）——都是*逐题、无状态*地基于一次新回答的置信度做决定。Majalis 门控的是**一份持久共享记忆的状态**：世界模型估计哪些*信念*当下已错、预测哪些即将被推翻（多时程风险曲线）、完全在想象中试演维护策略，并以保形保证控制*提交*决定。这正是它的成本曲线**随流长保持平坦**（而非逐题打折）的原因——也是经典黑板架构（Hearsay-II，1970 年代："下一个该触发哪个知识源？"）的控制难题，用学到的世界模型而非手写调度规则给出的解答。

## 实测结果

会话评测：带插问的证据流 + 不可靠信源；所有对比臂看到完全相同的事件，共享一本 token/美元账本（5 个种子，Wilson 95% 置信区间）。

| 对比臂 | 准确率 | 每问成本 | 说明 |
|---|---|---|---|
| **Majalis**（学习门控，默认） | **240/240，全部流长度** | **$0.0049–0.0054/问，平坦** | 门控决策 0 次 LLM 调用；约 12% 的问题发起辩论 |
| Majalis（手工门控，可选） | 303/304，全部流长度 | $0.0056，平坦 | 对比臂 `majalis`；对 6–16% 的问题辩论 |
| 单智能体 | 272/272 | $0.0079 → $0.0137，线性增长（32 步时 2.5×） | 每问重读整条流 |
| 朴素 MAD（3×3） | 32/32 | $0.0709 | Majalis 的 12.6 倍成本 |
| Majalis（去辩论消融） | 77/80 (96.2%) | $0.0060 | 3 个错误恰好是世界模型标记的谣言污染信念；门控辩论以每问 +$0.0004 全部纠正 |

门控质量，无需 API key（100 条未见证据流，<1 秒）：学习门控触发 **12.4%** / 捕获 **86.2%** 被污染信念板 / 误触发 **0.9%**，对比手工门控 **23.8% / 78.8% / 15.1%**。两者均满足覆盖界。

```bash
python scripts/offline_bench.py                                    # 门控质量 + 覆盖率，$0
python -m majalis.bench.session --arms single,majalis,mad --seeds 0,1,2,3,4
python scripts/e2e_live.py                                         # 对已部署服务的 14 项不变量端到端测试（约 $0.05）
```

**复现某个对比臂。** `--arms majalis` 运行手工门控，`--arms majalis-wm` 运行学习门控——
对比臂名称本身在代码中（`bench/session.py` 的 `REPLAYS`）固定了门控模式，
全新克隆无需任何隐藏设置即可复现任一行结果。若设置了 `MAJALIS_WM=learned|heuristic`
环境变量，它仍会覆盖对比臂的模式（用于临时调参），此时会向 stderr 输出醒目警告，
且结果将不再匹配已发布数值——复现 `results/session_summary.json` 请勿设置该变量。

**关于"确定性"。** 上方门控质量数据（`offline_bench.py`）是可证明确定性的——
零 LLM 调用，纯 numpy 基于固定种子重放。会话评测表格的 `--seeds` 参数是
*带种子但不保证确定性*：它作为 DashScope 的 `seed` 请求参数传递，Qwen Cloud
文档将其列为尽力而为（best-effort，与 OpenAI 自身的 `seed` 一致），因此即使
种子相同，实时重跑也可能与已提交数值相差一两题。

## 工作原理

星型拓扑社会，异构 Qwen 骨干（qwen3.7-max 提议者/裁决者 · qwen3.7-plus 质疑者 · qwen3.6-flash 抽取器，作者 ≠ 验证者），写入与我们赛道 1 记忆引擎 [Tenet](https://github.com/Nas01010101/tenet) 同一键控取代设计的双时序信念板（独立的进程内实现——相同的 `entity::attribute` 碰撞语义、退役而非删除）；学习世界模型随证据到达为每条信念重新打分，共形门控决定提交或辩论，辩论裁决写回信念板形成闭环。详见[论文](docs/paper/majalis.pdf) · [架构文档](docs/architecture.md)。

## 许可证

MIT
