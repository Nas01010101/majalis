<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
    <img alt="Agora" src="assets/logo-light.svg" width="300">
  </picture>

  <h3>你的智能体辩论太多了。Agora 的学习世界模型来决定何时值得辩论。</h3>

  <p>
    <a href="README.md">English</a> · <strong>简体中文</strong>
  </p>

  <p>
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT">
    <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/tests-27%20passing-brightgreen" alt="Tests">
    <img src="https://img.shields.io/badge/%E9%97%A8%E6%8E%A7%E5%86%B3%E7%AD%96-0%20LLM%20%E8%B0%83%E7%94%A8-blue" alt="零调用门控">
    <img src="https://img.shields.io/badge/Qwen%20Cloud-%E8%B5%9B%E9%81%93%203%EF%BC%9A%E6%99%BA%E8%83%BD%E4%BD%93%E7%A4%BE%E4%BC%9A-8A2BE2" alt="Qwen Cloud Hackathon">
  </p>

  <p>
    <a href="http://47.237.187.157:8080/zh">在线仪表盘</a> ·
    <a href="http://47.237.187.157:8080/zh/live">社会实况回放</a> ·
    <a href="http://47.237.187.157:8080/docs">API 演练场</a> ·
    <a href="docs/paper/agora.pdf">论文（双语摘要）</a> ·
    <a href="docs/architecture.md">架构文档</a>
  </p>
</div>

多智能体辩论大多是在用更高的成本重复自洽采样。Agora 为智能体团队提供一块**共享信念板**，以及一个**在社会自身运行日志上训练的世界模型**——只在信念状态可能被污染的地方触发辩论，对无辩论提交的答案给出共形保证（E[error | accepted] ≤ α），且**每次门控决策零 LLM 调用**。

```bash
git clone https://github.com/Nas01010101/agora && cd agora
pip install -e . && python examples/quickstart.py   # 无需 API key
```

```python
from agora.beliefs import BeliefBoard, parse_date_ord
from agora.wmnet import load_wm

board = BeliefBoard()
board.assert_fact("acme::ceo", "Jane Doe", parse_date_ord("Jan 2026"), source="Filing")
board.assert_fact("acme::ceo", "John Roe", parse_date_ord("Mar 2026"), source="Rumor")

wm = load_wm()
wm.wrong_now(board, "acme::ceo")   # ~1.0 —— 谣言顶掉了权威公告：值得辩论
```

## 为什么选 Agora？

- **辩论是一笔开销决策。** 两个训练头——`wrong_now`（这条信念现在是错的吗？）和 `superseded_next`（它会被推翻吗？）——加上在真实日志上拟合的融合器，共同决定"直接提交还是发起辩论"。不用裁判模型、不用额外采样：**触发决策 0 次 LLM 调用**。
- **世界模型你可以自己重训。** `python scripts/gen_wm_dataset.py && python train/train_wm.py`——端到端约 2 分钟（torch 训练，numpy 推理）。被替换的手工启发式保留为消融项（`AGORA_WM=heuristic`）。
- **有校准，不靠感觉。** ACCEPT 阈值在学习分数上做分裂共形校准；覆盖率在离线基准上实测验证（1,600 个留出问题上 accepted-error 2.1% ≤ α=0.05）。
- **无需 API key 即可验证。** `python scripts/offline_bench.py` 在一秒内把 100 条从未见过的证据流跑过真实门控：学习门控触发 12.4% / 捕获 86.2% 被污染信念 / 误触发 0.9%，对比手工门控的 23.8% / 78.8% / 15.1%。
- **成本对流长度保持平坦。** 感知成本摊销进信念板（每问 O(board) 而非 O(stream)）；单智能体基线成本线性增长，Agora 不会。

| 学习头 | 预测目标 | 对比被替换的手工基线 |
|---|---|---|
| `wrong_now` | P(信念板当前值是错的) | AUROC **0.999 对 0.79**（合成验证集）；在从未见过的**真实 LLM 信念板上 0.937** |
| `superseded_next` | P(该事实即将被权威信息推翻) | AUROC **0.657 对 0.496（= 随机水平）**——被替换的固定 Lomax 先验 |

## 实测结果（会话评测，Wilson 95% 置信区间）

带插问的证据流 + 不可靠信源；所有对比臂看到完全相同的事件、按相同标准评分、共享一本 token/美元账本。

| 对比臂 | 准确率 | 每问成本 | 说明 |
|---|---|---|---|
| **Agora（学习门控）** | **48/48** | **$0.0054–0.0057，随流长度保持平坦** | 门控只在谣言污染的信念上触发；比手工门控省 25% token |
| Agora（手工门控） | 272/272 | $0.0056，平坦 | 冻结基线 |
| 单智能体 | 272/272 | $0.0079 → $0.0137，线性增长 | 每问重读整条流 |
| 朴素 MAD（3×3） | 32/32 | $0.0709 | Agora 的 12.6 倍成本 |
| Agora（去辩论消融） | 77/80 (96.2%) | $0.0060 | 3 个错误恰好是世界模型标记的谣言污染信念；门控辩论以每问 +$0.0004 全部纠正 |

复现：`make bench` 与 `python -m agora.bench.session --arms single,agora,mad --seeds 0,1,2,3,4`。
零 API 离线基准（门控质量、学习对比手工、覆盖率、可靠性——1,600 个留出问题，<1 秒）：`python scripts/offline_bench.py`。

## 为什么这样设计

单一骨干上的多智能体辩论大多是高价的自洽采样（arXiv:2502.08788、2604.02460）。经受住对照评测的杠杆是：模型异构、验证型任务、以及*稀疏性*——只在辩论可能改变答案时才辩论（参见 iMAD，arXiv:2511.11306）。Agora 用一个持久的、带校准的学习世界模型做稀疏决策（而非无状态的单查询分类器），并把辩论裁决写回信念状态，形成闭环。

## 组件

- **信念底座**：[tenet-memory](https://github.com/Nas01010101/tenet) —— 双时序键值信念库，含每事实 P(仍有效)。
- **共形控制**：CalibratedGate / TrajectoryCRC / AnytimeAlarm（preact-wm）—— 对已接受断言的分布无关保证。
- **社会**：星型编排器，异构 Qwen 骨干（qwen3.7-max / plus / qwen3.6-flash），作者≠验证者分离，类型化工件交接。

## 论文

方法、实验设置与全部结果（含中文摘要）：[docs/paper/agora.pdf](docs/paper/agora.pdf)（Markdown 源文件 [docs/paper/agora.md](docs/paper/agora.md)）。

为 Qwen Cloud 全球 AI 黑客松（赛道 3：智能体社会）构建。

## 许可证

MIT
