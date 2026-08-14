# 快速、全面、方法学可靠的验证框架

> 本文件把 `research-plan.md`（主线）、`stage-gates.md`（D0–D9 门）与 `experiment-protocol.md`
> （确证问题）里的验证动作，重排成一张"先便宜杀死、后昂贵确证"的可执行表。
> 它不新增任何实验，只定义**顺序、判据与归属**。可执行版本见 `utest/gates.py`
> （`python -m utest.gates --help`），其输出 PASS/BLOCK/PENDING 三种状态。

版本：2026-08-13。上一级：`research-plan.md` §10、§20；同级：`stage-gates.md`。

---

## 0. 一条原则

**廉价筛查层只做单向门：失败即停，通过不算证据；昂贵确证层的阈值全部在不相交数据上预先冻结。**
"快"靠的是把能杀死项目的测试前置并跑最便宜的形式，而不是跳过它们；"全面"靠的是每个确证问题都有唯一主实验 + 明确读数 + 预注册判据，而不是多跑实验。

---

## 1. 三层验证金字塔

| 层 | 成本 | 回答的问题 | 能杀死什么 | 对应模块/产物 |
|---|---|---|---|---|
| **L0 零 GPU** | 0 | estimand 成立吗、尺子有量程吗 | 数据不足、指标无分辨率 | `utest/eligibility.py` → `e0.json`；D2.5 量程审计 |
| **L1 廉价探针** | ~1 次 no-memory rollout + N 次条件 forward | 内容通道存在吗、干预真的发生吗 | 寻址落空、hook 触发但输出不变、correct≈wrong | `utest/content_audit.py` + `utest/event_harness.py` → `intervention_contract.json` |
| **L2 解码 rollouts** | ~1786s/story | 效用有异质性吗、能预测吗、路由更优吗 | 无异质性、不可预测、不优于 relevance | `utest/memory_utility.py` → `utility_report.json` |

**关键杠杆**：SlotMem 实测 1786s/story，而固定轨迹探针是"缓存一条 no-memory 轨迹 + 每个条件只做 forward"。
铁律——**90% 的 GPU 花在 L1 探针上，L2 只跑"过了筛查"的 story 和 predictor 恰好需要的 label。**

---

## 2. Kill-fast 决策表（`utest/gates.py` 的可执行形式）

按顺序执行；任何一个 BLOCK 即停，退回 `research-plan.md` §18 对应退路。PENDING 表示尚未运行，是"必须先跑"的信号，**绝不能被读成通过**。

| Gate | 问题 | 最便宜的判据 | 证据来源 | BLOCK 退路 |
|---|---|---|---|---|
| **E0** | estimand 存在吗 | `N_e >= 128` | `e0.json` | 补第二训练源，禁止单源 controller 主张 |
| **Q2** | 干预真的发生吗 | 四臂 hash 一致 + 寻址命中 ≥1 slot + correct≠no_memory 的逐帧 L1 > 同 seed 技术重复地板 + writer 动了 | `intervention_contract.json` | 修 harness，不改模型 |
| **Q3** | 尺子有量程吗 | 可达上界（真实视频后段 vs 自身首帧）与噪声地板（no-memory 双 seed）的区间 > 噪声 | D2.5（W2 冻结） | 换实现或降为次要读数 |
| **Q1** | 内容通道存在吗 | correct 与 matched wrong 可分，≥10/12 同号且 median > δ_id | M2（dev-M2 12 story） | R2：内容因果审计，不训 controller |
| **Q4** | decoded 效用有异质性吗 | helpful 与 harmful 都在度量噪声之上出现 | M3 pilot（`utility_report.json`） | R3：all-memory 支配→无 estimand；no-memory 支配→checkpoint 失配 |
| **Q5** | utility 可从廉价特征预测吗 | held-out 校准优于 relevance baseline，无 donor/story/template 泄漏 | M4 | R4：保留 online teacher，不主张 deployed router |

**Q2 尤其要前置**：2026-08-08（注入通路全程活跃而 `content_zero` 与 `no_memory` 十指标逐位相同）与
2026-08-05 Phantom（p=0.740）的教训是 **hook 触发 ≠ 实验发生**。Q2 的"输出级发散断言 + 寻址命中断言"
必须在任何 L2 花钱之前通过。

### 三个"只由人冻结、不由数据推断"的信号

`gates.py` 的 `--signals-json` 只应携带这三个域裁决（W2/W4 冻结，禁止从待检数据反推）：

```json
{ "ruler_range_usable": true, "content_causal": true, "predictable": false }
```

其余信号（`n_eligible_stories`、`intervention_contract_valid`、`helpful/harmful_present`）由 `gates.py` 从三个报告直接推导。

---

## 3. 全面性 = 一张可追溯矩阵

用 `experiment-protocol.md` §1 的四个确证问题做**行**，用证据层做**列**。每个格子三选一：✅ 已测（脚本 + 输出哈希）／⬜ 预注册待测（seed/split/threshold 冻结单）／➖ N/A（写死理由）。**全面性不是"都测了"，而是"每个格子有归属、每个归属可复核"。**

| 确证问题 | 四臂内容因果 | P2/P3 归因 | decoded 异质性 | 路由 vs relevance | 第二系统 | 外部分布 |
|---|---|---|---|---|---|---|
| 1. 传内容？ | ✅ M2 | ✅ P3 | — | — | ⬜ | — |
| 2. 有异质？ | — | — | ✅ M3 pilot | — | — | — |
| 3. 归因加值？ | — | ✅ P3 on/off 消融 | — | — | — | — |
| 4. 路由更优？ | — | — | — | ✅ M5 | — | ⬜ M6 |

代码归位：`eligibility.py`→E0/Q3；`content_audit.py`→Q1；`memory_utility.py`→Q4/Q5；
`attention_probe_utils.py`→P2/P3；`prefix_contract.py`→Q2 的 hash 半场；`stage_reports.py`→M0a/M0b。

---

## 4. 速度杠杆

1. **Common random numbers**：四臂共享 prefix snapshot / prompt / noise / sampler（`research-plan.md` §5）。
   prefix、memory bank、prompt embedding、reference feature **全部缓存**，任何 arm 不重生成可复用中间量。
2. **P3 一次 rollout 换 N 个条件**：缓存一条 no-memory 轨迹后，correct/wrong/zero 每个只是 forward。
   这是最大乘数，必须把"单次完整 rollout 等价成本比"写进成本模型，不单报 MLP latency。
3. **Seed 预算分级**：1 seed 默认，2 seed 只在 calibration + 噪声子集（`research-plan.md` §13 收缩顺序）。
4. **P3-as-triage**：用无符号影响幅度 + 寻址命中先筛掉"明显无内容通道"的 story，
   只对高信息量 story 跑 L2 label——主动/分层选择 label，而非全量 rollout。
5. **L1 全部闭环在 dev-M2 的 12 story**，绝不外溢到 formal；L2 的 label 集是 predictor + formal 的最小并集。

---

## 5. 让"快"不变"糙"的四条护栏

1. **阈值/量程/δ 只在 `dev-metric` 冻结**，与 `dev-M2`、formal 不相交（`research-plan.md` §7.2）。
2. **Gate A 用 qualification seed，绝不与 formal seed 同源**（`research-plan.md` §10 M2；`memory_utility.py` 已断言不相交）。
3. **每个廉价读数标 "screening, not confirmatory"**：L1 通过只是放行，12 个 dev story 永不进 formal test。
4. **每个 gate 预注册一个判定式**（例：D3 的 `correct−wrong median > δ_id 且 ≥10/12 同号`），见数据前写死、见数据后不调。

---

## 6. 落到周上的紧凑排期

| 周 | 动作 | 停机判据 |
|---|---|---|
| W1 | L0 全量：E0 eligibility + 尺子量程（D2.5）+ determinism 自检 + hash schema | E0 不过 → 停单源 |
| W2 | M1 契约 + **Q1/Q2 冒烟**（4–6 story correct-vs-wrong decoded + 寻址命中 + 输出级发散） | Q2 不过 → 修 harness；Q1 不过 → R2 |
| W3 | 12-story dev-M2 全量 + P0–P3 + Gate A | 内容通道证伪 → 停方法线 |
| W4 | M3 pilot（8–12 story correct-vs-none）+ 人评锚点 + 功率 Monte Carlo | 无异质性 → R3 |
| W5 | label 预算实测审批；P3-as-triage 定 label 抽样 | 预算超 → 按 §13 收缩 |
| W6 | label cache + Q5 冒烟（pilot label 上训 MLP） | 不可预测 → R4 |
| W7–9 | predictor + router + formal 主比较（冻结后一次跑） | 不优于 relevance → R5 |
| W10–12 | 第二系统 inference-only + 外部分布 + 人评 + 写作 | 方向不一致 → R6 收缩标题 |

与 `research-plan.md` §14 的差异只有三处，都是把**能杀死项目的测试前置**：
Q1/Q2 冒烟提前到 W2、SlotMemory inference-only 审计提前、P3 探针转成 label triage 而非纯离线诊断。

---

## 7. 与现有产物的关系（不重复、只聚合）

- 本文件**定义顺序与判据**，`stage-gates.md` 定义**每门的证据要求**，`research-plan.md` 定义**退路与主张**。
- `utest/gates.py` 把第 2 节的决策表变成一条命令，读 `e0.json` / `intervention_contract.json` / `utility_report.json`，
  打印 `PASS/BLOCK/PENDING` 记分板，BLOCK 时退出码 1。
- 运行顺序见 `utest/README.md`（E0 → M0a → M0b → M1 → M2 → M3）；本文件是它的"决策层"。
