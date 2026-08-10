# FU-MD CVPR 研究计划

## Memory Is Not Always Helpful: Counterfactual Utility Control for Long-Horizon Video Generation

版本：2026-08-09  
状态：CVPR method-backed phenomenon 主线；取代“只做 SlotMem 审计即可投稿”的设定  
主平台：Wan2.2-I2V-A14B + SlotMem Stage-2 预训练权重  
主数据：NarraStream-Bench 中经预注册规则筛出的角色重现样本  
外部验证：ViStoryBench 或 ST-Bench 角色重现样本；第二记忆系统优先 SlotMemory/Wan2.1  

---

# 0. 即时生效的研究判决

1. **CVPR 主线必须有方法贡献。** 单一 SlotMem 四臂审计、负例统计或 probe 可作为动机与保底证据，不能单独承担 CVPR 主会创新性。
2. **不再把 relevance 当 utility。** prompt similarity、attention mass、recency 和角色匹配只表示“相关”，不表示该记忆会改善最终生成。
3. **Utility 只由 decoded outcome 定义。** latent denoising、hidden-state delta、reconstruction gain 均只作 attribution、筛查或特征。
4. **SlotMem-native 是主实验平台。** 冻结 Wan2.2、SlotMem LoRA、Memory Encoder、Writer 与 Reader；先验证预训练记忆通道，再训练独立 utility predictor。
5. **Self-Forcing 暂不承担主方法结果。** 当前 Wan2.1/Self-Forcing + Long-RVOS 结果保留为历史诊断；仅在 SlotMem 主线通过后作为跨 rollout regime 扩展。
6. **首轮因果臂为 no-memory / zero / correct / wrong。** 当前“slot 行置换 scramble”可能因 set-attention 的排列不变性成为数学 NOOP，不进入阻断判据。
7. **冻结 prefix，在 target recurrence chunk 分叉。** 所有生成侧比较共享逐字相同的 target prompt、reference/local context、初始噪声和 sampler；完整去噪轨迹允许因干预而分叉。
8. **phenomenon-first 仅是保底路径。** CVPR submission target 必须同时包含：低方差 counterfactual measurement、decoded-utility distillation、selective abstention/control。

---

# 1. 论文中心命题

现有长视频记忆方法主要优化“检索到相关历史”或“维持角色身份”。但一段相关、身份匹配的记忆仍可能因为姿态、视角、背景、状态、动作或去噪阶段不匹配而损害当前生成。项目研究：

> Can a long-video generator estimate the causal marginal effect of reading a memory before committing to that read, and abstain when the expected decoded outcome is harmful?

将长视频生成写为：

\[
Y_k = G_\theta(P_k,L_k,M_k,Z_k),
\]

其中 \(P_k\) 为 target prompt，\(L_k\) 为冻结的局部 prefix/reference condition，\(M_k\) 为候选长期记忆，\(Z_k\) 为初始噪声。生成侧反事实结果差为：

\[
\Delta O_k(M)=O(Y_k\mid M)-O(Y_k\mid \varnothing),
\]

\(O\) 是 decoded video 的多维结果向量，而不是 latent loss。研究目标是学习：

\[
\hat{\Delta O}_k=f_\phi(P_k,L_k,M_k,H_k,A_k),
\]

并执行质量约束下的选择：

\[
\pi_k(M)=\operatorname{READ}
\iff
\operatorname{LCB}(\widehat{\Delta C}_{id})>\delta_{id}
\land
\operatorname{LCB}(\widehat{\Delta Q_j})\ge-\delta_j,\ \forall j.
\]

其中 \(H_k\) 表示 gap/horizon，\(A_k\) 表示固定轨迹 attribution 特征，LCB 是保守置信下界。

---

# 2. 相对现有工作的创新边界

| 方法族 | 已解决 | 未解决、由本项目处理 |
|---|---|---|
| StoryMem / OneStory | 选择历史关键帧、构造全局记忆 | 记忆对当前 decoded outcome 的边际效用 |
| MemFlow | prompt-relevant frame/token retrieval | relevance 不能区分 helpful 与 harmful memory |
| Mixture of Contexts | query-to-history 稀疏路由 | attention relevance 不提供生成侧反事实效用 |
| Memorize When Needed | camera/revisit-aware gate | 非几何的身份、状态、动作和质量冲突 |
| SlotMemory | object-centric KV storage/retrieval | 读记忆前的 outcome-calibrated abstention |
| SlotMem | character-addressable slots、writer、localized injection | correct character memory 何时仍然不应读取 |

CVPR 版本只保留三个主贡献：

1. **Content-causal memory intervention protocol。** 在冻结 prefix 和 prompt 下，以 correct/wrong/zero/none 分离内容、通路与结构效应。
2. **Fixed-Trajectory Counterfactual Attribution。** 固定 \(z_t\) 轨迹，仅切换 memory condition，得到 step/layer/patch 级低方差影响图，但不把它误定义为 utility。
3. **Decoded-Utility Distillation and Conservative Routing。** 用昂贵的 decoded paired rollouts 监督轻量 predictor，预测多维 outcome delta 与 harm probability，在质量非劣约束下 READ/NOOP。

不主张：

- 第一个发现 memory 会有害；
- 第一个 selective memory 方法；
- 新 benchmark；
- latent improvement 等价于生成 improvement；
- SlotMem、SlotMemory 或所有 slot memory 普遍失败。

---

# 3. 模型与系统范围

## 3.1 主平台

固定：

- Backbone：Wan2.2-I2V-A14B；
- Memory system：SlotMem Stage-2 官方 checkpoint；
- Generation：SlotMem 官方 chunk-wise autoregressive inference；
- Parameters frozen：backbone、LoRA、Memory Encoder、Memory Writer、Character-Wise Cross-Attention；
- Trainable：独立 utility predictor；通过 Gate M4 前不得修改 generator 或 memory system；
- Environment：与 Self-Forcing 完全分离的 conda/env；
- Disk：约 126 GB Wan2.2 base + 约 21 GB SlotMem checkpoints + code/cache；
- Reproducibility：冻结 SlotMem commit、checkpoint SHA256、Wan2.2 manifest 和运行参数。

已有部署骨架：

- `VideoMem/scripts/setup_slotmem.sh`；
- `VideoMem/scripts/slotmem_content_audit.py`。

## 3.2 外部平台

外部验证按成本递增：

1. 同一 SlotMem checkpoint 上的第二数据分布；
2. SlotMemory/Wan2.1 的 inference-only 四臂内容审计；
3. 若代码与权重允许，在 SlotMemory 上训练只读 utility head；
4. Self-Forcing 上的 SlotMem-SF 移植仅作扩展，不是主线前置条件。

跨 backbone 的绝对指标不进入同一公平性能排名；只报告系统内 paired effect、标准化效应和方向一致性。

---

# 4. 数据策略

## 4.1 主数据

NarraStream-Bench 提供 multi-prompt narrative scripts。先运行 eligibility audit，仅纳入：

1. 同一 story 内存在稳定角色名称；
2. 角色在 memory chunk 可见；
3. 至少缺席一个完整 chunk；
4. 在 target chunk 明确重现；
5. reference image、source chunk 与 target prompt 可完整追溯；
6. 排除含糊指代或无法确定实例身份的样本；
7. 每个 story 至少形成一个可冻结 prefix 的 recurrence event。

## 4.2 实体标识

每个样本同时保存：

```json
{
  "story_id": "story_017",
  "entity_uid": "story_017::car_01",
  "character_name": "the white sedan",
  "memory_chunk_idx": 1,
  "target_chunk_idx": 4
}
```

- `character_name` 进入 prompt，供 SlotMem probe/addressing；
- `entity_uid` 不进入 prompt，只定义 correct/wrong；
- 文本相同、外观相似、类别相同都不能把跨 story 实例变成 correct。

## 4.3 Wrong donor 匹配

wrong donor 必须：

- `entity_uid` 不同；
- 优先来自不同 story/source；
- 匹配粗类别、颜色描述、角色数量、source visibility、gap bucket 和 slot shape；
- 不允许以明显异类 donor 夸大 correct-wrong gap；
- donor 配对表在运行前冻结并记录 seed。

## 4.4 Split

先统计 eligible recurrence stories 数量 \(N_e\)，再冻结 split，不能假设全部 narrative scripts 都满足 recurrence：

- development：拆成两个不相交子集（§7.2）——`dev-M2` 固定 12 个 story 跑四臂筛查门，`dev-metric` 另取若干 story 标定 \(\delta_{id}\)、指标量程与人评锚点。二者都只用于接线、阈值和指标稳定性，均不进 formal test；
- 若 \(N_e\ge192\)：formal test 固定 64、validation 固定 32，其余进入 train/calibration；
- 若 \(128\le N_e<192\)：formal test 固定 48、validation 固定 24，其余进入 train/calibration，并降低 harmful-rate headline 强度；
- 若 \(N_e<128\)：不得在单一数据源上同时训练 controller 和声称正式分布结论，必须先补第二训练数据源；
- split 按 story/source 分组，任何 reference image、entity 或原视频不得跨 split；
- 如果 formal eligible stories 少于 48，\(P(U<0)\) 不得作为强 headline，必须补第二公开数据源或缩小主张。

正式测试集在所有方法、阈值、metric weights 和 non-inferiority margins 冻结前不可读取。

## 4.5 外部数据

ViStoryBench 或 ST-Bench 只承担外部分布验证：

- 角色 reference 明确；
- story/source 与主数据不重叠；
- 不在外部测试集上重新选择阈值；
- 若任务格式与 SlotMem 不完全一致，适配规则先在 development stories 冻结。

Long-RVOS 保留为 oracle-mask/GT mechanistic control，不承担 narrative utility headline。它有 GT target video，因此专门用于验证 P3：除无符号 influence map 外，计算真实方向的去噪误差改善

\[
S^{GT}_{t,\ell,p}(m)=
\|v_{\theta,\ell,p}(z_t,\varnothing)-v^*_{t,\ell,p}\|^2-
\|v_{\theta,\ell,p}(z_t,m)-v^*_{t,\ell,p}\|^2.
\]

该读数验证 influence magnitude/方向诊断是否对应 latent 改善；它仍不能替代 narrative decoded utility。

## 4.6 W1 eligibility 生死门

eligibility audit 与 M0 并行，且在任何 M1–M3 GPU 扩展前完成：输出每条过滤规则的排除数、\(N_e\)、recurrence-event 数、source/story 去重报告和候选 split。若 \(N_e<128\)，立即停止“单数据源训练 controller + 正式分布结论”路线，先补第二训练数据源；不得先做四周实验再发现 estimand 不成立。

---

# 5. Prompt 与冻结 prefix 契约

## 5.1 基本原则

“固定 prompt”指同一 target chunk 在不同 memory arms 中逐字一致，而不是所有故事 chunk 使用同一句文本。

## 5.2 Prefix snapshot

每个 recurrence event：

1. 只生成一次 chunks \(1{:}k-1\)；
2. 保存 target 前的 image condition、末帧、local context、memory bank、sampler state；
3. 保存 target chunk 初始噪声；
4. 四臂从同一 snapshot 分叉；
5. 仅修改 `get_memory_payload()` 的返回或关闭 reader。

## 5.3 逐臂不变量

必须相等：

- target prompt bytes 与 SHA256；
- negative prompt；
- conditional/unconditional text embeddings；
- reference image；
- prefix/local-context state；
- initial noise；
- sampler、timesteps、CFG、resolution；
- target character query/name；
- injection layers 与 checkpoint。

允许不同：memory payload 与由此产生的后续去噪轨迹。

## 5.4 Prompt regimes

- Native prompt：主实验，原始 narrative prompt；
- Identity-light prompt：只作机制分析，在实验前统一删减外观细节；
- 两个 regime 分层报告，不合并；
- 任何 arm 不得单独改 prompt；
- 禁止加入“remember”“same as before”等只在 memory arm 出现的提示。

---

# 6. 因果干预与探针

## 6.1 首轮四臂

| Arm | 操作 | 回答的问题 |
|---|---|---|
| no_memory | 完全关闭 reader | 无长期记忆基线 |
| zero | 保持 payload shape，tokens 置零 | hook/位置/零值结构偏置 |
| correct | 同 story、同 `entity_uid` | 正确内容是否产生作用 |
| wrong | matched donor slots 替换，但 query/prompt 不变 | 作用是否依赖实例内容 |

## 6.2 Scramble 处置

Slot slots 作为 K/V 集合被 cross-attention 读取时，行置换满足近似排列不变性：

\[
\operatorname{Attn}(Q,PK,PV)=\operatorname{Attn}(Q,K,V).
\]

因此现有 row-permutation scramble 不进入 gate。只有通过 end-to-end self-check 证明输出改变后才可恢复。可选结构干预为：

- Memory Encoder 前的空间 token shuffle；
- 跨 layer memory swap；
- feature-channel permutation；
- slot dropout。

这些均为探索性，不替代 correct-wrong 内容因果判据。

## 6.3 P0：Provenance probe

逐 run 记录：

- story/chunk/entity/source/donor ID；
- prompts 与全部 hashes；
- code/checkpoint hashes；
- memory reads、none reads、transformed layers；
- seed/noise/sampler；
- arm 与 intervention effectiveness。

任何 arm 未真正执行干预则整组作废。

## 6.4 P1：Memory-bank probe

逐 character/layer/step 记录：

- slot count、norm、pairwise cosine；
- writer update 前后 cosine 与 residual norm；
- correct-wrong slot distance；
- source chunks 与 bank occupancy；
- attention entropy 与 selected roles。

## 6.5 P2：Read-path probe

在 Character-Wise Cross-Attention 前后记录：

\[
\Delta h_{t,\ell}=h^{after}_{t,\ell}-h^{before}_{t,\ell}.
\]

报告：

- \(\|\Delta h\|/\|h\|\)；
- target-mask 与 background 的 delta energy；
- correct-wrong-zero 的差异；
- step/layer 曲线；
- memory attention entropy；
- non-target spill ratio。

## 6.6 P3：Fixed-Trajectory Counterfactual Attribution

保存一条中性 no-memory target trajectory \(\{z_t\}\)。对每个固定 \(z_t\) 只做条件切换：

\[
A_{t,\ell,p}(m)=
\|v_{\theta,\ell,p}(z_t,m)-v_{\theta,\ell,p}(z_t,\varnothing)\|.
\]

输出：

- patch attribution map；
- target/background localization ratio；
- correct-wrong attribution gap；
- layer/timestep influence profile；
- 单次完整 rollout 等价成本比。

这里的 \(A\) 是**无符号影响幅度**：只回答 memory 把预测推动了多少，不能回答方向是否正确；零均值噪声也可能使其变大。它只能作为内容通路筛查、空间归因图或 teacher-side predictor 特征，decoded rollout label 才赋予 helpful/harmful 方向。P3 只能枪毙无内容通路，不能单独批准 utility controller。

另加不需要 GT 的 correct-wrong 伴随诊断：令

\[
d^{cw}=v_\theta(z_t,m_{correct})-v_\theta(z_t,m_{wrong}),\qquad
S^{cw}(m)=\frac{\langle v_\theta(z_t,m)-v_\theta(z_t,\varnothing),d^{cw}\rangle}{\|d^{cw}\|_2+\epsilon}.
\]

\(S^{cw}\) 只测 correct-wrong 条件方向可分性，不证明方向有益；由于部署时没有 wrong donor，它只进入 M2/teacher 诊断，不作为在线 student 输入。Long-RVOS 上再用 §4.5 的 \(S^{GT}\) 验证该仪器的效力。

---

# 7. Decoded outcome 与 Utility 定义

## 7.1 Outcome vector

在 target recurrence chunk 上计算主读数。**这是一个刻意的作用域限制，必须在论文里写明**：估计量是"此刻这一次读取的边际效果"，所以主 endpoint 只看被干预的那个 chunk。但 2026-08-08 观测到伤害会**跨 chunk 累积**，只测 target chunk 会系统性低估 harmful rate。因此同时报告 **target+1 chunk 的同一 outcome 向量作为次要读数**（不进主 endpoint、不参与 Holm 校正的主族），用于给出延迟伤害的下界。若次要读数显示 harmful rate 明显高于主读数，结论须改写为"逐 chunk 决策低估了累积代价"，而不是沿用主读数的数字。

对 target recurrence chunk 计算：

\[
O(Y)=
[C_{id}, A_{prompt}, Q_{bg}, Q_{motion}, Q_{flicker}, Q_{boundary}, Q_{anatomy}, Q_{non-target}].
\]

W2 结束前冻结以下 metric card、版本、预处理、方向与缺失值规则；未实现并完成 repeatability/human-anchor audit 前，不得定义 helpful/harmful：

| 分量 | 冻结的主实现 | 输入/掩码与方向 | 交叉验证及限制 |
|---|---|---|---|
| \(C_{id}\) | DINOv2-L/14 masked-crop cosine | Grounding-DINO 指代表达检测 + SAM2.1 跨帧 mask；target crop 对 canonical source/reference crop，逐帧取 trimmed mean，越大越好 | 人物清晰正脸子集加 ArcFace；盲评 identity 为最终锚点 |
| \(A_{prompt}\) | GPT-4.1、temperature 0、固定 JSON rubric；记录 API 返回的精确 model ID | 分别判 action、scene、entity count，盲 arm/方法，三项均值越大越好 | 与 SlotMem 论文主 evaluator 对齐；至少 20% development pairs 由盲人评交叉验证，不作为唯一 harm 证据 |
| \(Q_{bg}\) | masked DINOv2 background consistency | 在相邻帧共同有效的非实体 patch 上计算 DINOv2 cosine 后取均值；实体 mask 来自 Grounding-DINO + SAM2.1，越大越好 | 同时报告官方 VBench Background Consistency；mask 失败样本转人工审计，不回填全帧分数 |
| \(Q_{motion}\) | VBench Motion Smoothness + Dynamic Degree，**两项分别设门** | smoothness 走 non-inferiority；**dynamic degree 走绝对硬地板**，不用相对 no-memory 的非劣（基线可能本身已偏静，非劣会把"一起冻结"判为通过） | 与盲评“运动自然性”交叉验证 |
| \(Q_{flicker}\) | VBench Temporal Flickering | target chunk 全帧，按官方方向统一为越大越好 | 同时报告原始实现版本，禁止自行改归一化 |
| \(Q_{boundary}\) | NarraStream Boundary Smoothness | prefix 最后窗口与 target 开头窗口，越大越好 | 同报 Conditional Adjacent，主门只用前者 |
| \(Q_{anatomy}\) | VBench Human Anatomy | 仅 human-visible 子集，越大越好 | 非人物/不可见记为 N/A，不按 0 或均值填充；另行报告有效 \(n\) |
| \(Q_{non-target}\) | DINOv2 masked-crop cosine + entity-count preservation | Grounding-DINO + SAM2.1 对每个非目标实体追踪，分别与 prefix/source 匹配后取最差 identity cosine；另记检测数差，越大越好 | 实体漏检/错配进入盲评；不能用 target identity 分数替代 |

把 dynamic degree 从"只报告"提升为硬地板，是 2026-08-05 的直接产物：那次注入退化为**纯平滑，62/62**，而平滑类指标同时**变好**——冻结画面把 Motion Smoothness 打满。只对 smoothness 设门等于给冻结吸引子留正门。这不是假想：SlotMem 自己 README 的 VBench Dynamic Degree 跨方法从 0.3913 到 0.9130，2.3 倍差距，说明这个方法族里确实有方法在冻结。

VLM 限制改为：`A_prompt` 可以由固定 VLM 主评并由盲人评校准，但整体 harmful label 不得仅由同一个 VLM 决定；`C_id`、背景、运动、边界和非目标保持保留独立视觉/规则读数。所有 evaluator 输入、mask、模型版本、rubric 与输出必须缓存，formal test 前冻结。

### 7.1.1 尺子的量程：ceiling 正对照（blocking）

2026-07-25 作废整个 8-cell null 的三重混杂之一就是**尺子没有 ceiling**。没有它，\(\Delta C_{id}=0.01\) 没有可解释的量纲，\(\delta_{id}\) 只是一个裸数字。W2 冻结指标时必须同时测出并记录：

- \(C_{id}(\text{reference},\text{reference})\) —— 构造上界，检验实现是否自洽；
- \(C_{id}(\text{真实视频的后段 chunk},\ \text{其自身首帧})\) —— **真实外观变化下可达的上界**，即这把尺子在本任务上的实际量程顶端；
- \(C_{id}\) 在 no-memory 双 seed 之间的重复噪声 —— 量程底端。

二者之间的区间就是有效量程。所有 \(\Delta C_{id}\) 与 \(\delta_{id}\) 必须相对该量程报告（占量程的比例），而不是只报绝对值。同一套正对照对 \(Q_{bg}\)、\(Q_{non\text{-}target}\) 同样适用。

量程若窄到与重复噪声同量级，则该分量不具备做主 endpoint 的分辨率，须在 W2 换实现或降为次要读数——而不是等 M3 跑完再解释为什么效应量这么小。

## 7.2 Counterfactual deltas

\[
\Delta O_i(m)=O(Y_i^m)-O(Y_i^{none}).
\]

定义三类标签，而不是先拍脑袋混成一个分数：

- helpful：\(\Delta C_{id}>\delta_{id}\) 且所有质量项满足 non-inferiority；
- harmful：\(\Delta C_{id}< -\delta_{id}\)，或任一硬质量项越过 harm margin；
- neutral：其余情况。

\(\delta_{id},\delta_j\) 根据重复测量误差、human-anchor agreement 和 no-memory seed variability 冻结，formal test 不得调整。

**标定集必须与 M2 的 12 个 development stories 不相交。** 否则阈值被拟合在它自己要检验的那批数据上：M2 的判据是 `correct − wrong` 的 median 超过 \(\delta_{id}\)，而 \(\delta_{id}\) 又由同一批故事的 seed variability 估出——这批故事若恰好方差小，\(\delta_{id}\) 就小，M2 在同样这批上更容易过。这是 2026-07-20 "step900 的选择偏差必须用新 clip 确认"的同型问题。因此 development 预算拆成 `dev-metric`（标定 \(\delta\)、量程与人评锚点）与 `dev-M2`（12 个故事跑四臂），两者故事不重叠，且都不进 formal test。

主 \(\Delta O\) 在通过 Gate A（§10 M2）的 story 上统计——相对退化基线的差值不构成 utility——但 Gate A 由**独立 qualification seed** 判定，且**全体 eligible 与 Gate-A-qualified 两个总体的结果必须并列报告**，每个 \(P(U<0)\) 都标注其条件分布。

## 7.3 报告量

- 平均 \(\Delta C_{id}\) 与各 \(\Delta Q_j\)；
- helpful/neutral/harmful proportions 与 95% CI；
- \(P(U<0)\) 对应 harmful rate，但必须同时展示其构成；
- worst-case tail、CVaR 或 lower quantile；
- gap、类别、单/多角色和 prompt transition 分层结果；
- paired effect size 与 story-cluster CI。

---

# 8. Utility predictor 与控制策略

## 8.1 输入

部署 student 只允许使用生成 target 前、native 单次前向已经可观测的信息：

- SlotMem layer-wise role slots 的 pooled statistics；
- target query hidden features，在 probe mask 内池化；
- memory-query cosine、cross-attention summary；
- gap/horizon、角色数量、memory age/update count；
- target prompt embedding；
- checkpoint/timestep/layer identifiers。

不使用 target decoded pixels、未来 GT 或生成后指标。

P3 需要额外 fixed-trajectory 条件前向，因此只属于离线 teacher/label-generation 路径；不得隐藏在“轻量 MLP”前作为在线必需输入。额外训练一个 `P3-on online teacher` 作上界，并与不含 P3 的 deployed student 消融。若未来部署 P3，必须把完整 probe 前向计入端到端延迟。

## 8.2 最小模型

首版使用轻量 two-tower MLP：

1. memory tower 编码 slots；
2. query tower 编码 target/prompt/local context；
3. student concat \([m,q,m\odot q,|m-q|,horizon]\)；online teacher 才附加 attribution；
4. heads 输出多维 \(\widehat{\Delta O}\)、harm probability 与 uncertainty。

不引入 V-JEPA，不训练新视频 encoder，不修改 SlotMem generator。只有首版通过 M4 后，才比较小型 transformer 或 layer/timestep head。

## 8.3 监督来源

- 主监督：correct vs no-memory 的完整 decoded paired rollouts；
- wrong memory：只作辅助 hard-negative、内容校准和 ranking；
- fixed-trajectory attribution：只能作为 P3-on teacher 的输入特征或辅助一致性约束，不能作为 deployed student 的必需输入，也不能作为 utility label；
- seed 作为重复测量，不作为独立 story。

## 8.4 Loss

\[
\mathcal L=
\lambda_\Delta\mathcal L_{Huber}(\widehat{\Delta O},\Delta O)
+\lambda_h\mathcal L_{BCE}(\hat p_{harm},y_{harm})
+\lambda_r\mathcal L_{rank}
+\lambda_c\mathcal L_{Brier}.
\]

- `Huber`：多维 decoded delta regression，各维按 train split robust SD 标准化；
- `BCE`：harmful classification；
- `rank`：同一 target 的 correct、wrong、none 候选排序，只是训练期 hard-negative 正则；部署不存在“wrong memory”动作，wrong 不进入 READ/NOOP 决策候选；
- `Brier`：概率校准；
- loss weights 只在 validation 冻结；
- 不使用 denoising MSE、recon gain 或 latent advantage 代替 \(\Delta O\)。

## 8.5 Router

MVP 策略只做 chunk-level `READ/NOOP`：

\[
\pi(M)=\text{READ}
\iff
LCB(\widehat{\Delta C}_{id})>\delta_{id}
\land
LCB(\widehat{\Delta Q_j})\ge-\delta_j.
\]

只有 MVP 显著优于 all-memory 后，才扩展：

- memory-candidate selection；
- layer-level gate；
- denoising-step gate；
- continuous scale。

这样避免用一个全局 decoded label直接弱监督大量 step/layer action。

---

# 9. 公平基线

同一 SlotMem/Wan2.2 checkpoint 上比较：

1. `no-memory`；
2. `all-memory` / SlotMem-native；
3. `random-read`，读取率匹配本方法；
4. `recent`；
5. `prompt-similarity`；
6. `attention-mass`；
7. `character-address-only`；
8. `oracle decoded utility`，只作上界；
9. `online fixed-trajectory teacher`；
10. `distilled utility router`。

所有 policy 必须共享相同 memory bank、candidate set、prefix、prompt、noise、checkpoint 和可用信息。不得把外部模型绝对指标与同底座 policy 表混在一起。

---

# 10. 实验链与阶段门

## E0：Eligibility audit（零 GPU，W1）

在 M0 并行执行、在 M1/M2 扩展前判决。通过条件：过滤报告可复现，且 \(N_e\ge128\)；否则方法线只允许在补足第二训练数据源后继续。E0 失败不妨碍 M0 部署复现，但禁止启动 controller label 生产，也禁止预注册单源 \(P(U<0)\) headline。

## M0：官方复现

拆成两个目标，避免把“能跑”冒充“复现”：

- `M0a deployment`：确认 checkpoint、官方 sample 和 native inference 完整运行；
- `M0b numeric anchor`：运行前冻结 SlotMem 论文 Table 1 的主锚点：NarraStream-Bench `Subject Consistency = 0.8771`；次锚点为 ViStoryBench `Character Similarity = 0.8603`。只有使用论文相同 checkpoint、benchmark inputs、预处理和 evaluator 时，才要求主锚点绝对差不超过 0.02 或其 bootstrap 95% CI 覆盖 0.8771。若官方评测输入不可获得，M0b 标记为 blocked/non-comparable，不能用官方单样例替代数值复现，也不能声称复现论文性能。

通过：

- 官方样例生成成功；
- checkpoint/commit/runtime manifest 落盘；
- SlotMem memory read 日志非空；
- 单 story 的 wall time、峰值 VRAM、输出目录结构已记录。

失败：停止全部研究，先修部署；不改模型。

## M1：Prompt/prefix/intervention contract

目的：证明四臂只改变 memory。

通过：

- prompt/text-embedding/reference/prefix/noise hashes 跨臂一致；
- correct/wrong/zero 的 transformed layers > 0；
- **输出级干预生效**：`correct` 与 `no_memory` 的解码输出必须不同，判据是逐帧 L1 中位数超过**技术重复噪声地板**——即**同 seed、同条件、同 snapshot 重跑一次**所得的差（cuDNN 非确定性、atomics、TF32 之类的数值不确定性）。先做一次确定性自检：若该重跑逐位相同，地板为 0，任何非零差异即算生效。
  **不得用跨 seed 方差当地板**：不同 seed 之间差的是生成随机性，量级远大于技术噪声，用它设门会让干预明明生效的情况也判不过。跨 seed 方差是 §11 里 M3 生成不确定性的估计量，两者不可混用。hook 触发不等于输出改变——2026-08-08 的 held-out 评测里注入通路全程活跃，而 `content_zero` 与 `no_memory` 在全部 10 个指标上**逐位相同**；2026-08-05 的 Phantom（\(p=0.740\)，"实验没有发生"）是同一课在训练侧的形态。只查 hook 会让两者都通过契约；
- `zero` 与 `no_memory` 的关系是**被测量的结论，不是契约条件**。全零 K/V 经 softmax 后残差为零，因此二者相等是解析上的预期；若它们不等，那是加性/位置偏置存在的证据，须记入 §6.1 的 zero 臂读数，而不是判 M1 失败；
- **寻址命中断言**：每个 recurrence event、每个应当携带记忆的臂，target character 都解析到 ≥1 slot，并记录命中的角色与 slot 数。SlotMem 是 character-addressable，寻址落空时 `get_memory_payload` 会返回空载荷，`correct` 臂于是静默退化成 `zero` 却仍被计为 correct——这正是 2026-07-21 那个 `require_dataset_prompts` 静默回退模板的坑换了一个系统。P1 记录被读取角色是探针，不能替代本断言；
- none 确实不读 memory；
- 在含 `memory → update → target` 的 Stage-2 样例上，writer update count > 0、更新 residual norm 大于数值 epsilon、bank hash 至少改变一次；若 writer 未动，只能把该实验标成 static-bank read audit，不得主张 dynamic memory；
- run manifest 可独立复核。

失败：后续结果全部不可归因。

## M2：Content-causal read path

固定使用 12 个 development stories 运行 P0–P3 与四臂；技术失败必须按预冻结规则替换，不能缩小分母后通过。

通过：

- **Gate A：`no_memory` 臂自身是合法比较基线**（恢复旧管线的 `gate_a`，本轮重写中丢失）。在 target chunk 上，no-memory 输出必须满足质量硬门的绝对下限：主体可检出、\(Q_{motion}\) 的 dynamic degree 高于冻结地板、\(Q_{flicker}\) 与 \(Q_{anatomy}\) 在冻结范围内、且 chunk 内不出现主体替换。
  理由：MVP1 记录的教训是 **no-memory 是退化下限**。若基线本身不连贯，所有相对它的 \(\Delta\) 被系统性抬高，"memory 有用"平凡成立，而**有害尾部反而观测不到**——地板已经在最低处，\(P(U<0)\) 会被压向 0。

  **Gate A 必须用独立的 qualification seed 判定。** 若用 formal outcome 的同一个 seed 的 \(Y^{none}\) 去筛 story，再在同一 seed 上算 `correct − none`，就是在**条件化于被减数**：被筛掉的恰是 \(Y^{none}\) 实现值差的那些，剩余样本的 \(\Delta\) 分布被系统性压缩，harm rate 失真。这是 selection-on-the-dependent-variable，不是样本清洗。因此：

  - Gate A 用一组**冻结的 qualification seeds** 判定，与 formal comparison 使用的 seeds 不相交；
  - 用独立 seed 判定后，Gate A 成为 story 层面的**前分层变量**（这个故事的基线是否稳定），而不是对实现结果的条件化；
  - **两个总体都要报**：全体 eligible population，以及 Gate-A-qualified population；
  - \(P(U<0)\) 必须**明确标注是哪一个条件分布**，两者不得混用同一个数字。未通过 Gate A 的 story 单独列表并报告其数量与失败原因，不得静默丢弃。
- \(\Delta h\) 非零且非全局平滑；
- correct 与 wrong 的 attribution/hidden response 可分；
- zero 与 no-memory 的差异可解释且小于 correct content effect；
- target-region delta concentration 高于 background；
- 主 decoded identity 对比 `correct - wrong` 至少 10/12 stories 为预注册有利符号，且 median difference 超过 W2 冻结的 repeatability margin \(\delta_{id}\)。

M2 是**单向筛查门**：失败足以停止方法线；通过只允许进入 M3，不构成论文证据，12 个 development stories 不进入 formal test。

失败分支：

- read 未执行：修 harness；
- latent 可分但 decoded 不可分：checkpoint/数据不适配，不能训练 utility；
- correct≈wrong：SlotMem 在该设置下不是可用内容通道，切换 SlotMemory 或缩为审计论文。

## M3：Utility census

在 calibration/test 候选集上先做 correct vs none 双臂、至少两个生成 seed。wrong 子集在看任何 decoded outcome 前按 story hash 确定性抽取至少 25%，并按实体粗类、gap bucket、角色数量分层；抽样 manifest 冻结。wrong 仅用于内容校准和 \(\mathcal L_{rank}\) hard-negative，不进入部署动作集，也不得以结果导向选择“容易 donor”。

通过：

- decoded outcome 中存在可重复的 helpful/neutral/harmful 异质性；
- harmful 不是单一 metric artifact，并有质量/人评交叉验证；
- all-memory 与 no-memory 均不是逐 story 支配策略；
- formal effect/precision 计划可由 pilot SD 和 event rate 计算。

若 all-memory 全面支配：停止 router，转为“内容因果但无负 utility”的结果。  
若 no-memory 全面支配：预训练 checkpoint/数据失配，不主张通用 memory harm。

## M4：Utility prediction

训练 predictor，正式测试前冻结。

通过：

- harm probability 校准优于 relevance baselines；
- predicted multi-outcome deltas 在 held-out stories 上有稳定 rank/误差；
- oracle-policy gap 显著小于 relevance-policy gap；
- 不依赖 donor ID、story ID 或 prompt template 泄漏。

## M5：Policy value

定义主 endpoint 上的 oracle-gap closure：

\[
G_{oracle}=O_{oracle}-O_{all},\qquad
R_{close}=\frac{O_{student}-O_{all}}{G_{oracle}}.
\]

若 \(G_{oracle}\) 未超过 W4 冻结的 policy SESOI，则该数据上没有足够 estimand，禁止用不稳定比值宣称成功。

通过：

- utility router 相对 all-memory 的 paired decoded outcome CI 下界 > 0，且点改善至少达到 policy SESOI；
- \(R_{close}\) 点估计至少 25%，并报告 story-cluster bootstrap CI；
- 满足全部质量 non-inferiority hard gates；
- 优于 random/recent/similarity/attention routing；
- READ rate 既非全 0 也非全 1；
- 效果在至少两个数据 strata 或第二数据分布方向一致。

M5 是 CVPR 方法主张的最低门。

## M6：Generalization

优先顺序：

1. ViStoryBench/ST-Bench 外部分布；
2. SlotMemory inference-only content/utility audit；
3. utility head 轻量迁移；
4. Self-Forcing/SlotMem-SF 扩展。

至少完成 1；若标题使用宽泛的“long-horizon video generation”，还应完成 2 或明确收缩到 character-addressable narrative memory。

---

# 11. 统计与功效

## 11.1 分析单位

- 独立单位：story/source；
- recurrence events、chunks、frames、patches 和 seeds 均嵌套在 story 内；
- 不能把帧或 patch 当独立样本；
- 多事件 story 用 cluster bootstrap 或 mixed-effects model。

## 11.2 Common random numbers

完整生成比较共享 prefix snapshot、prompt、initial noise 和 seed；干预后 trajectory 合法分叉。  
P3 attribution 才共享固定完整 \(z_t\) trajectory。

## 11.3 主统计

- story-cluster paired bootstrap 95% CI；
- helpful/neutral/harmful proportion 使用 Wilson 或 Jeffreys CI；
- 多个次要 outcome 用 Holm correction；
- 同时报告 effect size、CI、raw story counts；
- 预注册 primary endpoint、quality hard gates 和唯一主要比较 `utility-router vs all-memory`。

## 11.4 样本量

不能在没有 decoded pilot SD 时声称完成均值功效分析。流程：

1. M3 development pilot 获取 story-level paired delta SD、harm rate、seed ICC；
2. 对 planned cluster model 做 Monte Carlo power；
3. SESOI 由 identity metric repeatability 与 human-meaningful difference 定义；
4. 目标 power 0.80，two-sided \(\alpha=0.05\)；
5. 报告 effect-size sensitivity curve，不使用 observed power。

比例估计的最坏情形 \(p=0.5\) 下，95% Wald 半宽仅作直观规划：

| 独立 stories | 约半宽 |
|---:|---:|
| 48 | ±0.14 |
| 64 | ±0.12 |
| 96 | ±0.10 |

因此 formal test 首选 64；只有在 eligible 数据不足但仍不少于 128 个独立 stories 时才降为 48，并必须收缩 \(P(U<0)\) 的 headline 强度。

---

# 12. 评价可靠性与人评

1. 所有自动 identity metric 先在 development 上与盲人评做相关性与错误审计；
2. 人评配对展示时隐藏 arm 和方法名，随机左右顺序；
3. 评价问题拆开：身份一致性、prompt/action、运动、视觉质量、非目标保持；
4. 不让同一 VLM同时生成 prompt、选择样本并裁决结果；
5. 报告 inter-rater agreement 与无决定比例；
6. VLM-based score 不作为唯一 harm 证据。

---

# 13. 计算预算

预算按 gate 发放，不预先整包批准。

| Gate | 计算内容 | 预算规则 |
|---|---|---|
| E0 | 全量 script eligibility/filter audit | 零 GPU；W1 完成，先于方法扩展 |
| M0 | 1 个官方 sample | 实测 runtime/VRAM 后建立成本模型 |
| M1 | 同 sample 四臂接线 | 只批准最小 arm/hook 验证 |
| M2 | 固定 12 stories，四臂，probe | M1 与 metric reliability 通过后 |
| M3-pilot | correct/none，少量 stories × 2 seeds | M2 通过后 |
| M3-formal labels | train/val/test paired rollouts | pilot 确认异质性后 |
| M4 | 轻量 predictor | label cache 固定后；不重复生成 |
| M5 | formal policy comparison | 模型和阈值冻结后一次运行 |
| M6 | 外部分布/第二系统 | M5 通过后 |

成本公式：

\[
GPUHours_{rollout}=N_{story}\times N_{seed}\times N_{arm}\times T_{story}\times N_{GPU}.
\]

P3/teacher 的额外 fixed-trajectory forwards 另记为 \(GPUHours_{probe}\)，端到端成本为两者之和；不得只报告 MLP latency。M0 后记录实际 \(T_{story}\) 与每 story 的 probe 时间，W5 label 预算才可批准：

\[
B_{label}=N_{train}(2N_{seed,main}+0.25N_{arm,wrong}N_{seed,wrong})T_{story}+GPUHours_{probe}.
\]

若超预算，按以下预定顺序收缩：先取消预注册 wrong 子集以外的 zero/wrong；再将训练 label 降为 1 seed、只在 calibration/formal noise subset 保留 2 seeds；再按 source-stratified sampling 缩减训练 labels。不得因此静默缩小 formal test；仍超预算则停止 controller claim。缓存 prefix、memory bank、prompt embeddings、reference features 和 decoded evaluator inputs，禁止重复生成可复用中间量。

---

# 14. 十二周 CVPR 节奏

| 周 | 交付物 | 停机检查 |
|---|---|---|
| W1 | E0 eligibility audit；并行完成独立 env、权重、官方样例、论文数值锚点、\(T_{story}\)/probe 成本 | E0 + M0 |
| W2 | prefix snapshot、prompt hash、四臂 harness、writer movement、**输出级干预与寻址命中断言**；实现并冻结八维 metric card 与盲评 rubric；**§7.1.1 量程正对照**；拆出不相交的 `dev-metric` / `dev-M2` | M1 + metric reliability + 量程 |
| W3 | P0–P3 probes、`dev-M2` 12-story content audit、**Gate A 基线连贯性** | M2 |
| W4 | decoded utility pilot、metric/human anchor | M3 pilot |
| W5 | 仅当 E0/M2/M3-pilot 通过且实测预算容纳时：冻结正式 split、启动 label generation | 条件预算批准 |
| W6 | 在 W5 条件成立时完成 label cache、冻结 utility definition | 数据污染审计 |
| W7 | predictor MVP、校准与 relevance baselines | M4 |
| W8 | conservative READ/NOOP router | 内部 policy gate |
| W9 | formal main comparison | M5 |
| W10 | 外部数据、关键 ablations、效率 | M6 |
| W11 | 人评、失败案例、主图表、论文初稿 | claim traceability |
| W12 | 复跑关键结果、内部 review、补充材料、提交缓冲 | 全部 hard gates |

W1 的 \(N_e<128\) 且无第二训练源，或 W3 结束仍无法证明内容通道，立即停止 CVPR method 主线。W5–W6 是由实测 \(T_{story}\)、probe 成本和 \(N_e\) 决定的条件窗口，不是无条件两周承诺。

---

# 15. 核心消融

只做与主张直接对应的消融：

1. correct/wrong/zero/none；
2. online teacher 的 fixed-trajectory features on/off，以及不含 P3 的 deployed student；
3. decoded label vs latent/reconstruction proxy label；
4. uncertainty/LCB on/off；
5. multi-outcome constraint vs identity-only；
6. chunk-level router vs relevance baselines；
7. native prompt vs identity-light prompt；
8. matched wrong vs easy wrong；
9. global router vs layer/timestep extension，仅在 MVP 通过后；
10. predictor size/latency 与完整 router 端到端延迟；P3-on teacher 必须计入额外条件前向。

不做与主线无关的 V-JEPA carrier、复杂 continual learning、INVALIDATE/UPDATE、长期 eviction controller，除非 M5 已完成且仍有预算。

---

# 16. 预期表格与图

## Figure 1：现象与动机

同一冻结 prefix/prompt/noise 下，all-memory 对部分 stories 有益、部分有害；展示 target 与 background 的反事实差异。

## Figure 2：方法

SlotMem frozen generator → fixed-trajectory probe → decoded rollout teacher labels → utility predictor → conservative READ/NOOP。

## Figure 3：Attribution

correct/wrong/zero 在 step × layer × patch 的影响图，证明内容与空间落点。

## Figure 4：Utility distribution

helpful/neutral/harmful 分布及 gap/角色类别分层，显示不是均值假象。

## Figure 5：Policy frontier

identity/quality/harm rate/latency 的 Pareto 图，对比 all-memory、relevance 与 utility router。

## Table 1：同底座主结果

Wan2.2 + SlotMem 上的 no-memory、all-memory、heuristics、teacher、student。

## Table 2：内容因果与 proxy failure

四臂 decoded outcome、fixed attribution、latent vs decoded mismatch。

## Table 3：消融与校准

feature、loss、uncertainty、quality constraint。

## Table 4：外部分布与效率

第二数据分布、第二记忆系统方向、runtime/VRAM/label cost。

---

# 17. CVPR Go/No-Go

## 最低 CVPR package

必须同时满足：

1. SlotMem 预训练记忆在目标数据上是内容敏感的；
2. decoded utility 存在可重复异质性；
3. 无符号 fixed-trajectory attribution 被明确限定为 influence magnitude；它在 online teacher 中对 decoded harm/helpfulness label 有额外预测信息，且 Long-RVOS 有符号 GT 诊断支持其效力，不能凭 attribution 范数本身判断 harm；
4. utility predictor 在 held-out story/source 上泛化并校准；
5. 不含在线 P3 的 deployed router 优于 all-memory 与 relevance routing，改善达到 policy SESOI 且至少闭合 25% 的有效 oracle gap；
6. 全部质量 hard gates 通过；
7. 至少一个外部分布结果方向一致；
8. 代码、日志、splits、prompt/donor manifests 可复现。

## 只够保底、未达到 CVPR method bar

- 只有四臂审计；
- 只有“memory 有时有害”的比例；
- 只有 attribution map；
- predictor 能拟合但 policy 不改善；
- 仅在单一 checkpoint/少量故事上成立；
- 改善来自全部 NOOP 或质量下降；
- 用 latent/recon proxy 冒充生成侧效用。

---

# 18. 退路

## R1：SlotMem public checkpoint 不适配

先确认官方 sample，再确认目标数据。若官方 sample 有效、目标数据无效，则结论限于分布迁移，不宣称 SlotMem 失败；切换 SlotMemory 或重构 in-domain narrative subset。

## R2：correct≈wrong

停止 utility；完成统一内容因果审计，考虑分析型投稿，不继续训练 controller。

## R3：utility 无异质性

若 all-memory 支配，utility control 没有 estimand；转为 content-causal memory measurement。若 no-memory 支配，报告 checkpoint/data mismatch，不泛化。

## R4：utility 可测但不可预测

保留 fixed-trajectory online teacher 与 utility structure 论文；不能主张实用 router。

## R5：predictor 可用但不优于 relevance

停止 method claim，分析 relevance 何时已足够；不通过追加复杂 controller 挽救。

## R6：跨系统不一致

收缩标题与结论为 character-addressable SlotMem/Wan2.2，不写“video memory generally”。

---

# 19. 论文允许的最终主张

只有 M5/M6 通过后允许：

> Relevance is not utility: even identity-matched memory can have heterogeneous causal effects on decoded long-video outcomes. A low-variance fixed-trajectory instrument exposes where memory acts, and a decoded-utility predictor can conservatively abstain from harmful reads while preserving the benefits of helpful memory.

最低保底表述：

> Under a frozen character-addressable memory system, controlled content interventions reveal that apparent memory gains can decompose into content-specific, content-insensitive, and decoded-harm regimes.

---

# 20. 立即执行顺序

1. 冻结本计划、SlotMem commit 与权重 manifests；
2. 零 GPU 运行 E0 eligibility audit，输出 \(N_e\) 与逐规则排除报告；
3. 与 E0 并行在 GPU server 运行 `setup_slotmem.sh`，不污染 Self-Forcing env；
4. 运行 M0a 官方 sample，并在相同官方评测条件可得时核对 M0b `Subject Consistency=0.8771`；记录 \(T_{story}\)、probe 成本和 VRAM；
5. 实现并冻结八维 metric card、mask/缺失值规则、VLM rubric 和盲评 anchor；
6. 修正 `slotmem_content_audit.py`：将 row-scramble 从 gate 移除，增加 end-to-end intervention self-check；
7. 实现 prefix snapshot 与 prompt/reference/noise hash contract，提升 writer movement 为 M1 门；
8. 增加 P1/P2 日志、P3 无符号 attribution、correct-wrong 方向诊断与 Long-RVOS GT 验证；
9. 运行固定 12 个 development stories 的 M2，要求 identity favourable sign 至少 10/12；
10. 仅在 E0 与 M2 通过后运行 decoded utility pilot；
11. pilot 后冻结 SESOI、质量 margins、split、wrong 25% 子集和正式样本量；
12. 由实测 \(T_{story}\) 审批 label 预算后生成 decoded label cache；
13. 训练不依赖在线 P3 的轻量 utility predictor；
14. 比较 conservative router 与 all-memory/relevance baselines 及 25% oracle-gap closure；
15. M5 通过后做外部分布与第二系统；
16. 按 contribution-to-evidence 表完成论文图表与写作。

---

# 21. 核心相关工作锚点

- SlotMem：character-addressable role slots、Memory Encoder/Writer 与 localized cross-attention，主平台来源。<https://arxiv.org/abs/2607.15772>
- SlotMemory：object-centric KV memory，并已观察到过大 memory capacity 的质量权衡。<https://arxiv.org/abs/2605.31033>
- MemFlow：prompt-relevant frame retrieval 与 token activation，是 relevance baseline 的主要重叠项。<https://arxiv.org/abs/2512.14699>
- Mixture of Contexts：learned query-to-history sparse routing，说明“学习选择历史”本身不是创新点。<https://arxiv.org/abs/2508.21058>
- Memorize When Needed：camera-aware memory gating，要求本项目把 utility 与几何 revisit relevance 明确区分。<https://arxiv.org/abs/2604.18215>
- OneStory：CVPR 2026 adaptive memory/frame selection，代表主会对完整方法、数据和长视频验证的基准要求。<https://openaccess.thecvf.com/content/CVPR2026/html/An_OneStory_Coherent_Multi-Shot_Video_Generation_with_Adaptive_Memory_CVPR_2026_paper.html>
- NarraStream-Bench/IAMFlow：multi-prompt streaming narrative 数据与 identity-aware memory baseline。<https://arxiv.org/abs/2605.18733>

本计划的文献定位必须始终表述为：**existing work selects relevant or addressable memory; this project estimates its causal decoded consequence and abstains under predicted harm.**
