# 学习拓展调研：从“会做 ASR”到“会研究、会判断、会负责”

调研日期：2026-08-18。

## 结论

仓库现有课程已经系统覆盖声音基础、PyTorch、CTC、流式、N-gram/WFST、前端、语义、泛化、部署和前沿模型。当前最值得补的不是第 47 个模型名，而是能够迁移到所有技术主题的研究与工程思维：

1. 把模糊现象改写成可证伪假设；
2. 把总分拆成错误类型、切片、分母和影响；
3. 用基线、对照、随机化、重复和因子实验避免错误归因；
4. 用配对区间、校准和最小有意义差异表达不确定性；
5. 用边界合同、不变量、变形测试和反事实定位根因；
6. 在质量、延迟、成本、隐私与风险之间做可审计决策。

因此，本轮新增 [`研究与工程思维 6 课`](notebooks/研究与工程思维课程索引.md)。它使用 ASR 小实验，但目标是训练通用思维，不要求先学完全部 ASR 主线。

在这条工程研究闭环之上，仓库又增加了 [`认知工具箱总手册`](COGNITIVE_TOOLKIT.md)与 [`认知拓展 8 课`](notebooks/认知拓展课程索引.md)，补齐论证与反例、基础率与贝叶斯、因果图、动态系统、决策与信息价值、信息论、概率预测校准和学习科学。新路线刻意标出每种工具的误用边界，并要求迁移到 ASR、工作和日常生活。

## 为什么选择这条路线

### 1. 标准评测并不止一个 WER

NIST SCTK 的 `sclite` 不只给总分，还能输出说话人、句子、混淆、插入/删除/替换、带标签子集和置信度分析。这说明“分数是多少”只是入口，“错在哪里、对谁错、为什么错”才导向行动。

- [NIST SCTK `sclite` 文档](https://github.com/usnistgov/SCTK/blob/master/doc/sclite.htm)
- [NIST SCTK 报告选项](https://github.com/usnistgov/SCTK/blob/master/doc/options.htm)

### 2. 好实验需要提前设计，而不是跑完后讲故事

NIST 工程统计手册把实验设计定义为：有计划地改变因素并观察响应，以有限实验获得有效、客观的结论。手册同时强调随机化、重复和分块；只做单因素逐次优化会漏掉因素交互。

- [NIST：什么是实验设计](https://itl.nist.gov/div898/handbook/pri/section1/pri11.htm)
- [NIST：随机化、重复与设计原则](https://www.itl.nist.gov/div898/handbook/pmd/section3/pmd33.htm)
- [NIST：单因素方法为什么会漏掉交互](https://www.itl.nist.gov/div898/handbook/pri/section2/pri212.htm)

### 3. 点估计不能独自支撑发布决定

有限测试集上的 WER 是样本估计。Bootstrap 可以近似统计量的抽样分布，但重采样单位必须匹配外推对象；比较两个系统时还要保留同一句/同一说话人的配对关系。置信分数也不等于正确概率，现代神经网络可能系统性过度自信。

- [SciPy：Bootstrap 置信区间](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [Guo 等：On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)

### 4. 可信系统需要切片、边界、风险与文档

NIST AI RMF 要求测试方法、指标、不确定性、部署条件、局限和风险可追溯，并建议对目标人群做分解评估。Model Cards 和 Datasheets for Datasets 分别补足模型与数据的用途、组成、评测条件和限制。

- [NIST AI RMF：Measure](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)
- [NIST AI RMF Playbook：分解评估](https://airc.nist.gov/airmf-resources/playbook/measure/)
- [Model Cards for Model Reporting](https://research.google/pubs/model-cards-for-model-reporting/)
- [Datasheets for Datasets](https://www.microsoft.com/en-us/research/publication/datasheets-for-datasets/)

## 课程覆盖矩阵

| 能力缺口 | 新课程 | 可验证产出 |
|---|---|---|
| 第一印象代替假设 | 01 可证伪假设 | 竞争假设卡、反证阈值、最小区分实验 |
| 总分掩盖失败模式 | 02 错误与切片 | S/D/I 对齐、切片表、辛普森悖论反例 |
| 多变量同时变化 | 03 对照与消融 | 2×2 因子实验、交互项、混杂审计 |
| 把小波动当改进 | 04 不确定性 | 说话人配对 Bootstrap、校准图、发布门禁 |
| 靠试错修故障 | 05 因果排错 | 管线边界、不变量、变形/反事实测试 |
| 单指标选型 | 06 Pareto 决策 | 硬约束、Pareto 前沿、风险与决策记录 |

## 后续仍值得学习的方向

以下不是本轮占位 Notebook，而是按收益和前置条件整理的真实拓展队列：

| 优先级 | 方向 | 为什么值得学 | 先修 | 建议作品 |
|---:|---|---|---|---|
| 1 | 语音科学与语音学 | 把 F0、共振峰、发音部位、音素混淆和声谱图联系起来 | 音频基础 01～06 | Praat 标注与最小对立词实验室 |
| 2 | 数据中心 ASR 与标注科学 | 很多上限来自覆盖、标签规范与分歧，而非模型 | 思维课 01～04 | 标注协议、双人一致性、数据卡和错误驱动采样 |
| 3 | 多语、口音与语言变化 | 平均指标会掩盖群体和域差异 | 主线 01～24、思维课 02/04 | 口音/设备/语速分层评测与适配边界 |
| 4 | 人机交互与语音 UX | 最低 WER 不一定带来最低任务失败率 | 语义课 37～41 | 确认、纠错、打断和低置信度回退原型 |
| 5 | 隐私、安全与对抗鲁棒 | 语音含身份、内容和环境信息，系统还会面对重放/注入 | 部署与前端课 | 威胁模型、重放测试、日志最小化和事件演练 |
| 6 | 端侧系统与硬件意识 | 延迟、内存、能耗和热约束决定真实可用性 | ONNX/INT8 25～30 | 质量—延迟—能耗 Pareto 基准 |
| 7 | 可观测性与 ML 事故响应 | 离线通过不能替代线上漂移、回滚和复盘能力 | 部署课、思维课 05/06 | 监控 SLO、故障注入、无责复盘和回滚演练 |

建议顺序：先完成本轮 6 课；随后优先做“语音科学与语音学”或“数据中心 ASR”，因为它们最能连接当前的音频软件分析、真实错误诊断和后续模型实验。

## 不把调研变成收藏清单

每新增一个方向，必须先写四项：目标场景、现有能力缺口、可运行最小实验、离场证据。若只能说“这是热门方向”，暂不进入课程；若无法设计反证和边界，也暂不声称掌握。
