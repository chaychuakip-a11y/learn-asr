# ASR Notebook 学习中心：零基础到前沿模型与真实盲测

这是 `notebooks/` 的唯一总入口。目录现在遵守三条规则：

1. 根目录只放可编辑、无预存输出的学习版 Notebook 和当前路线索引；
2. 已完整执行的输出对照统一放在 [`_executed/`](_executed/README.md)，并按课程类型分组；
3. 已被新索引取代的阶段性索引保存在 [`_archive/`](_archive/README.md)，不再作为学习入口。

完整的阶段依赖、时间安排和掌握门禁见仓库根目录的
[`LEARNING_PATH.md`](../LEARNING_PATH.md)。本页负责定位具体 Notebook，不再维护另一套互相冲突的学习顺序。

第一次学习时不要进入 `_executed/` 顺序阅读。请从
[Python/PyTorch 零基础路线](PyTorch零基础课程索引.md)、
[音频零基础桥梁路线](音频零基础课程索引.md)、
[第 01～41 课核心路线](核心课程索引_第01到41课.md)或
[语言模型专修路线](语言模型零基础_课程索引.md)中选择入口。

如果你能运行代码但看不懂每一行，先做
[代码伴读：零基础逐行理解 ASR](代码伴读_零基础逐行理解ASR.ipynb)，
卡住时查看[运行对照](_executed/pytorch_foundations/代码伴读_零基础逐行理解ASR_已运行.ipynb)。

## 文件怎么找

| 文件名或目录 | 作用 |
|---|---|
| `01_...ipynb` ～ `46_...ipynb` | ASR 主线学习版，按数字顺序学习 |
| `基础_01_...ipynb` ～ `基础_06_...ipynb` | Python/PyTorch 桥梁课 |
| `音频基础_01_...ipynb` ～ `音频基础_06_...ipynb` | 从振动、dB、PCM、SNR 到真实 WAV 的音频零基础桥梁课 |
| `语言模型零基础_01_...ipynb` ～ `09_...ipynb` | N-gram、OpenFst、WFST 与前沿 LM 专修 |
| `专题_...ipynb` | CTC、流式、WFST、部署、前端、语义和真实评测实验室 |
| `代码伴读_...`、`学习中枢_...`、`结课项目_...` | 辅助入口、诊断与综合项目 |
| [`_executed/`](_executed/README.md) | 分类保存的运行输出对照，不直接编辑 |
| [`_archive/`](_archive/README.md) | 被当前总入口取代的历史索引 |

课程材料已经完整交付为：

- 1 本含 14 个可运行小节的零基础逐行代码伴读；
- 6 节 Python/PyTorch 桥梁课；
- 6 节音频零基础桥梁课；
- 1 套 Audacity、Praat、Sonic Visualiser、Audition/Cool Edit [专业软件分析实验](../AUDIO_SOFTWARE_GUIDE.md)；
- 1 套含 24 条盲题、正常对照和命令行自测的 [音频问题诊断题库](../AUDIO_DIAGNOSIS_PRACTICE.md)；
- 46 节 ASR 主线课；
- 9 节 ASR 语言模型专修课；
- 1 个学习中枢、9 个专题实验室；
- 1 个可训练、可流式、可测试的结课声学引擎；
- FSDD 说话人泛化、六折 LOSO 和 AudioMNIST 外部盲测三层真实评测。

上述正式材料均提供两份 Notebook：根目录中不带 `_已运行` 的是无输出学习版，带输出的对照版集中在 `_executed/` 的分类目录中。

> 你现在可以先不答题。练习、离场票和闭卷验收都已保留，之后再做不会影响先阅读、运行和修改全部课程。

## 1. 唯一启动方式

在你克隆的 `learn-asr` 仓库根目录打开 PowerShell：

```powershell
uv sync --locked
uv run jupyter lab
```

代码读起来仍然吃力时，从 [零基础逐行代码伴读](代码伴读_零基础逐行理解ASR.ipynb) 开始，然后进入 [PyTorch 零基础课程索引](PyTorch零基础课程索引.md)。无论代码基础怎样，音频概念不扎实时都先完成 [音频零基础桥梁路线](音频零基础课程索引.md)，再进入 [第 1～41 课核心路线](核心课程索引_第01到41课.md)。

## 2. 从零到完整系统的推荐顺序

```text
零基础逐行代码伴读 01～14
→ PyTorch 导学与基础 01～06
→ 音频零基础 01～06：振动、周期、RMS、dB、PCM、SNR、WAV 审计
→ 专业音频软件实验：Audacity 波形/频谱 + Praat F0/Formant/TextGrid
→ 音频盲诊断：幅值、削波、DC、噪声、回声、采样率和多通道问题
→ ASR 主线 01～18：声音、Log-Mel、编码器、CTC、流式
→ 语言模型专修 01～08：N-gram、OpenFst、WFST、lattice
→ ASR 主线 19～41：LM/WFST、部署、音频前端、语义后处理
→ ASR 主线 42～46：Conformer、RNN-T/TDT、SSL、Audio-LM、Qwen3-ASR
→ 实时数字 CTC 结课项目
→ FSDD 说话人泛化 → 六折 LOSO → AudioMNIST 外部盲测
→ 语言模型 09：前沿 ASR LM 系统设计
```

没有 WSL/OpenFst 时，可以先跳过语言模型 03～08，继续学习声学主线；之后按 [安装与排错指南](../ASR_LM_ENVIRONMENT.md) 补做。

## 3. 主线课程地图

| 阶段 | 课次 | 完成后的能力 |
|---|---:|---|
| 代码伴读 | 伴读 01～14 | 变量、函数、shape、dtype/device、模型、训练、CTC、cache 与排错 |
| Python/PyTorch 预备 | 导学＋基础 01～06 | Tensor、shape、autograd、`nn.Module`、变长语音 Batch |
| 音频零基础桥梁 | 音频基础 01～06 | 从物理振动到 PCM 数组、dB、SNR 与真实 WAV 审计 |
| 专业音频软件实验 | GUI 实操 | Audacity、Praat、Sonic Visualiser、Audition/Cool Edit 概念映射与跨工具核对 |
| 音频盲诊断 | 24 条题 | 幅值、频谱、时间连续性、采样率、多通道问题的证据化诊断 |
| 声音与特征 | 01～06 | 从波形实现并验证 STFT、Mel 与 Log-Mel |
| 声学编码器 | 07～09 | mask、Linear、Conv1d、局部上下文与下采样 |
| CTC | 10～14 | 对齐路径、动态规划、`CTCLoss`、Prefix Beam、最小训练 |
| 流式 ASR | 15～18 | chunk/cache、因果模型、PGS、RTF 与延迟 |
| LM/WFST 系统 | 19～24 | N-gram、热词、FST、图组合、lattice、流式系统验收 |
| 量化与部署 | 25～30 | ONNX、INT8、HTTP/WebSocket、容器、并发、监控 |
| 麦克风前端 | 31～36 | 输入审计、降噪、VAD、AEC、波束形成、状态管线 |
| 后处理与语义 | 37～41 | 时间戳、Diarization、ITN、置信度、NLU、安全 LLM |
| 前沿建模 | 42～45 | Conformer、RNN-T/TDT、自监督、Audio Encoder＋LLM |
| 真实模型项目 | 46 | Qwen3-ASR 推理基线、资源预检、微调顺序、验收矩阵 |

第 01～41 课的逐课链接在 [核心路线索引](核心课程索引_第01到41课.md)。第 42～46 课如下：

1. [42：Conformer、卷积注意力与流式边界](42_Conformer_卷积注意力与流式边界.ipynb)｜[运行对照](_executed/asr_core/42_Conformer_卷积注意力与流式边界_已运行.ipynb)
2. [43：RNN-T/TDT、二维对齐和跳帧](43_RNNT与TDT_二维对齐和跳帧.ipynb)｜[运行对照](_executed/asr_core/43_RNNT与TDT_二维对齐和跳帧_已运行.ipynb)
3. [44：自监督语音预训练、遮挡与声学表示](44_自监督语音预训练_遮挡与声学表示.ipynb)｜[运行对照](_executed/asr_core/44_自监督语音预训练_遮挡与声学表示_已运行.ipynb)
4. [45：迷你音频语言模型、Audio Encoder 接入 LLM](45_迷你音频语言模型_AudioEncoder接入LLM.ipynb)｜[运行对照](_executed/asr_core/45_迷你音频语言模型_AudioEncoder接入LLM_已运行.ipynb)
5. [46：Qwen3-ASR 推理、微调与验收](46_Qwen3ASR_推理微调与验收.ipynb)｜[运行对照](_executed/asr_core/46_Qwen3ASR_推理微调与验收_已运行.ipynb)

前沿结论与论文依据见 [2026 ASR 前沿路线](../FRONTIER_ASR_2026.md)。第 46 课默认不下载大模型权重，也不会在没有 GPU、合规数据和真实运行结果时伪造训练成功。

## 4. 九节语言模型专修路线

完整逐课说明、依赖和里程碑见 [语言模型零基础课程索引](语言模型零基础_课程索引.md)。

| 课次 | 主题 | 学习版 | 运行对照 |
|---:|---|---|---|
| 01 | 从计数到 Bigram | [打开](语言模型零基础_01_从计数到Bigram.ipynb) | [对照](_executed/language_models/语言模型零基础_01_从计数到Bigram_已运行.ipynb) |
| 02 | 平滑、回退、OOV、困惑度 | [打开](语言模型零基础_02_平滑回退OOV与困惑度.ipynb) | [对照](_executed/language_models/语言模型零基础_02_平滑回退OOV与困惑度_已运行.ipynb) |
| 03 | FSA/FST 与第一张 OpenFst 图 | [打开](语言模型零基础_03_FSA_FST与第一张OpenFst图.ipynb) | [对照](_executed/language_models/语言模型零基础_03_FSA_FST与第一张OpenFst图_已运行.ipynb) |
| 04 | OpenFst 组合、确定化、最小化 | [打开](语言模型零基础_04_OpenFst组合确定化与最小化.ipynb) | [对照](_executed/language_models/语言模型零基础_04_OpenFst组合确定化与最小化_已运行.ipynb) |
| 05 | 从语料到 ARPA 与 G.fst | [打开](语言模型零基础_05_从语料到ARPA与Gfst.ipynb) | [对照](_executed/language_models/语言模型零基础_05_从语料到ARPA与Gfst_已运行.ipynb) |
| 06 | 词典 L、消歧与 HCLG/CTC-TLG | [打开](语言模型零基础_06_词典L消歧与HCLG_CTC_TLG.ipynb) | [对照](_executed/language_models/语言模型零基础_06_词典L消歧与HCLG_CTC_TLG_已运行.ipynb) |
| 07 | N-best、lattice 与二遍重打分 | [打开](语言模型零基础_07_Nbest_Lattice分数融合与二遍重打分.ipynb) | [对照](_executed/language_models/语言模型零基础_07_Nbest_Lattice分数融合与二遍重打分_已运行.ipynb) |
| 08 | 综合项目与冻结验收 | [打开](语言模型零基础_08_综合项目与闭卷验收.ipynb) | [对照](_executed/language_models/语言模型零基础_08_综合项目与闭卷验收_已运行.ipynb) |
| 09 | 前沿 ASR LM 系统设计 | [打开](语言模型零基础_09_前沿ASR语言模型系统设计实验室.ipynb) | [对照](_executed/language_models/语言模型零基础_09_前沿ASR语言模型系统设计实验室_已运行.ipynb) |

## 5. 专题实验室：把分散知识连成系统

| 实验室 | 学习版 | 运行对照 |
|---|---|---|
| 学习中枢与掌握度仪表盘 | [打开](学习中枢_诊断与掌握度仪表盘.ipynb) | [对照](_executed/labs/学习中枢_诊断与掌握度仪表盘_已运行.ipynb) |
| CTC：路径、动态规划、梯度、Prefix Beam | [打开](专题_CTC可视化实验室_从路径到流式解码.ipynb) | [对照](_executed/labs/专题_CTC可视化实验室_从路径到流式解码_已运行.ipynb) |
| 流式 ASR：Chunk、缓存、PGS、实时率 | [打开](专题_流式ASR实验室_Chunk缓存PGS与实时率.ipynb) | [对照](_executed/labs/专题_流式ASR实验室_Chunk缓存PGS与实时率_已运行.ipynb) |
| WFST：L/G 与流式 Token Passing | [打开](专题_WFST实验室_从L与G到流式TokenPassing.ipynb) | [对照](_executed/labs/专题_WFST实验室_从L与G到流式TokenPassing_已运行.ipynb) |
| ONNX/INT8 与服务验收 | [打开](专题_量化部署实验室_ONNX_INT8性能与服务验收.ipynb) | [对照](_executed/labs/专题_量化部署实验室_ONNX_INT8性能与服务验收_已运行.ipynb) |
| 音频前端：质量、VAD、AEC、波束形成 | [打开](专题_音频前端实验室_质量VAD_AEC与波束形成.ipynb) | [对照](_executed/labs/专题_音频前端实验室_质量VAD_AEC与波束形成_已运行.ipynb) |
| 语义后处理：时间戳、ITN、置信度、安全执行 | [打开](专题_语义后处理实验室_时间戳ITN置信度与安全执行.ipynb) | [对照](_executed/labs/专题_语义后处理实验室_时间戳ITN置信度与安全执行_已运行.ipynb) |
| FSDD 说话人泛化：划分、增强、盲测 | [打开](专题_FSDD说话人泛化实验_数据划分增强与盲测.ipynb) | [对照](_executed/labs/专题_FSDD说话人泛化实验_数据划分增强与盲测_已运行.ipynb) |
| FSDD 六折 LOSO：嵌套选择与说话人统计 | [打开](专题_FSDD六折LOSO_嵌套选择与说话人统计.ipynb) | [对照](_executed/labs/专题_FSDD六折LOSO_嵌套选择与说话人统计_已运行.ipynb) |
| AudioMNIST 外部盲测：跨域失败与适配边界 | [打开](专题_AudioMNIST外部盲测_冻结协议跨域失败与适配边界.ipynb) | [对照](_executed/labs/专题_AudioMNIST外部盲测_冻结协议跨域失败与适配边界_已运行.ipynb) |

## 6. 结课项目与真实评测阶梯

先完成 [实时数字 CTC 声学引擎](结课项目_实时数字CTC声学引擎_从WAV到流式文本.ipynb)；卡住时看[运行对照](_executed/capstone/结课项目_实时数字CTC声学引擎_从WAV到流式文本_已运行.ipynb)。项目包含：

- WAV 加载、Log-Mel、变长 Batch；
- CTC 训练、Greedy/Prefix Beam、Bigram LM；
- 流式 chunk/cache/partial hypothesis；
- CER、RTF、chunk 一致性和单元测试；
- 可选重新训练开关，不会默认覆盖 checkpoint。

然后依次做三层泛化评测：

1. [FSDD 说话人泛化指南](../FSDD_GENERALIZATION_GUIDE.md)：先验证“新说话人”而不是只会背训练样本；
2. [六折 LOSO 指南](../FSDD_LOSO_GUIDE.md)：每位说话人轮流只做一次外层测试，内层开发集负责选模型；
3. [AudioMNIST 外部盲测协议](../AUDIOMNIST_EXTERNAL_PROTOCOL.md)：冻结协议后跨数据集评测，观察域偏移、失败模式和允许的适配边界。

这条阶梯故意保留失败结果。跨域性能下降是课程证据，不应通过偷看外部测试标签或反复调参来“修漂亮”。

## 7. 运行成本与依赖边界

| 层级 | 可以做什么 | 额外条件 |
|---|---|---|
| 普通 CPU | PyTorch 基础、主线 01～45、大多数专题、结课项目 | `uv sync --locked` |
| WSL 工具链 | OpenFst/KenLM、语言模型 03～08 | 按安装指南配置 Ubuntu 工具 |
| 本地外部数据 | FSDD/AudioMNIST 泛化与盲测 | 按各自指南准备 `.local_data`，数据不提交 Git |
| GPU/模型权重 | Qwen3-ASR 真实推理与微调 | 显存、网络、许可、合规数据和可复现基线 |

## 8. 怎样先看完整课程、以后再答题

现在按以下方式学习即可：

1. 打开无输出学习版，先运行代码和修改参数；
2. 卡住时只在 `_executed/` 中对照同名 `_已运行.ipynb` 的当前小节；
3. 题目、离场票、闭卷实现先保留，不要求现在提交；
4. 以后准备验收时，再把答案、失败案例和独立实现记录到 [`LEARNING_LOG.md`](../LEARNING_LOG.md)。

“课程出完”现在指材料、代码、运行对照、索引、真实项目和验证器全部齐全；“真正学会”仍将在你之后愿意答题和亲手实现时单独验证。
