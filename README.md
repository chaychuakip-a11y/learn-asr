# Learn ASR：从声音基础到流式 CTC、WFST、语义与部署

一套面向初学者的可运行 Jupyter Notebook 课程。课程使用真实开源语音，从波形与频谱开始，逐步实现 CTC、流式解码、语言模型、WFST、音频前端、语义理解、量化和部署。

## 开始学习

先阅读 [唯一学习路径](LEARNING_PATH.md)。它按照能力依赖重新安排了阶段 0～12、预计时间、必修材料、阶段产出和离场门禁；不要直接按文件列表从头盲目运行。

需要安装 [uv](https://docs.astral.sh/uv/)。在项目根目录运行：

```powershell
uv sync --locked
uv run jupyter lab
```

如果很多代码还不会，先打开 [零基础逐行代码伴读](notebooks/代码伴读_零基础逐行理解ASR.ipynb)｜[已运行对照](notebooks/_executed/pytorch_foundations/代码伴读_零基础逐行理解ASR_已运行.ipynb)。它用 14 个可运行小节逐行解释变量、函数、shape、dtype/device、模型、训练循环、声学前端、CTC、流式 cache 和排错。然后按 [PyTorch 零基础 6 节课程索引](notebooks/PyTorch零基础课程索引.md) 学习，再完成新增的 [音频零基础 6 节桥梁课](notebooks/音频零基础课程索引.md)、[专业音频软件分析实验](AUDIO_SOFTWARE_GUIDE.md)和[24 条音频问题盲诊断](AUDIO_DIAGNOSIS_PRACTICE.md)，最后从 [Notebook 学习中心](notebooks/README.md) 进入 ASR 主线。桥梁课从振动、周期、RMS、dB、PCM、位深、通道、谐波、噪声与 SNR 讲到真实 WAV 审计，并使用 Audacity、Praat、Sonic Visualiser 或 Audition 把软件读数与代码结果交叉验证。

如果你的目标是从零学习 N-gram、OpenFst 与 ASR 语言模型，直接从 [9 课独立课程索引](notebooks/语言模型零基础_课程索引.md) 开始；这条路线不要求先学完声学模型课程，并带有可修改实验、交互控件、真实 OpenFst/KenLM 命令、失败对照和自动判题。运行第 3 课前按 [安装与排错指南](ASR_LM_ENVIRONMENT.md) 配好 WSL 工具链，实验时可查 [OpenFst/KenLM 速查表](ASR_LM_OPENFST_KENLM_CHEATSHEET.md)。

完成基础与综合项目后，打开 [09：前沿 ASR 语言模型系统设计实验室](notebooks/语言模型零基础_09_前沿ASR语言模型系统设计实验室.ipynb)，并配合 [ASR 语言模型前沿论文与系统路线](FRONTIER_ASR_LM_READING.md)，进入截至 2026-08-18 的 SpeechLLM、音频条件纠错、检索式 contextual ASR 和幻觉检测研究。

每课提供两份 Notebook：

- `NN_课程名.ipynb`：学习版本，建议自己逐格执行；
- `notebooks/_executed/<课程类别>/*_已运行.ipynb`：经过验证的输出对照版本；不要直接编辑。

## 课程路线

| 阶段 | 课程 | 核心内容 |
|---|---:|---|
| 逐行代码伴读 | 14 个小节 | Python、shape、模型、训练、CTC、cache、排错 |
| PyTorch 零基础 | 导学＋基础 1～6 | Python、Tensor、广播、autograd、nn.Module、DataLoader |
| 音频零基础桥梁 | 音频基础 1～6 | 振动、波形、周期、RMS、dB、PCM、位深、通道、谐波、SNR、WAV 审计 |
| 专业音频软件实验 | 1 套跨软件实验 | Audacity 波形/频谱、Praat F0/Formant/TextGrid、Sonic Visualiser、Audition/Cool Edit 对照 |
| 音频问题盲诊断 | 24 条盲题＋2 条正常对照 | 幅值、削波、DC、噪声、滤波、丢失、回声、采样率、多通道与混叠诊断 |
| 声音与特征 | 1～6 | 采样、dB、FFT、STFT、Mel、Log-Mel |
| 张量与编码器 | 7～9 | Padding、Mask、PyTorch、Conv1d、下采样 |
| CTC | 10～14 | blank、动态规划、CTCLoss、Prefix Beam、真实音频训练 |
| 流式 ASR | 15～18 | Chunk、缓存、因果编码器、PGS、RTF 与延迟 |
| LM 与 WFST | 19～23 | N-gram、热词、FST、HCLG、Lattice |
| 综合系统 | 24 | 流式 CTC 系统接口与验收 |
| 量化部署 | 25～30 | ONNX、INT8、HTTP、WebSocket、Docker、监控 |
| 音频前端 | 31～36 | PCM、NS、AGC、VAD、AEC、Beamforming |
| 后处理语义 | 37～41 | 时间戳、Diarization、ITN、置信度、NLU、LLM |
| 前沿模型与真实项目 | 42～46 | Conformer、RNN-T/TDT、自监督、AudioEncoder＋LLM、Qwen3-ASR 验收 |

CTC 是课程主轴。完成第 10～13 课后，使用 [CTC 可视化实验室](notebooks/专题_CTC可视化实验室_从路径到流式解码.ipynb) 在同一概率矩阵上贯通路径穷举、前向动态规划、`CTCLoss`、Prefix Beam 和跨 chunk 状态。

## 更高效的学习方式

每课按以下顺序完成：

1. 不运行代码，完成课前诊断和“知识接力”闭卷回忆。
2. 阅读一小节，在运行前预测图形、shape 或数值。
3. 每次只修改一个参数，记录“预测—结果—解释”。
4. 完成分层强化题，达到课程门槛再进入下一课；音频桥梁和主线按 19/24 执行。
5. 从空白 cell 重写本课核心函数。
6. 在第 1、7、30 天闭卷复习。

使用 [LEARNING_LOG.md](LEARNING_LOG.md) 记录错题和实验结论。课程中的参考答案用于核对，不建议第一次学习时提前展开。

学习中枢会根据最早的知识断点推荐起点，并用 0～4 级证据量表区分“看懂”“能解释”“能实现”和“能迁移”。个人结果可保存为本地 `learning_progress.json`，该文件默认不会提交到 Git。

## 项目结构

```text
notebooks/    唯一学习入口与无输出学习版；运行对照分类放在 notebooks/_executed/
acoustic_engine/ 可训练的离线/流式 CTC 结课项目、Prefix Beam 与 Bigram LM
learning_workspace/ 学习者亲手编码区与隐藏样例关卡入口
tests/        声学前端、模型、解码、流式一致性和 checkpoint 回归测试
data/         开源教学音频和数据说明
deployment/   ONNX Runtime + FastAPI/WebSocket 教学服务
artifacts/    小型 FP32/INT8 ONNX 教学模型
scripts/      音频生成、课程生成、升级与校验脚本
outputs/      早期课程生成的图像
```

## 验证

快速检查课程结构、Notebook JSON、执行错误和必需文件：

```powershell
uv run python scripts/validate_lm_course.py
uv run python scripts/validate_course.py
uv run python scripts/execute_course_labs.py
uv run python -m unittest discover -s tests -p "test_acoustic_engine.py"
```

前两条检查 LM 与整套课程的目录、源/运行对照和必需产物；第三条在内存中执行 8 个不依赖外部下载的核心实验。准备好 `.local_data` 后，可用 `uv run python scripts/execute_course_labs.py --external` 连同 FSDD 与 AudioMNIST 三个外部数据实验一起执行。

启动教学部署服务：

```powershell
uv run uvicorn deployment.app:app --host 127.0.0.1 --port 8000
```

接口包括 `GET /health`、`POST /infer` 和 `WS /stream`。

## 重要边界

- 第 14 课是小数据过拟合实验，不代表泛化性能。
- 第 24 课使用可控后验验证流式系统逻辑，不是假装成成熟声学模型。
- 第 25～30 课部署的是确定性随机权重教学模型，用于验证导出、cache、量化和服务协议。
- 第 37 课是说话人 embedding 接口演示，不是生产 diarization 模型。
- 第 38～41 课强调保留证据文本，语义模型不得凭空修改数字、姓名和否定词。
- 第 42～45 课使用 CPU 可运行的小模型解释前沿架构，不代表复现工业规模预训练。
- 第 46 课给出 Qwen3-ASR 推理、微调和验收流程；没有下载权重、准备合规数据并真实运行时，不宣称完成训练。
- `acoustic_engine/` 的数字语音训练集很小，目标是打通工程闭环，不能把训练集过拟合当成泛化能力。

更详细说明见 [COURSE_AUDIT.md](COURSE_AUDIT.md)。

## 数据与许可

教学录音来自 Free Spoken Digit Dataset，详情和署名见 [DATA_SOURCES.md](DATA_SOURCES.md)。上游数据采用 CC BY-SA 4.0。

本仓库采用完全公开的双许可：原创软件和 Notebook 代码单元使用 Apache-2.0；原创课程讲义、练习和图表说明使用 CC BY 4.0。FSDD 音频及其派生数据继续遵守上游 CC BY-SA 4.0。精确边界见 [LICENSE-SCOPE.md](LICENSE-SCOPE.md)，完整文本见 [LICENSE](LICENSE) 和 [LICENSE-CONTENT](LICENSE-CONTENT)。

## GitHub

仓库包含轻量级 GitHub Actions 校验。推送前运行：

```powershell
uv run python scripts/validate_course.py
git status
```

协作规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。

第一次发布的完整命令见 [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md)。
