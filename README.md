# Learn ASR：从声音基础到流式 CTC、WFST、语义与部署

一套面向初学者的可运行 Jupyter Notebook 课程。课程使用真实开源语音，从波形与频谱开始，逐步实现 CTC、流式解码、语言模型、WFST、音频前端、语义理解、量化和部署。

## 开始学习

需要安装 [uv](https://docs.astral.sh/uv/)。在项目根目录运行：

```powershell
uv sync --locked
uv run jupyter lab
```

然后先打开 [学习中枢：诊断与掌握度仪表盘](notebooks/学习中枢_诊断与掌握度仪表盘.ipynb)，完成 24 题闭卷诊断并生成八阶段掌握度图；再按照结果进入 [第 1～41 课完整索引](notebooks/课程索引_第01到41课.md)。

每课提供两份 Notebook：

- `NN_课程名.ipynb`：学习版本，建议自己逐格执行；
- `NN_课程名_已运行.ipynb`：经过验证的输出对照版本。

## 课程路线

| 阶段 | 课程 | 核心内容 |
|---|---:|---|
| 声音与特征 | 1～6 | 采样、dB、FFT、STFT、Mel、Log-Mel |
| 张量与编码器 | 7～9 | Padding、Mask、PyTorch、Conv1d、下采样 |
| CTC | 10～14 | blank、动态规划、CTCLoss、Prefix Beam、真实音频训练 |
| 流式 ASR | 15～18 | Chunk、缓存、因果编码器、PGS、RTF 与延迟 |
| LM 与 WFST | 19～23 | N-gram、热词、FST、HCLG、Lattice |
| 综合系统 | 24 | 流式 CTC 系统接口与验收 |
| 量化部署 | 25～30 | ONNX、INT8、HTTP、WebSocket、Docker、监控 |
| 音频前端 | 31～36 | PCM、NS、AGC、VAD、AEC、Beamforming |
| 后处理语义 | 37～41 | 时间戳、Diarization、ITN、置信度、NLU、LLM |

## 更高效的学习方式

每课按以下顺序完成：

1. 不运行代码，完成 3 题课前诊断。
2. 阅读一小节，在运行前预测图形、shape 或数值。
3. 每次只修改一个参数，记录“预测—结果—解释”。
4. 完成 12 道分层强化题，达到 19/24 分再进入下一课。
5. 从空白 cell 重写本课核心函数。
6. 在第 1、7、30 天闭卷复习。

使用 [LEARNING_LOG.md](LEARNING_LOG.md) 记录错题和实验结论。课程中的参考答案用于核对，不建议第一次学习时提前展开。

学习中枢会根据最早的知识断点推荐起点，并用 0～4 级证据量表区分“看懂”“能解释”“能实现”和“能迁移”。个人结果可保存为本地 `learning_progress.json`，该文件默认不会提交到 Git。

## 项目结构

```text
notebooks/    41 课源 Notebook 与已运行版本
data/         开源教学音频和数据说明
deployment/   ONNX Runtime + FastAPI/WebSocket 教学服务
artifacts/    小型 FP32/INT8 ONNX 教学模型
scripts/      音频生成、课程生成、升级与校验脚本
outputs/      早期课程生成的图像
```

## 验证

快速检查课程结构、Notebook JSON、执行错误和必需文件：

```powershell
uv run python scripts/validate_course.py
```

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
