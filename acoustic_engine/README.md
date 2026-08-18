# 结课项目：从 WAV 到数字文本的最小声学引擎

这个目录把现有教程串成一个真实、可训练、可保存和可推理的 CTC 闭环：

```text
WAV → Log-Mel [T,F] → Conv + BiGRU → logits [T,C] → CTC collapse → 数字文本
```

它使用仓库内 10 个开源数字片段合成 22 条训练语音。目标是验证数据、shape、梯度、CTC 和推理接口，不是报告泛化准确率。项目同时包含双向 GRU 离线基线、残差因果卷积流式模型、Prefix Beam、Add-k Bigram LM 与热词接口，用来比较状态、质量和延迟边界。

## 先验证，再训练

```powershell
uv run python -m unittest discover -s tests -p "test_acoustic_engine.py"
uv run python -m acoustic_engine.tutor
uv run python -m acoustic_engine.tutor --status
uv run python -m acoustic_engine.challenge --list
uv run python -m acoustic_engine.challenge --check sampling
uv run python -m acoustic_engine.mastery --status
uv run python -m acoustic_engine.mastery --system-audit
uv run python -m acoustic_engine.train_digits --epochs 200
uv run python -m acoustic_engine.recognize artifacts/tiny_digit_ctc.pt data/spoken_digits_parts/7_jackson_0.wav
uv run python -m acoustic_engine.recognize artifacts/tiny_digit_ctc.pt data/spoken_digits_parts/7_jackson_0.wav --decoder prefix_beam --beam-size 10
uv run python -m acoustic_engine.train_streaming_digits --epochs 500
uv run python -m acoustic_engine.recognize_stream artifacts/streaming_digit_ctc.pt data/spoken_digits_parts/7_jackson_0.wav --chunk-samples 800
uv run uvicorn acoustic_engine.api:app --host 127.0.0.1 --port 8000
uv run python -m acoustic_engine.benchmark artifacts/streaming_digit_ctc.pt data/spoken_digits_parts/0_jackson_0.wav data/spoken_digits_parts/1_jackson_0.wav
```

第一次学习时不要一口气读完全部源码。按下面的证据关卡进行。

## 关卡 1：声音进入电脑

- 前置教程：第 1～2 课。
- 只读：`features.py` 中的 `load_mono_audio`。
- 先预测：8 kHz、0.5 秒语音有多少采样点？双声道混为单声道后 shape 怎样变化？
- 通关证据：不看答案写出 `samples = sample_rate * seconds`，并能解释“每个采样值是某时刻的振幅”。
- 接口约束：训练和推理必须共用同一套幅度处理。离线基线由 `LogMelFrontend` 执行整句峰值归一化；流式模型明确关闭整句归一化，训练和推理都使用固定 PCM 标尺及同一组特征均值/标准差，因为收到开头时还不知道未来峰值。

## 关卡 2：波形变成 Log-Mel

- 前置教程：第 3～6 课。
- 只读：`LogMelFrontend.__call__`。
- 追踪 shape：`[samples] → [frequency,time] → [mel,time] → [time,mel]`。
- 实验：把 `hop_length` 从 80 改为 160，运行测试并解释帧数为什么约减半。
- 通关证据：从空白实现一个简化 STFT，并说清 `n_fft`、`win_length`、`hop_length` 和 `n_mels` 的区别。

## 关卡 3：声学编码器

- 前置教程：第 7～9 课。
- 只读：`model.py`。
- 先预测：输入 `[B,T,24]` 经过 `Conv1d` 前为什么要交换为 `[B,24,T]`？
- 通关证据：闭卷画出每一层 shape，并让测试中的错误 feature 维度触发可读报错。
- 排错案例：双向 GRU 若忽略真实 `lengths`，反向状态会读取尾部 padding；必须验证同一语音单独推理与放入变长 batch 的有效帧输出一致。

## 关卡 4：CTC 训练与解码

- 前置教程：第 10～14 课和 CTC 可视化实验室。
- 只读：`decoder.py`、`train_digits.py`。
- 手算：`[1,1,0,1,2,2,0]` 经过 CTC 合并后为何保留两个类别 1？
- 通关证据：闭卷实现 `ctc_collapse`，解释 blank、重复字符和 `logits [T,B,C]` 的要求。

## 关卡 5：完整引擎契约

- 只读：`engine.py`、`recognize.py`。
- 通关证据：能够保存、重新加载并识别同一 WAV；逐项解释 checkpoint 中模型参数、词表、前端参数、均值和标准差为什么必须版本一致。

## 关卡 6：Prefix Beam、语言模型与热词

- 前置教程：第 13、19、20 课。
- 只读：`decoder.py` 中的 `prefix_beam_search` 和 `language_model.py`。
- 先手算：为什么一条概率最大的 CTC 路径不一定对应总概率最大的文本？
- 实验：准备每行一个数字串的 UTF-8 文本文件，用 `--lm-corpus`、`--lm-weight`、`--hotword` 和 `--hotword-bonus` 比较候选变化。
- 通关证据：用全部路径穷举验证一个小概率矩阵的 beam 结果，并解释 LM 分数为何只能在 prefix 真正增加 token 时加入。

## 毕业标准

只有同时满足以下条件才算“会了”：

1. 不看源码重写最小版本；
2. 对一段新 WAV 打印并解释每一层 shape；
3. 故意制造采样率、feature 维度和 CTC length 错误并定位；
4. 明确区分训练集过拟合、独立测试集效果和生产能力；
5. 比较离线双向模型与因果流式模型，并重新测质量、跨 chunk 一致性和延迟。

## 关卡 7：真正跨 chunk 的流式状态

- 前置教程：第 15～18、24 课。
- 只读：`StreamingLogMelFrontend`、`CausalConvCTCAcousticModel.forward_chunk`（并与 GRU 版本比较）和 `streaming.py`。
- 三类状态必须分开：STFT 重叠采样、每层因果卷积的左上下文 cache、CTC 上一帧类别或 Prefix Beam。
- 实验：分别使用 400、800、1600 个采样点的 chunk，最终文字应一致；中间 partial 的出现时间可以不同。
- 通关证据：解释为什么 utterance 级峰值归一化不能直接用于实时输入，并证明整段模型 logits 与逐 chunk 拼接 logits 在容差内一致。
- 失败对照：单向 GRU 在这组极小数据上容易只记住第一段；不能用 LM 强行补第二个数字。最终教学 checkpoint 使用有限左上下文的因果卷积。

## 关卡 8：HTTP 与 WebSocket 会话

- 前置教程：第 28～30 课。
- `POST /recognize` 接收完整浮点采样数组；`WS /stream` 为每个连接独立保存前端、卷积 cache 和 CTC 状态。
- 先解释：为什么两个客户端绝不能共享同一个 `StreamingSession`？断开连接后哪些状态必须释放？
- 通关证据：同时打开两个 WebSocket，交错发送不同音频，最终文本仍分别正确；错误采样率必须明确拒绝而不是静默识别。
- 边界：JSON 浮点数组便于教学检查，但带宽效率低；生产接口通常传 PCM/Opus 二进制帧并包含版本、鉴权、限流与监控。

## 关卡 9：RTF、延迟与验收

- 前置教程：第 18、30 课。
- 用 `benchmark.py` 固定模型、音频、线程、chunk 和重复次数，同时报告 mean、P50、P95 与 RTF。
- 通关证据：解释“整段处理时间”“首 token 延迟”“chunk 算法延迟”不是同一个指标，并比较至少三种 chunk 大小。
- 边界：教学电脑上的小样本耗时不能外推为生产容量；必须在目标硬件、真实并发和长短音频分桶上重测。
