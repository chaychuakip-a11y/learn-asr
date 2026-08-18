# 音频问题盲诊断题库：先看证据，再核对答案

这套练习包含 1 条干净参考语音和 24 条盲命名音频。文件名不会泄露问题；部分样本没有注入明显故障。目标是形成固定诊断顺序，而不是凭第一耳朵猜答案。

## 1. 文件与命令

目录：`data/audio_diagnosis_lab/`

- `reference_clean_speech.wav`：干净参考，只用于对照；
- `case_01.wav` ～ `case_24.wav`：盲诊断题；
- `manifest.json`：公开题目清单，不含答案；
- `answer_key.json`：机器自测使用，第一次不要直接打开。

重新生成：

```powershell
uv run python scripts/build_audio_diagnosis_practice.py
```

查看分级题号，不显示答案：

```powershell
uv run python scripts/audio_diagnosis_quiz.py --list
```

查看所有允许的主诊断类别：

```powershell
uv run python scripts/audio_diagnosis_quiz.py --categories
```

## 2. 每一题固定走七步

不要先降噪、归一化或转单声道。先记录原始证据。

### 第 1 步：元数据

记录：

- sample rate；
- channel count；
- duration；
- sample format/bit depth；
- 与 `reference_clean_speech.wav` 是否一致。

如果采样值看起来相同，但采样率和时长不同，优先怀疑时间轴或元数据，不要先调音量。

### 第 2 步：试听，但不立即下结论

只记录听感，例如：

- 太小、刺耳、发闷、发薄；
- 嗡声、嘶声、固定啸叫；
- 爆音、间断、重复、拖尾；
- 单边、空间异常、下混后消失；
- 语速或音高整体错误。

听感是线索，不是最终证据。

### 第 3 步：波形

在 Audacity 中先用 Linear/Normalized Values，再用 dB 纵轴。检查：

- peak 与 RMS 是否过高或过低；
- 波形是否以 0 为中心；
- 是否出现平顶、单侧平顶或密集饱和；
- 是否有短脉冲、矩形零区间或突然截断；
- 振幅包络是否周期性起伏；
- 放大到采样点后是否只剩少量幅值台阶。

### 第 4 步：通道

立体声必须分开看左右通道：

- 两边是否都有信号；
- 是否完全相同；
- 是否一正一负；
- 事件是否错开若干采样点；
- 转单声道前后是否发生抵消或梳状滤波。

不要只看合并后的一个 RMS。

### 第 5 步：单段频谱

使用 `Analyze > Plot Spectrum`，固定 Hann window，并记录 FFT size。检查：

- 50/60 Hz 及整数倍附近是否有窄峰；
- 是否存在持续的单一高频峰；
- 高频或低频是否整体缺失；
- 削波/非线性是否增加谐波；
- 噪声是宽带抬升还是窄带干扰。

### 第 6 步：声谱图

波形 + Spectrogram 同屏观察：

- 水平细线：稳定频率干扰；
- 垂直细线：短时冲击；
- 整片背景变亮：宽带噪声底升高；
- 语音事件后持续拖尾：回声或混响；
- 高频/低频区域长期变暗：滤波问题；
- 不自然的镜像/折返内容：重采样混叠候选。

### 第 7 步：写唯一主诊断

先写报告再查答案：

```text
case_id:
primary diagnosis:
confidence: low / medium / high
metadata evidence:
waveform evidence:
channel evidence:
spectrum/spectrogram evidence:
audible evidence:
alternative hypothesis:
effect on ASR:
next non-destructive test:
```

如果只能写“听起来怪”，本题尚未完成。

## 3. 检查答案

猜测支持英文类别和常见中文别名：

```powershell
uv run python scripts/audio_diagnosis_quiz.py --check 07 clipping
uv run python scripts/audio_diagnosis_quiz.py --check 12 采样率错误
uv run python scripts/audio_diagnosis_quiz.py --check case_18 正常
```

答错时只给下一项检查动作，不直接泄露答案。需要提示：

```powershell
uv run python scripts/audio_diagnosis_quiz.py --hint 07
```

完成书面证据后再揭晓：

```powershell
uv run python scripts/audio_diagnosis_quiz.py --reveal 07
```

揭晓后还要做两件事：

1. 回到软件中指出答案所说的具体时间、频率、幅值或通道证据；
2. 用一句话说明它与最相似问题的区别，例如“回声是一条明显延迟副本，混响是多个分散衰减尾巴”。

## 4. 题库覆盖范围

题库会从以下问题中抽取并固定打乱：

| 类别 | 可能问题 |
|---|---|
| 幅值与非线性 | 幅值过低、硬削波、不对称削波、重度非线性压缩、周期性增益抽吸 |
| 基线与噪声 | DC offset、50 Hz 工频嗡声、宽带白噪声、固定高频啸叫 |
| 时间连续性 | clicks、dropouts、单次回声、混响、开头/结尾截断 |
| 频率响应 | 过度低通、过度高通、低位深量化、错误重采样混叠 |
| 元数据 | 采样率标签错误导致时长和音高一起变化 |
| 多通道 | 单边无声、左右反相、通道延迟、正常立体声对照 |
| 对照 | 正常单声道、正常立体声；不允许强行找故障 |

## 5. 分级练习方法

### Beginner

先完成 `--list` 中的 beginner 题。每题最多只用：元数据、波形、通道和试听。目标是识别明显幅值、削波、DC、丢失、采样率与单边通道问题。

通关：至少 6/7 正确；每题能指出一个数值或具体图形证据。

### Intermediate

加入 Plot Spectrum 和 Spectrogram。目标是区分：

- 宽带噪声与窄带干扰；
- 50 Hz 嗡声与高频啸叫；
- 低通与高通；
- 脉冲与丢失；
- 削波与量化台阶。

通关：至少 7/9 正确；FFT/window 设置必须记录。

### Advanced

必须比较参考音频，并使用相关性、crest factor、时间延迟或下混实验。目标是区分：

- 单次回声与混响；
- 通道反相与通道延迟；
- 硬削波与软饱和/重压缩；
- 量化失真与重采样混叠；
- 真故障与正常立体声。

通关：至少 6/8 正确；所有修改只对副本进行。

## 6. 常见误判

- **幅值小 ≠ 静音**：低增益语音仍有结构，单个采样点低于阈值也不能判静音。
- **0 dBFS ≠ 没有声音**：0 dBFS 是数字满幅上限；零幅值对应负无穷 dBFS。
- **peak 正常 ≠ 没削波**：削波后再降低音量，平顶仍在但 peak 已低于 0 dBFS。
- **声谱图更亮 ≠ 样本变大**：gain/range/colour map 可能只改变显示。
- **高频少 ≠ 一定低通故障**：说话内容、说话人和麦克风本来就影响频谱，必须与参考和时间段比较。
- **立体声两边都有声 ≠ 通道健康**：反相和毫秒级延迟只有对齐、相关或下混时才明显。
- **降噪后更安静 ≠ 更适合 ASR**：残差中出现可辨识语音说明算法正在删除目标信号。

## 7. 完成后的迁移任务

从你自己的合法录音中选 3 条，不查看来源描述，按同一模板分析。任何诊断都必须同时给出：

1. 至少两个互相独立的证据；
2. 一个替代解释；
3. 一个不会修改原文件的下一步验证；
4. 对 ASR 可能造成的影响；
5. 是否需要修复，还是只需在输入审计中拒绝。

真正的能力不是记住 24 个答案，而是面对新文件时仍能按相同顺序收集证据。
