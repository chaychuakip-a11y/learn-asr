# 专业音频软件分析实战：Audacity、Praat、Sonic Visualiser 与 Audition

这是一条 GUI 软件实践课。目标不是学会“修音”，而是把软件窗口中的波形、频谱和统计量，与课程里的采样率、PCM、RMS、dBFS、FFT、STFT、SNR、基频和共振峰逐项对应。

## 1. 软件怎么选

| 软件 | 本课程定位 | 最适合的任务 | 是否必修 |
|---|---|---|---|
| [Audacity](https://www.audacityteam.org/download/) | 通用波形编辑和频谱分析主工具 | 波形、通道、采样率、频谱、剪切、DC、噪声、导出 | 必修 |
| [Praat](https://praat.org/download_win.html) | 专业语音学分析主工具 | F0/Pitch、Formant、Intensity、Spectrogram、TextGrid | 必修 |
| [Sonic Visualiser](https://www.sonicvisualiser.org/download.html) | 精细频谱和多层标注 | 多时间尺度频谱、频谱切片、插件测量和叠加标注 | 选修 |
| [Adobe Audition](https://www.adobe.com/products/audition.html) | 商业级编辑、修复和频谱工作流 | Amplitude Statistics、Frequency Analysis、Spectral Editing、多轨 | 选修/有许可时 |
| Cool Edit / Cool Edit Pro | 旧版工作流参考 | 已有合法安装时可完成波形、频谱和统计实验 | 不作为主线环境 |

### 为什么不把 Cool Edit 作为必装软件

Cool Edit 的核心概念仍然有用，但旧版本、旧教程和第三方下载包的版本差异很大。本课程不要求从非官方站点下载安装包。已有合法安装可以对照使用；新学习环境优先用 Audacity，想走商业工作流时用 Adobe Audition。

本课程只依赖软件的稳定概念，不依赖界面颜色：

```text
波形编辑器       → 时间 × 幅度
频谱图           → 频率 × 幅度/功率
声谱图           → 时间 × 频率，颜色表示幅度/功率
幅度统计         → peak / RMS / DC / clipped samples
语音学分析       → pitch / formant / intensity / annotation
```

## 2. 安全规则：分析之前不要破坏证据

1. 永远保留原始 WAV，只对副本做降噪、归一化、重采样和裁剪。
2. 第一次打开文件先记录元数据和原始统计量，随后才应用效果。
3. “显示更亮”不等于能量变大；声谱图颜色范围可以只改变显示。
4. “听起来更干净”不等于 ASR 更准；增强前后都要保留文件并测 CER/WER。
5. dB 必须记录参考：dBFS、Praat intensity、频谱相对 dB 不能直接当成同一个量。
6. 立体声转单声道前要说明混合规则；不要用 `flatten` 把两个通道首尾拼接。
7. Audition 的 Waveform Editor 和许多 Audacity 离线效果会改写音频数据；先复制轨道或另存副本。

## 3. 准备统一实验音频

仓库已经生成一组确定性音频：

```powershell
uv run python scripts/build_audio_software_lab_assets.py
```

文件位于 `data/audio_software_lab/`：

| 文件 | 已知事实 | 用来验证什么 |
|---|---|---|
| `01_calibration_440hz_peak_minus12dbfs.wav` | 440 Hz，peak -12 dBFS | 频率峰值、peak 与 sine RMS |
| `02_two_tones_440hz_1000hz.wav` | 440 Hz + 1000 Hz | FFT size、窗函数、双峰 |
| `03_dc_offset_220hz.wav` | DC 约 +0.12 | 波形中心偏移、0 Hz 能量 |
| `04_hard_clipped_440hz.wav` | 硬削波 | 平顶、clipped samples、额外谐波 |
| `05_noise_profile_then_speech_snr10db.wav` | 前 0.5 秒纯噪声，后段 10 dB SNR 语音 | 噪声画像、残差信号、降噪副作用 |
| `06_stereo_right_delayed_1ms.wav` | 右声道延迟 16 点 = 1 ms | 通道、时间差、对齐 |
| `07_real_speech_digit_zero_8khz.wav` | 真实数字语音 | Praat 的 F0、Formant、Intensity、TextGrid |

精确生成参数和期望统计量在 `data/audio_software_lab/manifest.json`。不同软件因窗口、归一化、声道合并和 dB 定义不同，读数允许小幅差异；如果差异很大，先查设置，不要直接判定某个软件错误。

---

# 实验 A：Audacity 建立专业分析习惯

## A1. 打开文件但先不处理

打开 `01_calibration_440hz_peak_minus12dbfs.wav`，记录：

- 文件名与是否为副本；
- 采样率：16,000 Hz；
- 通道：mono；
- 时长：1.000 秒；
- 项目内部格式与导出文件位深不要混为一谈。

放大到能看见单个波形周期。用选区工具选择 10 ms，并回答：440 Hz 在 10 ms 内大约有多少个周期？预测后再数图。

## A2. 同屏显示波形和声谱图

将轨道切换到 **Multi-view / Waveform + Spectrogram**。建议第一次固定：

- 最低频率：0 Hz；
- 最高频率：8,000 Hz，因为采样率为 16 kHz；
- 频率尺度：Linear；
- Window：Hann；
- Window size：先用 1024；
- Dynamic range：约 80 dB。

要区分三件事：

1. 波形横轴是时间，纵轴是幅度；
2. 声谱图横轴是时间，纵轴是频率；
3. 改变颜色范围只改变显示，不改变 WAV。

## A3. 用 Plot Spectrum 找频率峰

全选音频，打开 **Analyze > Plot Spectrum**：

1. Algorithm 选择 Spectrum；
2. Window 选择 Hann；
3. Size 依次使用 1024、4096、8192；
4. 找最强峰，应在 440 Hz 附近；
5. 记录每次频率分辨率约为 `sample_rate / FFT_size`；
6. 导出一次 spectrum 文本，观察第一列频率、第二列 dB。

注意：更大的 FFT size 让频率间隔更细，但需要更长的观察窗；它不会创造原信号中不存在的信息。

## A4. 两个频率和窗函数实验

打开 `02_two_tones_440hz_1000hz.wav`：

- 找到约 440 Hz 与 1000 Hz 的两个峰；
- 比较 Hann 与 Rectangular window 的旁瓣/泄漏；
- 只选择 20 ms 与选择完整 1 s，比较频谱稳定性；
- 记录“观察窗长度”和“FFT size”各自改变了什么。

通关标准：不用菜单名也能解释为什么频谱峰不是无限细的竖线。

## A5. DC、剪切和谐波

分别打开文件 03、04：

### DC offset

- `03_dc_offset_220hz.wav` 的波形中心不在 0；
- 频谱中除 220 Hz 外，0 Hz 附近会有能量；
- 只在副本上尝试 Normalize 的 “Remove DC offset”；
- 前后记录 mean/中心位置，确认不是凭肉眼说“修好了”。

### Hard clipping

- `04_hard_clipped_440hz.wav` 的顶部和底部呈平坦形状；
- 频谱会出现原纯正弦没有的额外谐波；
- 降低音量只能把削平后的波形整体缩小，不能恢复丢失的形状。

通关标准：能解释“peak 没超过 0 dBFS”为什么不能证明音频从未削波。

## A6. 噪声画像与降噪副作用

打开 `05_noise_profile_then_speech_snr10db.wav`：

1. 选中前 0.5 秒纯噪声；
2. 打开 **Effect > Noise Removal and Repair > Noise Reduction**；
3. 获取 Noise Profile；
4. 复制轨道，原轨静音保留；
5. 在副本上选择全段，先用较小 Noise Reduction；
6. Preview 时切到 Residue，检查残差中是否出现可辨认语音；
7. 若残差含语音，说明参数正在伤害目标信号；
8. 比较处理前后波形、声谱图、听感和后续 Log-Mel/CER。

结论必须包含边界：该方法适合相对稳定的背景噪声，不适合把背景说话或变化很大的环境声“无损删除”。

---

# 实验 B：Praat 做语音学分析

使用 `07_real_speech_digit_zero_8khz.wav`。

## B1. 打开并进入 Sound Editor

1. 在 Praat Objects 窗口读取 WAV；
2. 选中 Sound 对象，点击 **View & Edit**；
3. 放大到一个稳定浊音片段；
4. 同时显示 Spectrogram、Pitch、Formant、Intensity。

如果曲线过密或明显跳倍/减半，不要立刻相信自动结果。先检查选择区间、噪声、pitch floor/top 和是否为浊音。

## B2. 读出 F0/Pitch

1. 在稳定浊音处放置光标；
2. 记录当前 Pitch/F0，单位 Hz；
3. 选择一小段稳定区间，记录 mean 或 median F0；
4. 修改 pitch floor/top，再看是否出现倍频或半频错误；
5. 解释 F0 与频谱中谐波间隔的关系。

注意：pitch range 是算法约束，不是说话人的标签。范围设置不合适会让曲线看起来很平滑却是错误的。

## B3. 读出 Formant

1. 在元音较稳定的位置读取 F1、F2、F3；
2. 把 Formant 点与声谱图中的水平能量带对照；
3. 改变最大 formant 或 formant 数量，观察结果敏感性；
4. 不要把 F0 与 F1 混淆：F0 来自声带周期，formant 是声道共振特征。

## B4. Intensity 与 dBFS 的边界

记录一段的 Intensity 曲线和均值，但必须在报告中写：

> 未经声压校准的普通 WAV，Praat intensity 不应被宣称为真实环境声压级 SPL。

它可以用于同一记录链中的相对比较，但不能直接与 Audacity 的 dBFS 当成同一个绝对量。

## B5. TextGrid 标注

创建 TextGrid，至少包含两层：

- `word`：整词区间；
- `phone_or_event`：音素或声学事件区间。

标注起点、终点和一个内部边界。保存为文本格式，并记录：

- 边界依据是波形、声谱图、听感还是三者共同证据；
- 边界不确定度大约多少毫秒；
- 改变声谱图窗口后边界判断是否变化。

---

# 实验 C：Sonic Visualiser 精细频谱（选修）

打开文件 02 或 07，添加 spectrogram layer：

1. 在同一时间轴叠加 waveform、spectrogram 和 time ruler；
2. 比较 linear、log 和 mel frequency scale；
3. 关闭图像 smoothing，观察真实 frequency bins；
4. 调整 window size、overlap 和 zero padding；
5. 增加一个 spectrum/slice layer，观察某一时刻的频率切片；
6. 添加时间点或区间标注，导出 annotation。

关键判断：图像插值更平滑不代表频率分析分辨率更高；频率尺度改变也不代表音频被重新采样。

---

# 实验 D：Adobe Audition / Cool Edit 风格工作流（选修）

## D1. Audition 的同类操作

1. 在 Waveform Editor 打开校准文件；
2. 显示 Waveform 与 Spectral Frequency Display；
3. 用 **Window > Amplitude Statistics** 执行 Scan/Scan Selection；
4. 记录 peak amplitude、RMS、DC offset、possibly clipped samples；
5. 用 Frequency Analysis 查 440 Hz/1000 Hz 峰；
6. 改变 spectral resolution，解释时间—频率分辨率权衡；
7. 只对副本使用 spectral selection 做修复，并保留处理前后文件。

Audition 的 Waveform Editor 是破坏性编辑环境，保存会改变文件；Multitrack 更适合保留非破坏性编排。分析课首先练习“Scan → 记录 → 复制 → 修改 → 再 Scan”。

## D2. 已有 Cool Edit 安装时的对应关系

不同 Cool Edit 版本菜单名会变化，按概念寻找：

| 任务 | Cool Edit 常见概念 | Audacity | Audition |
|---|---|---|---|
| 看幅度随时间 | Waveform/Edit View | Waveform | Waveform Editor |
| 看时间—频率 | Spectral View | Spectrogram/Multi-view | Spectral Frequency Display |
| 看单段频谱 | Frequency Analysis | Analyze > Plot Spectrum | Frequency Analysis |
| 看 peak/RMS/DC | Statistics | Measure RMS/相关分析器 | Amplitude Statistics |
| 改采样率 | Convert Sample Type | Resample/Export | Convert Sample Type |
| 稳态降噪 | Noise Profile/Noise Reduction | Noise Reduction | Noise Reduction/Restoration |

如果找不到完全相同菜单，以“输入是什么、输出是什么、是否改写样本”为准，不要照着旧截图机械点击。

---

# 跨软件统一记录表

每分析一个文件，复制下表到 `LEARNING_LOG.md`：

| 字段 | 结果 | 软件/设置 | 是否与 manifest/Notebook 一致 |
|---|---|---|---|
| 文件与副本状态 |  |  |  |
| sample rate / channels / duration |  |  |  |
| peak 与参考值 |  |  |  |
| RMS 与参考值 |  |  |  |
| DC mean/offset |  |  |  |
| clipped samples/ratio |  |  |  |
| dominant frequencies |  | FFT size / window |  |
| spectrogram window/hop/range |  |  |  |
| F0 range/median |  | Praat pitch settings |  |
| F1/F2/F3 |  | Praat formant settings |  |
| SNR 或噪声区间 |  | 定义与选择范围 |  |
| 处理动作 |  | 原始/副本路径 |  |
| 处理后副作用 |  | 听感/频谱/Log-Mel/CER |  |

## 最终通关门禁

- [ ] 能在 Audacity 中从零完成元数据、波形、声谱图和频谱分析。
- [ ] 能解释 FFT size、window、选区长度、频率尺度和颜色范围各改变什么。
- [ ] 能识别 DC offset、hard clipping、稳态噪声和声道延迟。
- [ ] 能在 Praat 中读取 F0、F1/F2/F3、Intensity，并说明可信边界。
- [ ] 能创建 TextGrid 并用波形、声谱图和听感共同标边界。
- [ ] 能对同一文件用 GUI 软件、`manifest.json` 和 Notebook 三方核对。
- [ ] 任何处理都保留原始证据，且不以“听起来更好”替代 ASR 指标。

完成基础操作后，进入 [24 条音频问题盲诊断题库](AUDIO_DIAGNOSIS_PRACTICE.md)。题库不会在文件名中暴露问题，并混入正常样本；必须先写元数据、波形、通道、频谱和声谱图证据，再用命令核对答案。

完成后回到主线第 1～6 课。你会发现采样、FFT、STFT 和 Mel 不再只是公式，而是软件界面里可以观察、改变并交叉验证的测量过程。
