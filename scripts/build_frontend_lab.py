"""Build the audio-frontend intensive notebook.

Run with:
    uv run python scripts/build_frontend_lab.py
"""

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "专题_音频前端实验室_质量VAD_AEC与波束形成.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md(
        """
        # 音频前端综合实验室：质量、VAD、AEC、波束形成与流式状态

        这不是一份只展示“漂亮波形”的 Notebook。你会用一条**真正的开源真人语音**，从麦克风输入一路走到声学特征入口，并回答五个工程问题：

        1. 输入到底有多响，`0 dBFS`、负 dB 和参考值分别是什么？
        2. 目标 `SNR` 如何精确构造，为什么它不同于音量？
        3. VAD 为什么需要帧、阈值、状态和 hangover？
        4. AEC 怎样用参考信号预测回声，`ERLE` 又衡量什么？
        5. 两只麦克风怎样形成空间选择性，流式 chunk 为什么必须带状态？

        **对应主线课程：第 31～36 课。** 建议用 2～3 次、每次 60～90 分钟完成，而不是一次看完。

        数据许可：Free Spoken Digit Dataset (FSDD)，CC BY-SA 4.0；项目中的拼接音频只用于教学。
        """
    ),
    md(
        """
        ## 学习地图与通过标准

        `PCM → 电平/失真检查 → DC 处理 → AEC → 降噪/AGC → VAD/Endpoint → 特征`

        完成本实验后，你应当能够：

        - 不把幅值、dBFS、SNR 和“人耳觉得响”混成同一个量；
        - 写出可跨任意 chunk 边界工作的有状态滤波器；
        - 从帧级 VAD 得到稳定端点，并解释误触发与截断；
        - 用 NLMS 演示回声路径估计，用 ERLE 验证效果；
        - 用 delay-and-sum 演示阵列增益，同时说出实验的简化条件；
        - 给前端优化设计离线、流式、ASR 质量三层验收。

        末尾有 **48 分测试**。建议达到 38 分，且所有自动断言通过。
        """
    ),
    code(
        """
        from pathlib import Path
        import math
        import sys

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        import soundfile as sf
        import librosa
        import librosa.display
        import ipywidgets as widgets
        from IPython.display import Audio, display

        # 优先使用系统中的中文字体；找不到时仍可运行，只是图中文字可能缺字。
        installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
        for font_name in ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS"]:
            if font_name in installed_fonts:
                plt.rcParams["font.sans-serif"] = [font_name]
                break
        plt.rcParams["axes.unicode_minus"] = False

        # 无论从项目根目录还是 notebooks/ 启动，都找到项目根目录。
        candidates = [Path.cwd(), Path.cwd().parent]
        ROOT = next(p for p in candidates if (p / "pyproject.toml").exists())
        AUDIO_PATH = ROOT / "data" / "spoken_digits_0_to_9_16k.wav"
        x, sr = sf.read(AUDIO_PATH, dtype="float64")
        if x.ndim == 2:
            x = x.mean(axis=1)

        print(f"音频: {AUDIO_PATH}")
        print(f"采样率: {sr} Hz；样本数: {len(x)}；时长: {len(x)/sr:.2f} s")
        display(Audio(x, rate=sr))
        """
    ),
    md(
        """
        ## 1. 先把四个容易混淆的量分开

        - **幅值 amplitude**：每个采样点的瞬时数值。本项目把 PCM 归一化到大约 `[-1, 1]`。
        - **Peak**：一段信号绝对幅值的最大值，擅长发现削波风险，但不代表持续能量。
        - **RMS**：平方、平均、开根号，描述一段信号的有效电平，比 peak 更接近“持续强度”。
        - **dB**：比值的对数表达，不是天然带单位的绝对刻度。

        对数字音频幅值，常用 `dBFS = 20 log10(amplitude / full_scale)`，参考幅值 `full_scale = 1`。

        因此 `0 dBFS` 是数字系统能表达的满幅参考，不是“没有声音”；正常语音通常是负数。`-6 dBFS` 的幅值约为满幅的 0.501，`-20 dBFS` 约为 0.1。功率比则使用 `10 log10(P/P_ref)`。

        > 关键：软件中的 `1000` 可能是 16-bit 整数 PCM 数值；这里的 `0.0305` 大约才对应它。必须先问“数据格式和参考值是什么”。
        """
    ),
    code(
        """
        EPS = 1e-12

        def rms(y):
            y = np.asarray(y, dtype=np.float64)
            return float(np.sqrt(np.mean(y * y) + EPS))

        def amp_to_dbfs(a):
            return 20.0 * np.log10(np.maximum(np.asarray(a), EPS))

        def level_report(y):
            y = np.asarray(y, dtype=np.float64)
            return {
                "peak": float(np.max(np.abs(y))),
                "peak_dBFS": float(amp_to_dbfs(np.max(np.abs(y)))),
                "RMS": rms(y),
                "RMS_dBFS": float(amp_to_dbfs(rms(y))),
                "DC_mean": float(np.mean(y)),
                "clipping_ratio_%": float(100 * np.mean(np.abs(y) >= 0.999)),
            }

        report = level_report(x)
        for k, v in report.items():
            print(f"{k:18s}: {v:.6f}")

        t = np.arange(len(x)) / sr
        fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
        ax[0].plot(t, x, lw=0.55)
        ax[0].axhline(1, color="r", ls="--", alpha=.6, label="0 dBFS")
        ax[0].axhline(-1, color="r", ls="--", alpha=.6)
        ax[0].set(xlabel="时间 (s)", ylabel="归一化幅值", title="整段波形：横轴是时间，纵轴是瞬时幅值")
        ax[0].legend()
        ax[1].hist(x, bins=100, color="tab:blue", alpha=.8)
        ax[1].set(xlabel="采样值", ylabel="样本数", title="幅值分布：是否偏离 0、是否挤在边界")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ### 读图训练 1（每题 2 分）

        1. 波形中某一点为 `0`，能否断言那一刻“没有声音”？为什么？
        2. 一段信号 peak 很高但 RMS 很低，可能是什么情形？
        3. `-6 dBFS` 相对 `0 dBFS` 的幅值是几倍？它是否等于声压级 `6 dB SPL`？

        <details><summary>参考答案</summary>

        1. 不能。振荡信号会反复穿过零点；“一段是否有声”需要看时间窗口内的能量/频谱。
        2. 可能只有很短的脉冲或爆音；peak 只捕捉最大点，RMS 会被长时间低能量稀释。
        3. 约 0.501 倍。dBFS 的参考是数字满幅，dB SPL 的参考是 20 微帕声压，不能直接等同。

        </details>
        """
    ),
    md(
        """
        ## 2. 交互破坏一条语音：增益、DC、噪声与削波

        下方四个滑块模拟四种不同问题。观察三张图，不要只听：

        - **波形**看时间位置、峰值、DC 偏移和削波平顶；
        - **频谱**看整段中各频率的总体能量，但看不到变化发生在什么时候；
        - **Log-Mel**同时保留粗略时间与频率结构，是 ASR 常用输入。

        每次拖动后再松手计算，避免长音频连续重绘卡顿。
        """
    ),
    code(
        """
        rng = np.random.default_rng(2026)
        fixed_noise = rng.standard_normal(len(x))

        def add_noise_at_snr(clean, noise, snr_db):
            clean = np.asarray(clean, dtype=np.float64)
            noise = np.asarray(noise, dtype=np.float64)
            scale = rms(clean) / (rms(noise) * 10 ** (snr_db / 20))
            return clean + scale * noise, scale * noise

        def show_corruption(gain_db=0, dc_offset=0.0, snr_db=30, clip_level=1.0):
            gained = x * 10 ** (gain_db / 20) + dc_offset
            noisy, _ = add_noise_at_snr(gained, fixed_noise, snr_db)
            y = np.clip(noisy, -clip_level, clip_level)
            spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
            freqs = np.fft.rfftfreq(len(y), 1/sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160,
                                                win_length=400, n_mels=64, power=2)
            logmel = librosa.power_to_db(mel, ref=np.max)

            fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
            ax[0].plot(np.arange(len(y))/sr, y, lw=.45)
            ax[0].set(xlabel="时间 (s)", ylabel="幅值", title="波形")
            ax[0].set_ylim(-1.1, 1.1)
            ax[1].plot(freqs, amp_to_dbfs(spec / max(spec.max(), EPS)), lw=.7)
            ax[1].set(xlim=(0, 8000), ylim=(-100, 2), xlabel="频率 (Hz)", ylabel="相对峰值 (dB)", title="整段频谱")
            librosa.display.specshow(logmel, sr=sr, hop_length=160, x_axis="time", y_axis="mel", ax=ax[2])
            ax[2].set_title("Log-Mel：亮色代表相对能量较强")
            plt.tight_layout(); plt.show()
            print(level_report(y))

        corruption_ui = widgets.interactive(
            show_corruption,
            gain_db=widgets.FloatSlider(min=-24, max=18, step=3, value=0, description="增益 dB", continuous_update=False),
            dc_offset=widgets.FloatSlider(min=-.3, max=.3, step=.05, value=0, description="DC", continuous_update=False),
            snr_db=widgets.FloatSlider(min=-5, max=40, step=5, value=30, description="SNR dB", continuous_update=False),
            clip_level=widgets.FloatSlider(min=.1, max=1, step=.1, value=1, description="削波阈值", continuous_update=False),
        )
        display(corruption_ui)
        """
    ),
    md(
        """
        ### 实验任务

        1. 固定其他项，把增益从 `0 dB` 调到 `+6 dB`。未削波前幅值理论上乘多少？
        2. 把 DC 调到 `0.2`。波形和频谱哪里发生明显变化？
        3. 把削波阈值降到 `0.2`。波形顶部有什么形状？频谱为什么会出现额外高频？
        4. 把 SNR 从 `30 dB` 降到 `0 dB`。Log-Mel 的暗区为什么被“铺亮”？

        <details><summary>提示与答案</summary>

        `+6 dB` 幅值约乘 1.995；DC 使波形中心上移，并在 0 Hz 附近增加能量；硬削波产生平顶和非线性谐波；低 SNR 时噪声底上升，原本的低能量时频区域也出现能量。

        </details>
        """
    ),
    md(
        """
        ## 3. SNR：信号相对噪声，而不是相对满幅

        `SNR_dB = 10 log10(P_signal/P_noise) = 20 log10(RMS_signal/RMS_noise)`。

        它的参考不是固定的 `1`，而是当前噪声功率：

        - `20 dB`：信号功率是噪声的 100 倍，RMS 是 10 倍；
        - `0 dB`：信号与噪声功率相等；
        - `-5 dB`：噪声功率反而更大。

        所以“5 dB 就是强 5 倍”是错误的。5 dB 对应幅值比约 1.778，功率比约 3.162，而且必须说明是谁相对谁。
        """
    ),
    code(
        """
        clean = x[:3 * sr]
        noisy_5, noise_5 = add_noise_at_snr(clean, fixed_noise[:len(clean)], 5)
        achieved_snr = 10 * np.log10(np.mean(clean**2) / np.mean(noise_5**2))
        print(f"目标 SNR = 5.000 dB；实际 SNR = {achieved_snr:.6f} dB")
        print(f"RMS 比 = {rms(clean)/rms(noise_5):.4f}，理论 10^(5/20) = {10**(5/20):.4f}")
        print(f"功率比 = {np.mean(clean**2)/np.mean(noise_5**2):.4f}，理论 10^(5/10) = {10**(5/10):.4f}")

        assert abs(achieved_snr - 5) < 1e-9
        """
    ),
    md(
        """
        ## 4. 流式 DC blocker：算法很短，状态才是重点

        一阶 DC blocker：`y[n] = x[n] - x[n-1] + r*y[n-1]`。

        处理下一个 chunk 时，需要保留两个状态：上一个输入 `x[n-1]` 和上一个输出 `y[n-1]`。如果每个 chunk 都把它们清零，边界处会产生伪瞬态；chunk 大小变化时结果也不再一致。
        """
    ),
    code(
        """
        class StreamingDCBlocker:
            def __init__(self, r=0.995):
                self.r = float(r)
                self.prev_x = 0.0
                self.prev_y = 0.0

            def process(self, chunk):
                chunk = np.asarray(chunk, dtype=np.float64)
                out = np.empty_like(chunk)
                px, py = self.prev_x, self.prev_y
                for i, sample in enumerate(chunk):
                    current = sample - px + self.r * py
                    out[i] = current
                    px, py = sample, current
                self.prev_x, self.prev_y = px, py
                return out

        dc_input = x + 0.18
        whole_filter = StreamingDCBlocker()
        dc_whole = whole_filter.process(dc_input)

        def random_chunks(y, seed=7):
            generator = np.random.default_rng(seed)
            pos, chunks = 0, []
            while pos < len(y):
                size = int(generator.integers(1, 2049))
                chunks.append(y[pos:pos+size])
                pos += size
            return chunks

        trial_errors = []
        for seed in range(100):
            stream_filter = StreamingDCBlocker()
            dc_stream = np.concatenate([stream_filter.process(c) for c in random_chunks(dc_input, seed)])
            trial_errors.append(float(np.max(np.abs(dc_whole - dc_stream))))
        max_error = max(trial_errors)

        print(f"输入 DC 均值: {np.mean(dc_input):.5f}")
        print(f"跳过启动瞬态后的输出均值: {np.mean(dc_stream[sr:]):.7f}")
        print(f"整段 vs 100 组随机 chunk 最大误差: {max_error:.3e}")
        assert max_error < 1e-12

        zoom = slice(0, sr // 2)
        plt.figure(figsize=(12, 3))
        plt.plot(t[zoom], dc_input[zoom], label="含 DC 输入", alpha=.8)
        plt.plot(t[zoom], dc_stream[zoom], label="DC blocker 输出", alpha=.8)
        plt.xlabel("时间 (s)"); plt.ylabel("幅值"); plt.title("启动瞬态后，波形中心回到 0 附近")
        plt.legend(); plt.tight_layout(); plt.show()
        """
    ),
    md(
        """
        ### 流式检查题（每题 2 分）

        1. 为什么“随机切 100 多个 chunk 后与整段结果一致”比只测固定 20 ms chunk 更强？
        2. 除了滤波状态，流式前端还常保存哪些状态？至少说 3 个。
        3. 状态应该由全局变量保存，还是由每个会话独立拥有？为什么？

        <details><summary>参考答案</summary>

        1. 它覆盖更多边界位置与极短 chunk，能暴露 off-by-one、漏样本、边界重置等问题。
        2. 重采样相位/余数、STFT 尾部样本、CMVN 统计、VAD hangover、AEC 滤波器权重、时间戳累计量等。
        3. 每个会话独立拥有；全局状态会让并发用户之间串音、时间戳错位或互相污染统计量。

        </details>
        """
    ),
    md(
        """
        ## 5. VAD 与 Endpoint：把连续能量变成稳定边界

        最简单的能量 VAD 也包含四层概念：

        1. **frame**：在短时间片内统计能量；
        2. **hop**：每隔多少样本产生一次判断；
        3. **threshold**：高于阈值暂判语音；
        4. **hangover**：短暂停顿后继续保留若干帧，避免一个词被切碎。

        Endpoint 通常还要加入“最短语音”“最短尾部静音”“最大语句长度”等状态机规则。阈值越低不等于越好：误触发会增加算力与假识别，阈值过高则会截掉轻声和首尾音素。
        """
    ),
    code(
        """
        speech = x[:3 * sr]
        vad_signal = np.concatenate([np.zeros(sr), speech, np.zeros(sr)])
        frame_len, hop = int(.025 * sr), int(.010 * sr)

        def frame_rms_db(y, frame_len=400, hop=160):
            values = []
            for start in range(0, max(1, len(y) - frame_len + 1), hop):
                values.append(float(amp_to_dbfs(rms(y[start:start+frame_len]))))
            return np.asarray(values)

        def apply_hangover(raw, frames=10):
            out = np.zeros_like(raw, dtype=bool)
            remaining = 0
            for i, active in enumerate(raw):
                if active:
                    remaining = frames
                elif remaining > 0:
                    remaining -= 1
                out[i] = active or remaining > 0
            return out

        vad_db = frame_rms_db(vad_signal, frame_len, hop)
        vad_times = (np.arange(len(vad_db)) * hop + frame_len / 2) / sr

        def show_vad(threshold_db=-38, hangover_ms=150):
            raw = vad_db > threshold_db
            smooth = apply_hangover(raw, round(hangover_ms / (1000 * hop / sr)))
            fig, ax = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
            tt = np.arange(len(vad_signal))/sr
            ax[0].plot(tt, vad_signal, lw=.5)
            ax[0].axvspan(1, 4, color="green", alpha=.08, label="构造时的语音区")
            ax[0].set(ylabel="幅值", title="输入：1 s 静音 + 3 s 真人语音 + 1 s 静音")
            ax[0].legend(loc="upper right")
            ax[1].plot(vad_times, vad_db, label="帧 RMS dBFS")
            ax[1].axhline(threshold_db, color="r", ls="--", label="阈值")
            ax[1].fill_between(vad_times, -100, 0, where=raw, alpha=.18, label="原始 VAD")
            ax[1].fill_between(vad_times, -100, 0, where=smooth, alpha=.18, label="hangover 后")
            ax[1].set(xlabel="时间 (s)", ylabel="dBFS", ylim=(-90, 0), title="帧级能量与二值决策")
            ax[1].legend(ncol=4, loc="lower center")
            plt.tight_layout(); plt.show()
            starts = np.where(np.diff(np.r_[False, smooth].astype(int)) == 1)[0]
            ends = np.where(np.diff(np.r_[smooth, False].astype(int)) == -1)[0]
            print("检测区间:", [(round(vad_times[s], 3), round(vad_times[min(e, len(vad_times)-1)], 3)) for s, e in zip(starts, ends)])

        vad_ui = widgets.interactive(
            show_vad,
            threshold_db=widgets.FloatSlider(min=-60, max=-15, step=3, value=-38, description="阈值", continuous_update=False),
            hangover_ms=widgets.IntSlider(min=0, max=600, step=50, value=150, description="延迟关闭", continuous_update=False),
        )
        display(vad_ui)
        """
    ),
    md(
        """
        ### VAD 实验任务

        - 将阈值逐步升高到 `-20 dBFS`：观察一句话是否被切成多段。这是**漏检/截断风险**。
        - 将阈值降到 `-60 dBFS`：在真实底噪中往往会出现**误触发**；本段纯零静音太理想，所以生产测试必须加入设备噪声。
        - 将 hangover 从 `0` 增到 `600 ms`：连续性变好，但最终延迟和无效计算都增加。

        **思考：** 端点晚 300 ms 是识别准确率问题、延迟问题，还是两者都可能？为什么？
        """
    ),
    md(
        """
        ## 6. AEC：已知远端参考，估计扬声器到麦克风的回声路径

        通话场景中：`麦克风 = 回声 + 近端说话 + 噪声`。AEC 拥有扬声器正在播放的远端参考，可训练一个自适应滤波器预测回声，再从麦克风中相减。

        本实验使用 NLMS。它用输入能量归一化更新步长，比固定步长 LMS 更稳定。指标 `ERLE = 10 log10(P_echo / P_residual)`：在**只有回声的区间**，数值越高说明残余回声越小。ERLE 不是 ASR 准确率，也不适合在强近端讲话区间盲算。

        Double-talk 时近端语音不是回声，若继续把它当误差训练，会让滤波器发散。因此演示中使用已知近端真值冻结更新；真实系统只能检测/估计 double-talk。
        """
    ),
    code(
        """
        def nlms(reference, microphone, taps=128, mu=0.6, adapt_mask=None):
            reference = np.asarray(reference, dtype=np.float64)
            microphone = np.asarray(microphone, dtype=np.float64)
            padded = np.pad(reference, (taps - 1, 0))
            weights = np.zeros(taps)
            error = np.zeros_like(microphone)
            estimate = np.zeros_like(microphone)
            if adapt_mask is None:
                adapt_mask = np.ones(len(microphone), dtype=bool)
            for n in range(len(microphone)):
                ref_vector = padded[n:n+taps][::-1]
                estimate[n] = np.dot(weights, ref_vector)
                error[n] = microphone[n] - estimate[n]
                if adapt_mask[n]:
                    weights += mu * error[n] * ref_vector / (np.dot(ref_vector, ref_vector) + 1e-8)
            return error, estimate, weights

        duration = 4 * sr
        far = np.resize(x[:2*sr], duration)
        near_source = np.resize(x[3*sr:5*sr], duration)
        near = np.zeros(duration)
        near[2*sr:] = 0.35 * near_source[2*sr:]

        # 可学习的稀疏房间回声路径：20、57、93 个采样延迟。
        echo_path = np.zeros(128)
        echo_path[[20, 57, 93]] = [0.65, 0.25, -0.12]
        echo = np.convolve(far, echo_path, mode="full")[:duration]
        microphone = echo + near
        known_echo_only = np.abs(near) < 1e-12

        residual, echo_estimate, learned_path = nlms(
            far, microphone, taps=128, mu=.65, adapt_mask=known_echo_only
        )
        eval_region = slice(sr//2, 2*sr)
        erle = 10*np.log10(np.mean(echo[eval_region]**2) / (np.mean(residual[eval_region]**2) + EPS))
        path_similarity = np.corrcoef(echo_path, learned_path)[0, 1]

        print(f"echo-only 区间 ERLE: {erle:.2f} dB")
        print(f"真实/学习回声路径相关系数: {path_similarity:.4f}")
        print("2 s 后进入 double-talk，演示中冻结自适应更新以保护近端语音。")

        fig, ax = plt.subplots(3, 1, figsize=(13, 7), sharex=False)
        aec_t = np.arange(duration)/sr
        ax[0].plot(aec_t, microphone, lw=.5, label="麦克风（回声+近端）")
        ax[0].plot(aec_t, echo, lw=.5, alpha=.7, label="回声真值")
        ax[0].axvline(2, color="r", ls="--", label="double-talk 开始")
        ax[0].set(ylabel="幅值", title="AEC 输入"); ax[0].legend(ncol=3)
        ax[1].plot(aec_t, residual, lw=.5, label="AEC 残差")
        ax[1].plot(aec_t, near, lw=.5, alpha=.7, label="近端真值")
        ax[1].set(ylabel="幅值", title=f"AEC 输出；echo-only ERLE={erle:.1f} dB"); ax[1].legend()
        ax[2].stem(echo_path, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="真实路径")
        ax[2].plot(learned_path, color="C1", lw=1, label="NLMS 学习路径")
        ax[2].set(xlabel="延迟 (samples)", ylabel="系数", title="自适应滤波器在学习什么"); ax[2].legend()
        plt.tight_layout(); plt.show()

        assert erle > 20
        assert path_similarity > .95
        """
    ),
    md(
        """
        ### AEC 检查题（每题 2 分）

        1. AEC 与普通降噪最大的输入差别是什么？
        2. 为什么必须在回声路径变化后重新收敛？
        3. 为什么 ERLE 很高仍不能证明通话体验或 ASR 一定好？
        4. 为什么把 double-talk 区间算入 ERLE 可能产生误导？

        <details><summary>参考答案</summary>

        1. AEC 有扬声器播放信号这个参考，可估计相关回声；普通降噪通常没有噪声的精确参考。
        2. 扬声器—房间—麦克风传递路径变了，旧滤波器预测的不再是当前回声。
        3. 还可能存在近端语音失真、噪声、非线性、端点问题；最终需听感和 ASR 指标共同验收。
        4. 残差中包含希望保留的近端语音，它不是消除失败产生的残余回声。

        </details>
        """
    ),
    md(
        """
        ## 7. 两麦 delay-and-sum：用空间延迟增强目标方向

        同一声源到达不同麦克风的时间略有差异。如果已知目标方向的延迟，可先把目标对齐再求平均：目标同相叠加，其他方向的干扰通常不能完全对齐。

        为了让因果关系看清，本实验使用整数采样延迟、无混响、已知目标方向。真实阵列还要处理麦克风几何、亚采样延迟、频率相关相位、混响、增益/时钟校准和声源移动。
        """
    ),
    code(
        """
        def delay_signal(y, samples):
            out = np.zeros_like(y)
            if samples == 0:
                return y.copy()
            if samples > 0:
                out[samples:] = y[:-samples]
            else:
                out[:samples] = y[-samples:]
            return out

        n = 3 * sr
        target = x[:n]
        interferer = np.resize(x[4*sr:6*sr], n)
        target_delay = 6
        interferer_delay = -9

        target_1 = target
        target_2 = delay_signal(target, target_delay)
        noise_1 = .7 * interferer
        noise_2 = .7 * delay_signal(interferer, interferer_delay)
        mic1 = target_1 + noise_1
        mic2 = target_2 + noise_2

        # 将第二路左移 6 samples，对齐目标；尾部边界不参与指标。
        aligned_mic2 = delay_signal(mic2, -target_delay)
        beam = .5 * (mic1 + aligned_mic2)
        beam_target = .5 * (target_1 + delay_signal(target_2, -target_delay))
        beam_noise = .5 * (noise_1 + delay_signal(noise_2, -target_delay))
        valid = slice(32, n-32)

        snr_mic1 = 10*np.log10(np.mean(target_1[valid]**2) / np.mean(noise_1[valid]**2))
        snr_beam = 10*np.log10(np.mean(beam_target[valid]**2) / np.mean(beam_noise[valid]**2))
        array_gain = snr_beam - snr_mic1
        print(f"Mic 1 SNR: {snr_mic1:.2f} dB")
        print(f"波束形成后 SNR: {snr_beam:.2f} dB")
        print(f"本仿真的阵列增益: {array_gain:.2f} dB")

        segment = slice(sr, sr + 1000)
        local_t = np.arange(1000)/sr * 1000
        fig, ax = plt.subplots(2, 1, figsize=(13, 6))
        ax[0].plot(local_t, mic1[segment], label="Mic 1", alpha=.8)
        ax[0].plot(local_t, mic2[segment], label="Mic 2（目标晚 6 samples）", alpha=.8)
        ax[0].set(ylabel="幅值", title="对齐前：两路目标存在到达时间差"); ax[0].legend()
        ax[1].plot(local_t, target_1[segment], label="目标真值", alpha=.8)
        ax[1].plot(local_t, beam[segment], label="对齐求和输出", alpha=.8)
        ax[1].set(xlabel="局部时间 (ms)", ylabel="幅值", title=f"目标方向对齐；SNR 增益 {array_gain:.2f} dB")
        ax[1].legend(); plt.tight_layout(); plt.show()

        assert array_gain > 2
        """
    ),
    md(
        """
        ### 阵列检查题（每题 2 分）

        1. 两路直接平均为什么不一定增强目标？
        2. 如果估计方向错了，会发生什么？
        3. 本实验为什么能直接计算目标/干扰 SNR，而真实录音通常不能？

        <details><summary>参考答案</summary>

        1. 到达延迟导致相位不一致，某些频率甚至会相消；应先按目标方向对齐。
        2. 目标不能同相叠加，阵列增益下降甚至损伤目标语音。
        3. 仿真中目标与干扰是分别构造的，有干净真值；真实麦克风只观测混合信号。

        </details>
        """
    ),
    md(
        """
        ## 8. 前端顺序不是随意排列

        一个常见逻辑顺序如下，但真实产品要按设备和模型共同验证：

        1. **解码 PCM / 通道与时钟检查**：格式错了，后面全错；
        2. **重采样与基础校正**：建立统一采样率、处理 DC/通道增益；
        3. **AEC**：要尽量使用仍与扬声器参考匹配的麦克风信号；
        4. **波束形成 / 降噪**：利用空间和统计结构抑制干扰；
        5. **AGC**：稳定工作电平，但必须避免放大残余噪声和削波；
        6. **VAD / Endpoint**：决定何时送模型、何时结束；
        7. **Log-Mel / CMVN**：严格遵守训练时的特征契约。

        AEC、降噪、AGC 的具体先后可以因系统设计而变，不能把这张列表当成不需实验的定律。每一步都可能改善某个前端指标，却伤害 ASR。
        """
    ),
    code(
        """
        # 一个最小的“状态归谁”示意，不假装实现完整产品算法。
        class FrontendSession:
            def __init__(self, sample_rate=16000):
                self.sample_rate = sample_rate
                self.dc = StreamingDCBlocker()
                self.total_input_samples = 0
                self.feature_tail = np.empty(0, dtype=np.float64)
                self.vad_hangover_frames = 0

            def process_pcm(self, chunk):
                chunk = np.asarray(chunk, dtype=np.float64)
                cleaned = self.dc.process(chunk)
                start_sample = self.total_input_samples
                self.total_input_samples += len(chunk)
                return {
                    "samples": cleaned,
                    "start_time_s": start_sample / self.sample_rate,
                    "end_time_s": self.total_input_samples / self.sample_rate,
                }

        alice, bob = FrontendSession(), FrontendSession()
        result_a1 = alice.process_pcm(x[:317])
        result_b1 = bob.process_pcm(x[:503])
        result_a2 = alice.process_pcm(x[317:1000])
        print("Alice 第二块时间:", result_a2["start_time_s"], "→", result_a2["end_time_s"])
        print("Bob 独立累计样本:", bob.total_input_samples)
        assert alice.total_input_samples == 1000
        assert bob.total_input_samples == 503
        assert abs(result_a2["start_time_s"] - 317/sr) < 1e-12
        """
    ),
    md(
        """
        ## 9. 生产验收：不要让一个漂亮数字替代完整证据

        | 层级 | 必测内容 | 典型失败 |
        |---|---|---|
        | 信号单元测试 | 目标 SNR、DC、削波、ERLE、阵列增益 | 公式/参考值/符号错 |
        | 流式等价 | 随机 chunk、空块、1-sample 块、尾块 | 状态重置、漏样本、重复样本 |
        | 会话隔离 | 多会话交错、重连、结束清理 | 串音、状态泄漏 |
        | ASR 质量 | clean/noisy/device/距离/口音/长度分桶 CER/WER | 前端指标升、识别反降 |
        | 性能 | RTF、P50/P95/P99、CPU、内存、功耗 | 平均快但尾延迟超标 |
        | 鲁棒与回滚 | 极端幅值、NaN、采样率错、功能开关 | 崩溃或无法快速回退 |

        **最重要的实验纪律：** 每次只改一个因素；保存原始音频与处理后音频；固定评测清单；同时报告总体结果和困难分桶；不隐藏质量退化。
        """
    ),
    md(
        """
        ## 10. 48 分综合测试（先独立作答）

        每题 4 分。建议把答案写到自己的学习日志，并给出原因，不只写一个词。

        1. 为什么数字音频中的 `0 dBFS` 很强，而不是“零强度”？
        2. `-20 dBFS` 的幅值相对满幅是多少？功率相对满幅约是多少？
        3. 信号比噪声高 `5 dB` 时，幅值比和功率比各是多少？
        4. peak、RMS、SNR 分别回答什么问题？
        5. 为什么幅值低于 `1000` 不能跨 Cool Edit、Sonic Visualiser 和 NumPy 直接比较？
        6. VAD 的 frame、hop、threshold、hangover 各自做什么？
        7. hangover 墑大时，连续性和延迟通常如何变化？
        8. AEC 比普通降噪多了哪个关键信号？ERLE 的参考是谁？
        9. double-talk 时为什么要减慢或冻结自适应？
        10. delay-and-sum 为什么必须按目标方向对齐？
        11. 举出 4 个必须跨 chunk 保存的前端状态，并说明状态归属。
        12. 某前端让 SNR 提高 6 dB，但 WER 变差。你会先检查哪些证据，能否上线？

        <details><summary>评分要点</summary>

        1. 参考幅值是数字满幅 1；0 dB 表示与参考相等。
        2. 幅值 0.1，功率 0.01。
        3. 幅值约 1.778，功率约 3.162。
        4. peak 看最大瞬时幅值/削波；RMS 看窗口有效电平；SNR 看信号功率相对噪声功率。
        5. 显示的数据格式、位深、归一化和参考可能不同，必须先统一标尺。
        6. frame 是统计窗，hop 是决策步进，threshold 转为语音/非语音，hangover 抑制短暂掉线。
        7. 连续性通常提高，尾延迟与计算通常增加。
        8. 多了远端扬声器参考；ERLE 比较原回声功率与残余回声功率。
        9. 近端语音不是回声，拿它更新会污染路径估计。
        10. 目标到达各麦的相位/延迟不同，不对齐可能相消。
        11. 如滤波器历史、重采样相位、STFT 尾部、VAD 状态、AEC 权重、CMVN；每个会话独立。
        12. 查语音失真、分桶、特征契约、削波、端点、不同设备；最终目标是 ASR 与产品指标，证据不足或 WER 明显退化不能直接上线。

        </details>
        """
    ),
    md(
        """
        ## 11. 离开实验室前的口述验收

        不看答案，用 3 分钟对另一个人讲清楚：

        > “一条 16 kHz 浮点 PCM 进入系统后，我怎样判断输入质量；为什么 dB 必须带参考；如何构造 5 dB SNR；VAD、AEC、波束形成分别利用什么信息；最后怎样证明流式实现正确且真的帮助 ASR。”

        如果其中某一段说不顺，回到对应图，把滑块改到一个极端值，再用自己的话写下观察。**能预测图会怎样变化，才算真正理解。**

        下一专题建议：语义后处理综合实验室——时间戳、说话人、ITN、置信度、N-best 与 NLU/LLM 证据链。
        """
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python (learn-asr)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
)
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUT)
print(f"Wrote {OUT} with {len(cells)} cells")
