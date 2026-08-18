from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

from notebook_layout import (
    ensure_executed_directories,
    executed_path,
    sanitize_notebook_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


@dataclass(frozen=True)
class Lesson:
    number: int
    slug: str
    title: str
    question: str
    concepts: tuple[str, str, str, str]
    cells: list
    implementation: str
    next_step: str


def intro(lesson: Lesson) -> list:
    return [
        md(
            f"""
# 音频零基础 {lesson.number}/6：{lesson.title}

这节课从生活中的声音出发，再落到 Python 数组。**先建立直觉，再看公式，最后用代码验证。**

| 项目 | 内容 |
|---|---|
| 本课要回答的问题 | {lesson.question} |
| 前置要求 | 会运行 Notebook；看不懂代码时先读 Python/PyTorch 基础 1～3 |
| 建议投入 | 90～120 分钟，分成“直觉”和“代码”两次完成也可以 |
| 四个关键词 | {'、'.join(lesson.concepts)} |
| 通关证据 | 能用自己的话解释、能算一个数字例子、能画图验证、能从空白实现核心函数 |

本课不是词汇表。每出现一个量，都要写清楚：**它描述什么、单位是什么、数组中怎样表示、改变后听觉或图形怎样变化。**
"""
        ),
        md(
            f"""
## 课前回忆：先写猜测，不查答案

1. 对问题“{lesson.question}”写下你现在的答案；不会就写“不知道”。
2. 四个关键词中，圈出最陌生的一个：{' / '.join(lesson.concepts)}。
3. 画一条横轴是时间的线，标出 0 秒、0.5 秒和 1 秒。
4. 写下一个可验证的预测：如果把声音的某个参数加倍，图或听感会怎样？

学完后回到这里，用另一种颜色修正。保留错误猜测，它是学习证据。
"""
        ),
        md(
            """
## 固定观察框架：物理世界 → 数字 → 图 → 听感

```text
声源振动 → 空气压力随时间变化 → 麦克风电信号 → 离散采样值 x[n]
                                                ↓
                                      波形 / 数值统计 / 频谱
```

后面所有 ASR 前端课都沿这条链展开。波形不是声音本身，而是麦克风在一串离散时刻记录下来的数字。
"""
        ),
    ]


def finish(lesson: Lesson) -> list:
    return [
        md(
            f"""
## 分层练习：不要一次做完

### A. 直觉与单位（每题 1 分）

1. 不看上文，用一句话定义：{lesson.concepts[0]}。
2. 为 `{lesson.concepts[1]}` 写出单位；如果它没有单位，要明确说明。
3. 举一个生活中的例子解释 `{lesson.concepts[2]}`。
4. 画图说明 `{lesson.concepts[3]}` 增大时，横轴或纵轴怎样变化。

### B. 数字与预测（每题 2 分）

5. 自己构造一个包含具体数字的计算例子，并标出每一步单位。
6. 把本课一个参数改成 0.5 倍和 2 倍；运行前先画出预期图形。
7. 找出一个“代码能运行，但物理含义错误”的输入，解释为什么错误。
8. 用 `shape / dtype / min / max` 四项审计本课最重要的数组。

### C. 实现与迁移（每题 3 分）

9. 从空白实现：**{lesson.implementation}**，不能复制上面的函数。
10. 为实现写正常、边界和错误输入三类测试。
11. 换一组参数或换一条音频，验证结论是否仍成立。
12. 用 90 秒向没学过编程的人解释本课，只允许使用两个术语，并必须包含一个数字例子。

满分 24 分。达到 19 分且第 9～10 题完成，才建议继续；15～18 分次日重做；低于 15 分先回到图和单位。
"""
        ),
        md(
            f"""
## 最小掌握门禁

- [ ] 我能把本课每个量说成“含义 + 单位 + 数组表示”。
- [ ] 我能在运行前预测参数变化的方向。
- [ ] 我能从空白完成核心实现并覆盖边界输入。
- [ ] 我能说出一个常见误解，以及用什么证据推翻它。
- [ ] 我已把错题写入根目录 `LEARNING_LOG.md`，并安排明天、7 天、30 天复习。

下一步：{lesson.next_step}
"""
        ),
    ]


LESSONS = [
    Lesson(
        1,
        "振动波形与时间轴",
        "振动、波形与时间轴",
        "声音是什么，为什么能画成一条随时间变化的曲线？",
        ("振动", "时间轴", "波形", "采样值"),
        [
            md("## 1. 从慢振动开始：先看得见，再谈听得见\n\n物体来回运动叫振动。麦克风记录的是这种变化引起的空气压力变化。下面先画 2 Hz 的慢振动。"),
            code(
                """
import numpy as np
import matplotlib.pyplot as plt

duration_seconds = 1.0
points_for_drawing = 1000
time_seconds = np.linspace(0, duration_seconds, points_for_drawing, endpoint=False)
slow_vibration = np.sin(2 * np.pi * 2 * time_seconds)

plt.figure(figsize=(10, 3))
plt.plot(time_seconds, slow_vibration)
plt.xlabel("time (seconds)")
plt.ylabel("relative displacement")
plt.title("2 cycles in 1 second")
plt.grid(True)
plt.show()

assert time_seconds.shape == slow_vibration.shape == (1000,)
"""
            ),
            md("## 2. 横轴与纵轴分别回答什么\n\n横轴回答“什么时候”，纵轴回答“相对平衡位置偏了多少”。正负号表示方向，不代表好坏。"),
            code(
                """
for index in [0, 125, 250, 375, 500]:
    print(f"index={index:3d}, time={time_seconds[index]:.3f} s, value={slow_vibration[index]:+.3f}")

print("shape:", slow_vibration.shape)
print("dtype:", slow_vibration.dtype)
print("min/max:", slow_vibration.min(), slow_vibration.max())
"""
            ),
            md("## 3. 毫秒只是更方便的时间单位\n\n1 秒 = 1000 毫秒。ASR 常用 25 ms 帧长、10 ms 帧移，但它们仍然只是时间。"),
            code(
                """
def milliseconds_to_seconds(milliseconds: float) -> float:
    if milliseconds < 0:
        raise ValueError("milliseconds must be non-negative")
    return milliseconds / 1000.0


for ms in [1, 10, 25, 100, 1000]:
    print(f"{ms:4d} ms = {milliseconds_to_seconds(ms):.3f} s")

assert milliseconds_to_seconds(25) == 0.025
"""
            ),
            md("## 4. 数组只是按时间顺序保存的测量值\n\n`x[n]` 中的 `n` 是样本序号。只看 `n` 不知道真实时间；还必须知道采样率，这会在第 4 节桥梁课完整学习。"),
            code(
                """
first_ten = slow_vibration[:10]
print(first_ten)
print("第 4 个值 x[3] =", first_ten[3])
assert len(first_ten) == 10
"""
            ),
        ],
        "实现 milliseconds_to_seconds，并写出负数输入策略",
        "音频基础 2：周期、频率、相位与正弦波。",
    ),
    Lesson(
        2,
        "周期频率相位与正弦波",
        "周期、频率、相位与正弦波",
        "“每秒振动多少次”怎样决定波形的疏密和音高？",
        ("周期", "频率", "赫兹", "相位"),
        [
            md("## 1. 频率是每秒完成的周期数\n\n频率单位 Hz（赫兹）等价于“次/秒”。周期是一次完整重复所需的秒数，两者互为倒数。"),
            code(
                """
def period_seconds(frequency_hz: float) -> float:
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    return 1.0 / frequency_hz


for hz in [2, 100, 440, 1000]:
    print(f"{hz:4d} Hz -> period={period_seconds(hz):.6f} s = {period_seconds(hz)*1000:.3f} ms")
"""
            ),
            md("## 2. 同一秒内，频率越高，波形越密\n\n先只比较图的疏密，不急着听。"),
            code(
                """
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 0.02, 1000, endpoint=False)
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
for axis, hz in zip(axes, [100, 400]):
    axis.plot(t * 1000, np.sin(2 * np.pi * hz * t))
    axis.set_title(f"{hz} Hz")
    axis.set_ylabel("amplitude")
    axis.grid(True)
axes[-1].set_xlabel("time (ms)")
plt.tight_layout()
plt.show()
"""
            ),
            md("## 3. 相位改变从周期的哪里开始\n\n相位平移波形，但不改变频率。单独听纯音时相位常不明显；叠加和多麦克风处理中却很重要。"),
            code(
                """
frequency_hz = 100
for phase_name, phase_radians in [("0", 0), ("pi/2", np.pi/2), ("pi", np.pi)]:
    plt.plot(t * 1000, np.sin(2*np.pi*frequency_hz*t + phase_radians), label=phase_name)
plt.xlim(0, 15)
plt.xlabel("time (ms)")
plt.ylabel("amplitude")
plt.legend(title="phase")
plt.grid(True)
plt.show()
"""
            ),
            md("## 4. 离散数组中一个周期有多少点\n\n采样率除以频率，得到“每周期样本数”。结果不一定是整数。"),
            code(
                """
def samples_per_period(sample_rate_hz: int, frequency_hz: float) -> float:
    if sample_rate_hz <= 0 or frequency_hz <= 0:
        raise ValueError("rates must be positive")
    return sample_rate_hz / frequency_hz


print("16 kHz recording, 440 Hz tone:", samples_per_period(16_000, 440), "samples/cycle")
print("16 kHz recording, 1 kHz tone:", samples_per_period(16_000, 1_000), "samples/cycle")
assert samples_per_period(16_000, 1_000) == 16
"""
            ),
        ],
        "实现 period_seconds 和 samples_per_period，并验证 frequency×period=1",
        "音频基础 3：振幅、RMS、功率与 dB。",
    ),
    Lesson(
        3,
        "振幅RMS功率与dB",
        "振幅、RMS、功率与 dB",
        "波形有多大怎样量化，为什么 dB 必须说明参考值？",
        ("峰值", "RMS", "功率", "dB"),
        [
            md("## 1. 峰值看最大瞬时幅度，RMS 看整体有效大小\n\n两者回答不同问题。相同峰值的信号可以有不同 RMS。"),
            code(
                """
import numpy as np
import matplotlib.pyplot as plt

def peak_and_rms(signal: np.ndarray) -> tuple[float, float]:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        raise ValueError("signal must not be empty")
    peak = float(np.max(np.abs(signal)))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    return peak, rms


t = np.linspace(0, 1, 16_000, endpoint=False)
sine = 0.5 * np.sin(2*np.pi*440*t)
square = 0.5 * np.sign(np.sin(2*np.pi*440*t))
print("sine peak/rms:", peak_and_rms(sine))
print("square peak/rms:", peak_and_rms(square))
"""
            ),
            md("## 2. 功率与幅度平方相关\n\n教学中常用波形平方的平均值表示相对功率；RMS 正好是它的平方根。"),
            code(
                """
power = float(np.mean(sine ** 2))
_, rms = peak_and_rms(sine)
print("mean square power:", power)
print("rms squared:", rms ** 2)
assert np.isclose(power, rms ** 2)
"""
            ),
            md("## 3. dB 是比值的对数，不是孤立单位\n\n幅度比用 `20 log10`；功率比用 `10 log10`。两者在功率与幅度平方对应时一致。"),
            code(
                """
def amplitude_db(amplitude: float, reference: float = 1.0, floor_db: float = -120.0) -> float:
    if amplitude < 0 or reference <= 0:
        raise ValueError("amplitude must be non-negative and reference positive")
    ratio = max(amplitude / reference, 10 ** (floor_db / 20))
    return float(20 * np.log10(ratio))


for amplitude in [1.0, 0.5, 0.1, 0.01, 0.0]:
    print(f"amplitude={amplitude:>4} -> {amplitude_db(amplitude):7.2f} dB re 1.0")
"""
            ),
            md("## 4. dBFS 以数字系统最大幅度为参考\n\n归一化浮点波形通常以 1.0 为满幅参考。0 dBFS 是上限附近，不表示没有声音；静音趋向负无穷。"),
            code(
                """
peak, rms = peak_and_rms(sine)
print("peak dBFS:", amplitude_db(peak, reference=1.0))
print("RMS dBFS:", amplitude_db(rms, reference=1.0))

clipped = np.clip(2.5 * sine, -1.0, 1.0)
print("clipped sample ratio:", np.mean(np.abs(clipped) >= 0.999))
plt.plot(t[:300] * 1000, sine[:300], label="original")
plt.plot(t[:300] * 1000, clipped[:300], label="clipped")
plt.xlabel("time (ms)"); plt.ylabel("amplitude"); plt.legend(); plt.grid(True); plt.show()
"""
            ),
        ],
        "实现 peak_and_rms 与 amplitude_db，并解释零幅度的处理策略",
        "音频基础 4：采样、量化、PCM、位深与通道。",
    ),
    Lesson(
        4,
        "采样量化PCM位深与通道",
        "采样、量化、PCM、位深与通道",
        "连续变化的声音怎样变成 WAV 文件中的整数或浮点数组？",
        ("采样率", "量化", "位深", "通道"),
        [
            md("## 1. 采样是在固定时刻取值\n\n采样率 16 kHz 表示每秒记录 16,000 个数；它不是音频中最高频率，也不是文件大小。"),
            code(
                """
def sample_count(duration_seconds: float, sample_rate_hz: int) -> int:
    if duration_seconds < 0 or sample_rate_hz <= 0:
        raise ValueError("duration must be non-negative and sample rate positive")
    return round(duration_seconds * sample_rate_hz)


for duration in [0.01, 0.025, 1.0, 2.5]:
    print(duration, "s at 16 kHz ->", sample_count(duration, 16_000), "samples")
assert sample_count(0.025, 16_000) == 400
"""
            ),
            md("## 2. 采样率越高，同一时间段的点越密\n\n下面用高密度曲线代表连续信号的近似，只用于帮助观察。"),
            code(
                """
import numpy as np
import matplotlib.pyplot as plt

dense_t = np.linspace(0, 0.01, 5000, endpoint=False)
dense_x = np.sin(2*np.pi*440*dense_t)
plt.plot(dense_t*1000, dense_x, color="gray", label="dense reference")
for sr in [2000, 8000]:
    t = np.arange(sample_count(0.01, sr)) / sr
    x = np.sin(2*np.pi*440*t)
    plt.scatter(t*1000, x, s=18, label=f"{sr} samples/s")
plt.xlabel("time (ms)"); plt.ylabel("amplitude"); plt.legend(); plt.grid(True); plt.show()
"""
            ),
            md("## 3. 量化把连续幅度映射到有限等级\n\n位深越低，等级越少，量化误差通常越明显。真实 PCM 常见 16 bit。"),
            code(
                """
def quantize_unit(signal: np.ndarray, bits: int) -> np.ndarray:
    if bits < 2:
        raise ValueError("bits must be at least 2")
    levels = 2 ** bits
    clipped = np.clip(signal, -1.0, 1.0)
    return np.round((clipped + 1) * (levels - 1) / 2) * 2 / (levels - 1) - 1


x = np.linspace(-1, 1, 17)
for bits in [2, 3, 8]:
    q = quantize_unit(x, bits)
    print(bits, "bit -> unique levels used:", len(np.unique(q)), "max error:", np.max(np.abs(x-q)))
"""
            ),
            md("## 4. PCM、dtype 与通道共同构成输入契约\n\n单声道常见 shape `[T]`，双声道常见 `[T,2]`。不能在不知道通道含义时随意 `flatten()`。"),
            code(
                """
mono = np.array([0.0, 0.5, -0.5, 0.999], dtype=np.float32)
pcm16 = np.round(np.clip(mono, -1, 1) * 32767).astype(np.int16)
restored = pcm16.astype(np.float32) / 32767
stereo = np.stack([mono, 0.5 * mono], axis=1)

print("mono:", mono.shape, mono.dtype)
print("pcm16:", pcm16.shape, pcm16.dtype, pcm16)
print("stereo:", stereo.shape, stereo.dtype)
print("round-trip max error:", np.max(np.abs(mono-restored)))
assert stereo.shape == (4, 2)
"""
            ),
        ],
        "实现 sample_count 与 quantize_unit，并审计 mono/stereo 的 shape",
        "音频基础 5：叠加、谐波、噪声与 SNR。",
    ),
    Lesson(
        5,
        "叠加谐波噪声与SNR",
        "叠加、谐波、噪声与 SNR",
        "复杂声音怎样由简单成分叠加，噪声强弱又怎样量化？",
        ("叠加", "谐波", "噪声", "SNR"),
        [
            md("## 1. 波形可以逐点相加\n\n线性叠加是频谱、滤波、回声和多麦克风处理的共同起点。"),
            code(
                """
import numpy as np
import matplotlib.pyplot as plt

sr = 16_000
t = np.arange(sr) / sr
fundamental = 0.6 * np.sin(2*np.pi*200*t)
second_harmonic = 0.2 * np.sin(2*np.pi*400*t)
third_harmonic = 0.1 * np.sin(2*np.pi*600*t)
mixture = fundamental + second_harmonic + third_harmonic

plt.plot(t[:400]*1000, fundamental[:400], label="200 Hz")
plt.plot(t[:400]*1000, mixture[:400], label="mixture", alpha=0.8)
plt.xlabel("time (ms)"); plt.ylabel("amplitude"); plt.legend(); plt.grid(True); plt.show()
"""
            ),
            md("## 2. 谐波是基频整数倍附近的成分\n\n真实语音比这个例子复杂，但“基频 + 谐波 + 噪声”的模型能建立重要直觉。"),
            code(
                """
frequency_hz = np.fft.rfftfreq(len(mixture), d=1/sr)
magnitude = np.abs(np.fft.rfft(mixture))
top = np.argsort(magnitude)[-6:][::-1]
for index in top:
    print(f"{frequency_hz[index]:7.1f} Hz magnitude={magnitude[index]:.1f}")
"""
            ),
            md("## 3. SNR 比较信号功率与噪声功率\n\nSNR 也必须说明信号与噪声怎样定义。正 dB 表示信号功率更强，0 dB 表示二者功率相等。"),
            code(
                """
def power(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0:
        raise ValueError("signal must not be empty")
    return float(np.mean(signal ** 2))


def snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    noise_power = power(noise)
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(power(clean) / noise_power))


rng = np.random.default_rng(7)
noise = rng.normal(size=len(mixture))
noise *= np.sqrt(power(mixture) / power(noise))
print("equal-power SNR:", snr_db(mixture, noise), "dB")
"""
            ),
            md("## 4. 按目标 SNR 缩放噪声\n\n只改噪声比例，不改干净信号，才能进行受控实验。"),
            code(
                """
def mix_at_snr(clean: np.ndarray, raw_noise: np.ndarray, target_snr_db: float):
    if clean.shape != raw_noise.shape:
        raise ValueError("clean and noise must have the same shape")
    target_noise_power = power(clean) / (10 ** (target_snr_db / 10))
    scale = np.sqrt(target_noise_power / power(raw_noise))
    scaled_noise = raw_noise * scale
    return clean + scaled_noise, scaled_noise


for target in [20, 10, 0, -5]:
    noisy, scaled_noise = mix_at_snr(mixture, rng.normal(size=len(mixture)), target)
    print(f"target={target:>3} dB, measured={snr_db(mixture, scaled_noise):7.3f} dB, peak={np.max(np.abs(noisy)):.3f}")
"""
            ),
        ],
        "实现 snr_db 与 mix_at_snr，并用测量值验证三个目标 SNR",
        "音频基础 6：读取、试听、可视化和审计真实 WAV。",
    ),
    Lesson(
        6,
        "真实WAV读取试听与输入审计",
        "真实 WAV：读取、试听、可视化与输入审计",
        "拿到一条 WAV 后，怎样先确认它是什么，再交给 ASR？",
        ("WAV", "元数据", "波形审计", "输入契约"),
        [
            md("## 1. 读取前先明确路径，读取后先看元数据\n\n这节使用仓库自带的 FSDD 数字语音。不要先画复杂图，先确认采样率、shape、dtype 和时长。"),
            code(
                """
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from IPython.display import Audio, display

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent
audio_path = ROOT / "data" / "0_jackson_0.wav"
audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)

print("path:", audio_path)
print("sample_rate:", sample_rate, "Hz")
print("shape/dtype:", audio.shape, audio.dtype)
print("duration:", len(audio) / sample_rate, "seconds")
assert audio.ndim == 1
"""
            ),
            md("## 2. 试听是证据之一，但不能代替测量\n\n扬声器和音量设置会影响听感；代码仍要审计数值。"),
            code(
                """
display(Audio(audio, rate=sample_rate))

time_seconds = np.arange(len(audio)) / sample_rate
plt.figure(figsize=(11, 3))
plt.plot(time_seconds, audio)
plt.xlabel("time (seconds)")
plt.ylabel("amplitude")
plt.title(audio_path.name)
plt.grid(True)
plt.show()
"""
            ),
            md("## 3. 写一个最小输入审计器\n\n审计器不判断“识别一定好不好”，只检查输入是否满足明确契约，并暴露风险信号。"),
            code(
                """
def audit_waveform(waveform: np.ndarray, sample_rate_hz: int) -> dict:
    waveform = np.asarray(waveform)
    if waveform.ndim not in {1, 2}:
        raise ValueError(f"expected [T] or [T,C], got {waveform.shape}")
    if waveform.shape[0] == 0:
        raise ValueError("waveform must not be empty")
    if sample_rate_hz <= 0:
        raise ValueError("sample rate must be positive")
    values = waveform.astype(np.float64)
    return {
        "sample_rate_hz": sample_rate_hz,
        "channels": 1 if waveform.ndim == 1 else waveform.shape[1],
        "samples_per_channel": waveform.shape[0],
        "duration_seconds": waveform.shape[0] / sample_rate_hz,
        "dtype": str(waveform.dtype),
        "peak": float(np.max(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "dc": float(np.mean(values)),
        "clipped_ratio": float(np.mean(np.abs(values) >= 0.999)),
        "finite": bool(np.isfinite(values).all()),
    }


report = audit_waveform(audio, sample_rate)
for key, value in report.items():
    print(f"{key:20s}: {value}")
"""
            ),
            md("## 4. 错误元数据会改变时间解释\n\n同一数组如果误标采样率，样本值没变，但时长、音高和后续帧数解释都会错。"),
            code(
                """
for claimed_rate in [sample_rate // 2, sample_rate, sample_rate * 2]:
    claimed_duration = len(audio) / claimed_rate
    print(f"claimed rate={claimed_rate:5d} Hz -> claimed duration={claimed_duration:.3f} s")

assert np.isclose(report["duration_seconds"], len(audio) / sample_rate)
assert report["finite"]
assert report["peak"] <= 1.0
"""
            ),
            md("## 5. 把审计结果变成明确的入口门槛\n\n生产系统的门槛取决于模型契约。教学示例只接受单声道、8 kHz、有限浮点数；不要把这些数字误当成所有 ASR 的通用标准。"),
            code(
                """
def require_teaching_input(report: dict) -> None:
    if report["channels"] != 1:
        raise ValueError("this lesson expects mono audio")
    if report["sample_rate_hz"] != 8_000:
        raise ValueError("this lesson expects 8 kHz audio")
    if not report["finite"]:
        raise ValueError("waveform contains NaN or infinity")
    if report["peak"] > 1.0:
        raise ValueError("normalized float waveform is outside [-1, 1]")


require_teaching_input(report)
print("teaching input contract passed")
"""
            ),
        ],
        "从空白实现 audit_waveform，并用空数组、立体声、NaN 和错误采样率做故障注入",
        "进入 ASR 主线第 1 课《声音与采样》，把这些直觉用于采样定理、混叠和真实语音。",
    ),
]


def build_notebook(lesson: Lesson):
    notebook = nbf.v4.new_notebook(
        cells=intro(lesson) + lesson.cells + finish(lesson),
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
            "audio_foundation_course": {
                "lesson": lesson.number,
                "total_lessons": len(LESSONS),
                "title": lesson.title,
            },
        },
    )
    _, notebook = nbf.validator.normalize(notebook, strip_invalid_metadata=True)
    return notebook


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    ensure_executed_directories()
    for lesson in LESSONS:
        stem = f"音频基础_{lesson.number:02d}_{lesson.slug}"
        source_path = NOTEBOOK_DIR / f"{stem}.ipynb"
        executed_output = executed_path(source_path)
        source_notebook = build_notebook(lesson)
        nbf.write(source_notebook, source_path)

        executed_notebook = copy.deepcopy(source_notebook)
        NotebookClient(
            executed_notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        sanitize_notebook_outputs(executed_notebook)
        _, executed_notebook = nbf.validator.normalize(executed_notebook, strip_invalid_metadata=True)
        nbf.write(executed_notebook, executed_output)
        print(f"built {stem}: {len(source_notebook.cells)} cells")


if __name__ == "__main__":
    main()
