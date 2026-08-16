"""从波形走到 Log-Mel：一个只依赖 Python 标准库的 ASR 声学特征练习。

运行：python lesson_audio_features.py
输出：demo_signal.csv、spectrum_frame.csv、logmel.csv
"""

from __future__ import annotations

import csv
import math
import wave
from pathlib import Path


ROOT = Path(__file__).parent


def make_signal(sample_rate: int = 16_000, seconds: float = 1.0) -> list[float]:
    """生成一个随时间变化的合成信号，模拟两个音段。"""
    signal = []
    for n in range(int(sample_rate * seconds)):
        t = n / sample_rate
        # 0~0.5 秒是 440 Hz，0.5~1 秒是 660 Hz；再加一点谐波。
        f = 440.0 if t < 0.5 else 660.0
        value = 0.6 * math.sin(2 * math.pi * f * t)
        value += 0.2 * math.sin(2 * math.pi * 2 * f * t)
        signal.append(value)
    return signal


def hann_window(length: int) -> list[float]:
    """Hann 窗：让一帧的两端平滑降到 0，减少频谱泄漏。"""
    if length == 1:
        return [1.0]
    return [0.5 - 0.5 * math.cos(2 * math.pi * n / (length - 1))
            for n in range(length)]


def fft(x: list[complex]) -> list[complex]:
    """递归 radix-2 FFT；输入长度必须是 2 的幂。"""
    n = len(x)
    if n == 1:
        return x
    if n & (n - 1):
        raise ValueError("FFT 输入长度必须是 2 的幂")
    even = fft(x[0::2])
    odd = fft(x[1::2])
    result = [0j] * n
    for k in range(n // 2):
        angle = -2 * math.pi * k / n
        twiddle = complex(math.cos(angle), math.sin(angle)) * odd[k]
        result[k] = even[k] + twiddle
        result[k + n // 2] = even[k] - twiddle
    return result


def frame_signal(signal: list[float], frame_size: int, hop_size: int) -> list[list[float]]:
    """切帧；最后不足一帧的部分用 0 补齐。"""
    frames = []
    for start in range(0, len(signal), hop_size):
        frame = signal[start:start + frame_size]
        if not frame:
            break
        frames.append(frame + [0.0] * (frame_size - len(frame)))
        if start + frame_size >= len(signal):
            break
    return frames


def power_spectrum(frame: list[float], fft_size: int) -> list[float]:
    """加窗、补零、FFT，并保留非负频率的功率。"""
    window = hann_window(len(frame))
    padded = [complex(v * w, 0.0) for v, w in zip(frame, window)]
    padded += [0j] * (fft_size - len(padded))
    spectrum = fft(padded)
    return [abs(z) ** 2 / fft_size for z in spectrum[:fft_size // 2 + 1]]


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def log_mel(power: list[float], sample_rate: int, fft_size: int,
            mel_bins: int = 40, low_hz: float = 0.0,
            high_hz: float | None = None) -> list[float]:
    """三角 Mel 滤波器组 + log 能量。"""
    high_hz = high_hz or sample_rate / 2
    mel_points = [hz_to_mel(low_hz) + i * (hz_to_mel(high_hz) - hz_to_mel(low_hz))
                  / (mel_bins + 1) for i in range(mel_bins + 2)]
    bins = [int(math.floor((fft_size + 1) * mel_to_hz(m) / sample_rate))
            for m in mel_points]
    result = []
    for i in range(1, mel_bins + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        energy = 0.0
        for k in range(left, min(center, len(power))):
            if center > left:
                energy += power[k] * (k - left) / (center - left)
        for k in range(center, min(right, len(power))):
            if right > center:
                energy += power[k] * (right - k) / (right - center)
        result.append(math.log(max(energy, 1e-12)))
    return result


def write_csv(path: Path, rows: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def main() -> None:
    sample_rate = 16_000
    frame_ms, hop_ms = 25, 10
    frame_size = sample_rate * frame_ms // 1000  # 400 个采样点
    hop_size = sample_rate * hop_ms // 1000       # 160 个采样点
    fft_size = 512

    signal = make_signal(sample_rate)
    frames = frame_signal(signal, frame_size, hop_size)
    spectra = [power_spectrum(frame, fft_size) for frame in frames]
    logmels = [log_mel(s, sample_rate, fft_size) for s in spectra]

    write_csv(ROOT / "demo_signal.csv", [[i / sample_rate, value]
                                         for i, value in enumerate(signal)])
    write_csv(ROOT / "spectrum_frame.csv",
              [[k * sample_rate / fft_size, value]
               for k, value in enumerate(spectra[25])])
    write_csv(ROOT / "logmel.csv", logmels)

    print(f"采样率: {sample_rate} Hz -> 每秒 {sample_rate} 个采样点")
    print(f"帧长: {frame_ms} ms = {frame_size} 点，帧移: {hop_ms} ms = {hop_size} 点")
    print(f"音频长度: {len(signal) / sample_rate:.1f} 秒，帧数: {len(frames)}")
    print(f"FFT: {fft_size} 点，非负频率 bin: {len(spectra[0])}")
    print(f"Log-Mel 形状: {len(logmels)} x {len(logmels[0])}")
    print("已写出 demo_signal.csv、spectrum_frame.csv、logmel.csv")


if __name__ == "__main__":
    main()
