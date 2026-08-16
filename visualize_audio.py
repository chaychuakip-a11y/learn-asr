"""读取真实 WAV，并可视化 waveform / spectrum / STFT / Log-Mel。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


ROOT = Path(__file__).parent
INPUT = ROOT / "data" / "0_jackson_0.wav"
OUTPUT = ROOT / "outputs" / "audio_features.png"


def resample_linear(x: np.ndarray, old_sr: int, new_sr: int) -> np.ndarray:
    """教学用线性重采样；生产环境可换成 scipy.resample_poly。"""
    if old_sr == new_sr:
        return x
    old_t = np.arange(len(x)) / old_sr
    new_len = round(len(x) * new_sr / old_sr)
    new_t = np.arange(new_len) / new_sr
    return np.interp(new_t, old_t, x)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40) -> np.ndarray:
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    points = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(points) / sr).astype(int)
    bank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1:m + 2]
        if center > left:
            bank[m - 1, left:center] = np.arange(left, center) / (center - left)
        if right > center:
            bank[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return bank


def main() -> None:
    target_sr = 16_000
    frame_length = round(0.025 * target_sr)
    hop_length = round(0.010 * target_sr)
    n_fft = 512

    audio, sr = sf.read(INPUT)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = resample_linear(audio.astype(np.float64), sr, target_sr)

    # center=False：按真实时间从左到右切帧；每帧乘 Hann 窗。
    starts = np.arange(0, max(1, len(audio) - frame_length + 1), hop_length)
    frames = np.stack([
        np.pad(audio[start:start + frame_length], (0, max(0, frame_length - len(audio[start:start + frame_length]))))
        for start in starts
    ])
    windowed = frames * np.hanning(frame_length)[None, :]
    stft = np.fft.rfft(windowed, n=n_fft, axis=1)
    power = np.abs(stft) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / target_sr)
    times = starts / target_sr

    mel = mel_filterbank(target_sr, n_fft, 40) @ power.T
    log_mel = np.log(np.maximum(mel, 1e-10))

    OUTPUT.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(12, 13), constrained_layout=True)
    axes[0].plot(np.arange(len(audio)) / target_sr, audio, linewidth=0.7)
    axes[0].set(title=f"Waveform (original {sr} Hz, analyzed at {target_sr} Hz)", xlabel="Time (s)", ylabel="Amplitude")

    frame_index = min(20, len(power) - 1)
    spectrum_db = 10 * np.log10(np.maximum(power[frame_index], 1e-10))
    axes[1].plot(freqs, spectrum_db)
    axes[1].set(title=f"Power spectrum of frame {frame_index} (dB)", xlabel="Frequency (Hz)", ylabel="Power (dB)")
    axes[1].set_xlim(0, target_sr / 2)
    axes[1].grid(alpha=0.25)

    stft_db = 10 * np.log10(np.maximum(power.T, 1e-10))
    stft_image = axes[2].pcolormesh(times, freqs, stft_db, shading="auto", cmap="magma")
    axes[2].set(title="STFT spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")
    axes[2].set_ylim(0, 8000)
    fig.colorbar(stft_image, ax=axes[2], label="Power (dB)")

    logmel_image = axes[3].pcolormesh(times, np.arange(log_mel.shape[0]), log_mel, shading="auto", cmap="viridis")
    axes[3].set(title="Log-Mel spectrogram", xlabel="Time (s)", ylabel="Mel bin")
    fig.colorbar(logmel_image, ax=axes[3], label="Log energy")

    fig.savefig(OUTPUT, dpi=150)
    print(f"输入: {INPUT}")
    print(f"原始采样率: {sr} Hz; 分析采样率: {target_sr} Hz")
    print(f"波形采样点: {len(audio)}; STFT 形状: {power.shape}; Log-Mel 形状: {log_mel.shape}")
    print(f"图片已保存: {OUTPUT}")


if __name__ == "__main__":
    main()
