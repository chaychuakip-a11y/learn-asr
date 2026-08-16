"""观察矩形窗与 Hann 窗对频谱泄漏的影响。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
OUTPUT = ROOT / "outputs" / "window_comparison.png"


def main() -> None:
    sr = 16_000
    n = 400                 # 25 ms
    n_fft = 4096            # 只增加频谱采样密度
    tone_hz = 440.0         # 故意不让它正好落在 400 点 FFT bin 上
    t = np.arange(n) / sr
    signal = np.sin(2 * np.pi * tone_hz * t)
    rectangular = np.ones(n)
    hann = np.hanning(n)

    def spectrum(window):
        x = signal * window
        y = np.fft.rfft(x, n=n_fft)
        db = 20 * np.log10(np.maximum(np.abs(y) / np.max(np.abs(y)), 1e-8))
        return db

    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    rect_db, hann_db = spectrum(rectangular), spectrum(hann)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(rectangular, label="Rectangular window")
    axes[0].plot(hann, label="Hann window")
    axes[0].set(title="Window shape", xlabel="Sample index", ylabel="Weight")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(freqs, rect_db, label="Rectangular")
    axes[1].plot(freqs, hann_db, label="Hann")
    axes[1].set(title="Spectral leakage: 440 Hz tone", xlabel="Frequency (Hz)", ylabel="Relative level (dB)")
    axes[1].set_xlim(0, 2000)
    axes[1].set_ylim(-100, 5)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.savefig(OUTPUT, dpi=150)
    print(f"图片已保存: {OUTPUT}")


if __name__ == "__main__":
    main()
