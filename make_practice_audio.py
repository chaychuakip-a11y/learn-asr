"""把 FSDD 的 0~9 人声样本拼成练习音频。"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).parent
PARTS = ROOT / "data" / "spoken_digits_parts"
OUT = ROOT / "data"


def main() -> None:
    sample_rate = 8_000
    silence = np.zeros(round(0.20 * sample_rate), dtype=np.float32)
    pieces: list[np.ndarray] = []
    boundaries: list[tuple[int, float, float]] = []
    cursor = 0

    for digit in range(10):
        audio, sr = sf.read(PARTS / f"{digit}_jackson_0.wav", dtype="float32")
        if sr != sample_rate:
            raise ValueError(f"digit {digit}: expected {sample_rate} Hz, got {sr}")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        start = cursor / sample_rate
        end = (cursor + len(audio)) / sample_rate
        boundaries.append((digit, start, end))
        pieces.extend([audio, silence])
        cursor += len(audio) + len(silence)

    joined_8k = np.concatenate(pieces[:-1])
    joined_16k = librosa.resample(joined_8k, orig_sr=sample_rate, target_sr=16_000)
    sf.write(OUT / "spoken_digits_0_to_9_8k.wav", joined_8k, sample_rate)
    sf.write(OUT / "spoken_digits_0_to_9_16k.wav", joined_16k, 16_000)

    lines = ["transcript: zero one two three four five six seven eight nine", ""]
    lines += [f"{digit}\t{start:.3f}\t{end:.3f}" for digit, start, end in boundaries]
    (OUT / "spoken_digits_0_to_9.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"8 kHz duration: {len(joined_8k) / sample_rate:.3f} s")
    print(f"16 kHz duration: {len(joined_16k) / 16_000:.3f} s")
    print("Created spoken_digits_0_to_9_8k.wav and spoken_digits_0_to_9_16k.wav")


if __name__ == "__main__":
    main()
