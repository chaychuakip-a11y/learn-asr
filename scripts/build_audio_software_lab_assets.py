from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "audio_software_lab"
SAMPLE_RATE = 16_000


def db_to_amplitude(dbfs: float) -> float:
    return 10 ** (dbfs / 20.0)


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(signal, dtype=np.float64) ** 2)))


def dbfs(value: float, floor: float = -160.0) -> float:
    if value <= 0:
        return floor
    return float(20 * np.log10(value))


def describe(signal: np.ndarray, sample_rate: int) -> dict[str, object]:
    values = np.asarray(signal, dtype=np.float64)
    time_axis = 0 if values.ndim == 1 else 0
    peak = float(np.max(np.abs(values)))
    channel_count = 1 if values.ndim == 1 else values.shape[1]
    return {
        "sample_rate_hz": sample_rate,
        "channels": channel_count,
        "samples_per_channel": int(values.shape[time_axis]),
        "duration_seconds": values.shape[time_axis] / sample_rate,
        "peak": peak,
        "peak_dbfs": dbfs(peak),
        "rms": rms(values),
        "rms_dbfs": dbfs(rms(values)),
        "dc_mean": float(np.mean(values)),
        "clipped_ratio_at_0_999": float(np.mean(np.abs(values) >= 0.999)),
    }


def write(name: str, signal: np.ndarray, sample_rate: int, notes: str) -> dict[str, object]:
    path = OUTPUT_DIR / name
    sf.write(path, signal.astype(np.float32), sample_rate, subtype="FLOAT")
    report = describe(signal, sample_rate)
    report.update({"file": name, "notes": notes})
    return report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, object]] = []

    t = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
    amplitude = db_to_amplitude(-12.0)
    tone_440 = amplitude * np.sin(2 * np.pi * 440 * t)
    report = write(
        "01_calibration_440hz_peak_minus12dbfs.wav",
        tone_440,
        SAMPLE_RATE,
        "1 s mono float WAV; 440 Hz sine; peak -12 dBFS; sine RMS is about -15.01 dBFS.",
    )
    report["expected_dominant_frequencies_hz"] = [440]
    assets.append(report)

    two_tones = 0.35 * np.sin(2 * np.pi * 440 * t) + 0.18 * np.sin(2 * np.pi * 1000 * t)
    report = write(
        "02_two_tones_440hz_1000hz.wav",
        two_tones,
        SAMPLE_RATE,
        "Two stationary tones for FFT size, window, and spectrogram comparison.",
    )
    report["expected_dominant_frequencies_hz"] = [440, 1000]
    assets.append(report)

    dc_signal = 0.25 * np.sin(2 * np.pi * 220 * t) + 0.12
    report = write(
        "03_dc_offset_220hz.wav",
        dc_signal,
        SAMPLE_RATE,
        "Waveform centre is shifted upward by a DC mean of about +0.12.",
    )
    report["expected_dominant_frequencies_hz"] = [0, 220]
    assets.append(report)

    unclipped = 1.8 * np.sin(2 * np.pi * 440 * t)
    clipped = np.clip(unclipped, -1.0, 1.0)
    report = write(
        "04_hard_clipped_440hz.wav",
        clipped,
        SAMPLE_RATE,
        "Flat waveform tops and extra harmonics demonstrate hard clipping.",
    )
    report["expected_dominant_frequencies_hz"] = [440]
    report["expected_extra_content"] = "odd harmonics caused by clipping"
    assets.append(report)

    source_audio, source_rate = sf.read(ROOT / "data" / "0_jackson_0.wav", dtype="float32")
    if source_audio.ndim != 1:
        source_audio = source_audio.mean(axis=1)
    speech = librosa.resample(source_audio, orig_sr=source_rate, target_sr=SAMPLE_RATE)
    rng = np.random.default_rng(20260818)
    raw_noise = rng.normal(size=len(speech))
    target_snr_db = 10.0
    noise_scale = math.sqrt(
        (rms(speech) ** 2 / (10 ** (target_snr_db / 10.0))) / (rms(raw_noise) ** 2)
    )
    noise = raw_noise * noise_scale
    noise_only = rng.normal(scale=rms(noise), size=round(0.5 * SAMPLE_RATE))
    noisy_speech = np.concatenate([noise_only, speech + noise])
    report = write(
        "05_noise_profile_then_speech_snr10db.wav",
        noisy_speech,
        SAMPLE_RATE,
        "First 0.5 s is noise-only; remaining region is speech mixed at measured 10 dB SNR.",
    )
    report["noise_only_seconds"] = 0.5
    report["speech_region_target_snr_db"] = target_snr_db
    report["speech_region_measured_snr_db"] = 10 * math.log10(
        rms(speech) ** 2 / rms(noise) ** 2
    )
    assets.append(report)

    short_t = np.arange(round(1.0 * SAMPLE_RATE)) / SAMPLE_RATE
    left = 0.5 * np.sin(2 * np.pi * 600 * short_t)
    delay_samples = 16
    right = np.concatenate([np.zeros(delay_samples), left[:-delay_samples]])
    stereo = np.stack([left, right], axis=1)
    report = write(
        "06_stereo_right_delayed_1ms.wav",
        stereo,
        SAMPLE_RATE,
        "Stereo test: right channel is delayed by 16 samples, exactly 1 ms at 16 kHz.",
    )
    report["right_channel_delay_samples"] = delay_samples
    report["right_channel_delay_ms"] = delay_samples / SAMPLE_RATE * 1000
    assets.append(report)

    speech_copy = OUTPUT_DIR / "07_real_speech_digit_zero_8khz.wav"
    shutil.copyfile(ROOT / "data" / "0_jackson_0.wav", speech_copy)
    report = describe(source_audio, source_rate)
    report.update(
        {
            "file": speech_copy.name,
            "notes": "Real FSDD utterance for Praat pitch, formant, intensity, and TextGrid practice.",
        }
    )
    assets.append(report)

    manifest = {
        "format_version": 1,
        "purpose": "Deterministic reference audio for GUI audio-analysis software labs.",
        "important": [
            "Measurements may differ slightly because tools use different windows, scaling, channel rules, and dB references.",
            "Praat intensity is not a calibrated physical SPL measurement unless the recording chain is calibrated.",
            "Always preserve the original file and perform destructive edits only on a copy.",
        ],
        "assets": assets,
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(assets)} WAV files and manifest to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
