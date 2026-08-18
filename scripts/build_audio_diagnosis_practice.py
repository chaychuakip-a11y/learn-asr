from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "audio_diagnosis_lab"
SOURCE_PATH = ROOT / "data" / "spoken_digits_0_to_9_16k.wav"
SEED = 20260818


def rms(signal: np.ndarray) -> float:
    values = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2)))


def peak(signal: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(signal))))


def dbfs(value: float) -> float:
    return float(20 * np.log10(max(float(value), 1e-12)))


def describe(signal: np.ndarray, sample_rate: int) -> dict[str, object]:
    values = np.asarray(signal, dtype=np.float64)
    channels = 1 if values.ndim == 1 else values.shape[1]
    frames = values.shape[0]
    return {
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "samples_per_channel": frames,
        "duration_seconds": frames / sample_rate,
        "peak": peak(values),
        "peak_dbfs": dbfs(peak(values)),
        "rms": rms(values),
        "rms_dbfs": dbfs(rms(values)),
        "dc_mean": float(np.mean(values)),
        "clipped_ratio_at_0_999": float(np.mean(np.abs(values) >= 0.999)),
    }


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    scale = math.sqrt((rms(clean) ** 2 / (10 ** (snr_db / 10))) / (rms(noise) ** 2))
    return clean + noise * scale


def add_echo(signal: np.ndarray, delay_samples: int, gain: float) -> np.ndarray:
    result = signal.copy()
    result[delay_samples:] += gain * signal[:-delay_samples]
    return result


def filtered(signal: np.ndarray, sample_rate: int, kind: str, cutoff_hz: float) -> np.ndarray:
    sos = butter(6, cutoff_hz, btype=kind, fs=sample_rate, output="sos")
    return sosfiltfilt(sos, signal).astype(np.float64)


def quantize(signal: np.ndarray, bits: int) -> np.ndarray:
    levels = 2**bits
    return np.round((np.clip(signal, -1, 1) + 1) * (levels - 1) / 2) * 2 / (levels - 1) - 1


def build_cases(clean: np.ndarray, sample_rate: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(SEED)
    t = np.arange(len(clean)) / sample_rate
    cases: list[dict[str, object]] = []

    def add(
        issue: str,
        signal: np.ndarray,
        *,
        output_rate: int = sample_rate,
        severity: str,
        audible: str,
        visible: str,
        first_checks: list[str],
        explanation: str,
    ) -> None:
        cases.append(
            {
                "issue": issue,
                "signal": np.asarray(signal, dtype=np.float64),
                "output_rate": output_rate,
                "severity": severity,
                "audible": audible,
                "visible": visible,
                "first_checks": first_checks,
                "explanation": explanation,
            }
        )

    add(
        "clean",
        clean,
        severity="none",
        audible="natural reference speech",
        visible="centred waveform with headroom; no persistent narrow spectral line",
        first_checks=["metadata", "peak/RMS", "waveform", "spectrogram"],
        explanation="This is a clean control. A diagnosis workflow must allow the answer 'no obvious injected fault'.",
    )
    add(
        "low_level",
        clean * 0.08,
        severity="strong",
        audible="very quiet but otherwise recognisable",
        visible="waveform remains centred and undistorted but peak/RMS are about 21.9 dB lower",
        first_checks=["peak dBFS", "RMS dBFS", "compare shape with reference"],
        explanation="Gain is too low. Raising it later also raises the recorded noise floor; low level is not the same as silence.",
    )
    add(
        "hard_clipping",
        np.clip(clean * 4.5, -1, 1),
        severity="strong",
        audible="harsh distortion",
        visible="flat tops at both +1 and -1; strong added harmonics; clipped ratio above zero",
        first_checks=["waveform flat tops", "clipped samples", "spectrum harmonics"],
        explanation="The signal exceeded digital full scale and was hard-clipped. Turning it down cannot reconstruct the lost shape.",
    )
    add(
        "asymmetric_clipping",
        np.clip(clean * 2.8, -0.85, 0.32),
        severity="strong",
        audible="distorted with asymmetric waveform character",
        visible="positive peaks flatten near +0.32 while negative peaks reach about -0.85; DC may shift",
        first_checks=["compare positive/negative limits", "DC mean", "spectrum"],
        explanation="Only one side clips early, often indicating biased analogue electronics or asymmetric processing.",
    )
    add(
        "dc_offset",
        clean * 0.7 + 0.18,
        severity="medium",
        audible="often subtle; possible clicks when editing boundaries",
        visible="waveform centre is shifted upward; non-zero mean; energy near 0 Hz",
        first_checks=["waveform centre", "DC mean", "0 Hz spectrum region"],
        explanation="A constant offset consumes headroom and can create clicks at cuts, but it is not useful speech energy.",
    )
    hum = 0.08 * np.sin(2 * np.pi * 50 * t) + 0.035 * np.sin(2 * np.pi * 100 * t)
    add(
        "mains_hum_50hz",
        clean + hum,
        severity="medium",
        audible="low continuous hum",
        visible="persistent horizontal spectral lines at 50 Hz and 100 Hz",
        first_checks=["long-window spectrum", "spectrogram low frequencies", "silence regions"],
        explanation="A mains-frequency fundamental plus harmonic was added. Check the region frequency used by the local power grid before assuming 50 or 60 Hz.",
    )
    add(
        "broadband_white_noise",
        mix_at_snr(clean, rng.normal(size=len(clean)), 5.0),
        severity="strong",
        audible="steady hiss over speech",
        visible="raised broadband noise floor across most frequencies and silent gaps",
        first_checks=["silence-region spectrum", "spectrogram background", "estimate SNR"],
        explanation="White noise was mixed at 5 dB SNR. It is broadband, unlike a narrow hum or whistle.",
    )
    whistle = 0.055 * np.sin(2 * np.pi * 2800 * t)
    add(
        "narrowband_whistle_2800hz",
        clean + whistle,
        severity="medium",
        audible="constant high-pitched whistle",
        visible="thin horizontal line at 2800 Hz throughout the spectrogram",
        first_checks=["spectrogram horizontal line", "Plot Spectrum peak", "select silence"],
        explanation="A stationary narrowband interferer is better diagnosed by its exact frequency than by a generic noise label.",
    )
    clicked = clean.copy()
    click_positions = [round(v * sample_rate) for v in (0.35, 0.88, 1.42, 2.15, 2.72)]
    for position in click_positions:
        if position < len(clicked):
            clicked[position : position + 2] = [0.98, -0.98]
    add(
        "impulsive_clicks",
        clicked,
        severity="medium",
        audible="several short sharp clicks",
        visible="isolated near-vertical impulses; broadband vertical stripes in spectrogram",
        first_checks=["zoom waveform", "spectrogram vertical stripes", "time positions"],
        explanation="Clicks are short in time and therefore broad in frequency. They should not be treated like steady background noise.",
    )
    dropout = clean.copy()
    dropout_ranges = [(0.75, 0.84), (1.65, 1.78), (2.45, 2.52)]
    for start, end in dropout_ranges:
        dropout[round(start * sample_rate) : round(end * sample_rate)] = 0
    add(
        "dropouts",
        dropout,
        severity="strong",
        audible="brief unnatural gaps",
        visible="rectangular zero-valued gaps cutting through speech",
        first_checks=["waveform zero gaps", "gap durations", "spectrogram missing bands"],
        explanation="Samples were replaced by zeros. This differs from natural silence because the transitions are abrupt and can cut through words.",
    )
    add(
        "single_echo_120ms",
        add_echo(clean, round(0.12 * sample_rate), 0.48),
        severity="strong",
        audible="distinct delayed repetition",
        visible="similar waveform pattern repeats 120 ms later; comb-like spectrum",
        first_checks=["listen for repeat", "autocorrelation", "compare patterns 120 ms apart"],
        explanation="A single delayed copy was mixed at 120 ms. It is an echo, not a long diffuse reverberation tail.",
    )
    reverberant = clean.copy()
    for delay_ms, gain in [(35, 0.45), (61, 0.32), (93, 0.23), (141, 0.16), (210, 0.10)]:
        reverberant = add_echo(reverberant, round(delay_ms / 1000 * sample_rate), gain)
    reverberant *= 0.85 / max(peak(reverberant), 0.85)
    add(
        "reverberation",
        reverberant,
        severity="strong",
        audible="smeared room-like tail rather than one repeat",
        visible="energy persists after speech events; onsets and gaps are smeared",
        first_checks=["spectrogram decay tails", "speech gap energy", "compare with single echo"],
        explanation="Several decaying delayed copies approximate reverberation. The main clue is distributed decay, not one clearly separated repeat.",
    )
    add(
        "lowpass_muffled",
        filtered(clean, sample_rate, "lowpass", 1200),
        severity="strong",
        audible="muffled, missing consonant brightness",
        visible="strong attenuation above about 1200 Hz",
        first_checks=["spectrogram high-frequency loss", "spectrum vs reference", "listen to consonants"],
        explanation="An overly aggressive low-pass filter removed high-frequency speech cues.",
    )
    add(
        "highpass_thin",
        filtered(clean, sample_rate, "highpass", 900),
        severity="strong",
        audible="thin or telephone-like, missing low-frequency body",
        visible="strong attenuation below about 900 Hz",
        first_checks=["spectrum low-frequency loss", "spectrogram", "compare vowel energy"],
        explanation="An overly aggressive high-pass filter removed the low-frequency foundation and some vowel/formant information.",
    )
    add(
        "low_bit_quantization",
        quantize(clean, 5),
        severity="strong",
        audible="grainy quantisation noise",
        visible="waveform uses only a small set of horizontal amplitude levels",
        first_checks=["zoom to samples", "count amplitude levels", "noise floor"],
        explanation="The signal was quantised to only 5 bits (32 levels). The staircase is an amplitude-resolution problem, not a low sample-rate problem.",
    )
    envelope = 0.25 + 0.75 * (0.5 + 0.5 * np.sin(2 * np.pi * 1.7 * t))
    add(
        "gain_pumping",
        clean * envelope,
        severity="medium",
        audible="volume repeatedly swells and falls",
        visible="slow periodic envelope at about 1.7 Hz",
        first_checks=["RMS over time", "waveform envelope", "listen for periodic gain"],
        explanation="A periodic gain envelope simulates pumping from poor AGC/compression. The underlying pitch need not change.",
    )
    add(
        "wrong_sample_rate_metadata",
        clean,
        output_rate=sample_rate // 2,
        severity="strong",
        audible="plays twice as long and about one octave lower",
        visible="header reports 8 kHz and duration doubles even though sample values match the 16 kHz reference",
        first_checks=["sample rate metadata", "duration", "pitch vs reference"],
        explanation="The same 16 kHz samples were labelled as 8 kHz without resampling. This is a time-axis metadata error, not a gain error.",
    )
    left_only = np.stack([clean, np.zeros_like(clean)], axis=1)
    add(
        "right_channel_silent",
        left_only,
        severity="strong",
        audible="plays from one side; mono downmix is quieter",
        visible="left waveform is present and right waveform is exactly zero",
        first_checks=["channel count", "separate channel waveforms", "per-channel RMS"],
        explanation="One stereo channel contains no signal. A combined statistic can hide which channel failed.",
    )
    inverted = np.stack([clean, -clean], axis=1)
    add(
        "stereo_polarity_inversion",
        inverted,
        severity="strong",
        audible="stereo may sound wide; equal-weight mono sum cancels nearly completely",
        visible="right waveform is the exact negative of the left",
        first_checks=["overlay channels", "correlation", "mono sum"],
        explanation="The channels have opposite polarity. Each channel alone is valid, but naive mono downmix can cancel the speech.",
    )
    delay_samples = 32
    delayed_right = np.concatenate([np.zeros(delay_samples), clean[:-delay_samples]])
    delayed_stereo = np.stack([clean, delayed_right], axis=1)
    add(
        "stereo_channel_delay_2ms",
        delayed_stereo,
        severity="medium",
        audible="subtle spatial/comb-filter effect when downmixed",
        visible="right channel events occur 32 samples (2 ms) later",
        first_checks=["zoom both channels", "cross-correlation", "mono spectrum"],
        explanation="A small inter-channel delay can damage mono downmix or beamforming even when both channels look individually clean.",
    )
    truncated = clean[round(0.42 * sample_rate) : -round(0.27 * sample_rate)]
    add(
        "truncated_boundaries",
        truncated,
        severity="strong",
        audible="speech starts and ends abruptly, losing content",
        visible="first/last events are cut with little or no surrounding context",
        first_checks=["duration vs reference", "start/end zoom", "transcript alignment"],
        explanation="The recording was cropped inside speech. Endpoint logic must preserve words and context, not merely minimise silence.",
    )
    decimated = clean[::4]
    naive_up = np.repeat(decimated, 4)[: len(clean)]
    add(
        "naive_resampling_aliasing",
        naive_up,
        severity="strong",
        audible="rough, aliased high-frequency distortion",
        visible="stair-step waveform and mirrored/incorrect spectral content",
        first_checks=["zoom samples", "spectrogram aliases", "compare with proper low-pass resampling"],
        explanation="The signal was downsampled by taking every fourth sample without anti-alias filtering, then repeated back to 16 kHz.",
    )
    compressed = np.tanh(5 * clean)
    compressed *= 0.88 / peak(compressed)
    add(
        "heavy_nonlinear_compression",
        compressed,
        severity="strong",
        audible="unnaturally dense and distorted",
        visible="quiet and loud parts become similar; rounded saturation; reduced crest factor; extra harmonics",
        first_checks=["crest factor", "waveform envelope", "spectrum harmonics"],
        explanation="Strong nonlinear saturation reduced dynamic range without producing exact full-scale flat tops.",
    )
    clean_stereo = np.stack([clean, clean], axis=1)
    add(
        "clean_stereo",
        clean_stereo,
        severity="none",
        audible="centred normal stereo copy",
        visible="two identical healthy channels",
        first_checks=["channels", "per-channel statistics", "correlation"],
        explanation="This is a second clean control. Two identical channels are not automatically a fault, though they carry redundant information.",
    )
    return cases


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source, sample_rate = sf.read(SOURCE_PATH, dtype="float64")
    if source.ndim != 1:
        source = source.mean(axis=1)
    duration_samples = min(len(source), round(3.2 * sample_rate))
    clean = source[:duration_samples]
    clean = clean / max(peak(clean), 1e-12) * 0.45

    sf.write(OUTPUT_DIR / "reference_clean_speech.wav", clean, sample_rate, subtype="PCM_16")
    cases = build_cases(clean, sample_rate)
    permutation = np.random.default_rng(SEED).permutation(len(cases))

    answer_cases=[]
    public_cases=[]
    beginner={"clean","low_level","hard_clipping","dc_offset","dropouts","wrong_sample_rate_metadata","right_channel_silent"}
    intermediate={"asymmetric_clipping","mains_hum_50hz","broadband_white_noise","narrowband_whistle_2800hz","impulsive_clicks","lowpass_muffled","highpass_thin","low_bit_quantization","truncated_boundaries"}
    for case_number, source_index in enumerate(permutation, start=1):
        item=cases[int(source_index)]
        case_id=f"case_{case_number:02d}"
        filename=f"{case_id}.wav"
        signal=np.asarray(item.pop("signal"))
        output_rate=int(item.pop("output_rate"))
        difficulty="beginner" if item["issue"] in beginner else "intermediate" if item["issue"] in intermediate else "advanced"
        sf.write(OUTPUT_DIR/filename,signal,output_rate,subtype="PCM_16")
        measurements=describe(signal,output_rate)
        answer={"case_id":case_id,"file":filename,"difficulty":difficulty,**item,"measurements":measurements}
        answer_cases.append(answer)
        public_cases.append(
            {
                "case_id":case_id,
                "file":filename,
                "difficulty":difficulty,
                "sha256_note":"Use the filename as the stable exercise ID; do not infer the answer from order.",
            }
        )

    public_manifest={
        "format_version":1,
        "reference":"reference_clean_speech.wav",
        "case_count":len(public_cases),
        "workflow":"Inspect metadata, listen, waveform, channels, spectrum, spectrogram, then state one primary diagnosis with evidence.",
        "cases":public_cases,
    }
    answer_key={
        "format_version":1,
        "seed":SEED,
        "source":"data/spoken_digits_0_to_9_16k.wav (FSDD-derived; see DATA_SOURCES.md)",
        "cases":answer_cases,
    }
    (OUTPUT_DIR/"manifest.json").write_text(json.dumps(public_manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUTPUT_DIR/"answer_key.json").write_text(json.dumps(answer_key,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"wrote reference + {len(cases)} blind cases to {OUTPUT_DIR}")


if __name__=="__main__":
    main()
