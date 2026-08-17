from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import statistics
import time

from .features import load_mono_audio
from .streaming import StreamingAcousticEngine


@dataclass(frozen=True)
class BenchmarkSummary:
    mean_seconds: float
    p50_seconds: float
    p95_seconds: float
    real_time_factor: float


def summarize(latencies: list[float], audio_seconds: float) -> BenchmarkSummary:
    if not latencies or audio_seconds <= 0:
        raise ValueError("latencies and positive audio duration are required")
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered) + 0.999999) - 1))
    mean = statistics.fmean(ordered)
    return BenchmarkSummary(
        mean_seconds=mean,
        p50_seconds=statistics.median(ordered),
        p95_seconds=ordered[p95_index],
        real_time_factor=mean / audio_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the streaming teaching engine")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--chunk-samples", type=int, nargs="+", default=[400, 800, 1600])
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    engine = StreamingAcousticEngine.load(args.checkpoint)
    waveforms = [
        load_mono_audio(path, engine.frontend_config.sample_rate)
        for path in args.audio
    ]
    audio_seconds = sum(waveform.numel() for waveform in waveforms) / engine.frontend_config.sample_rate
    for chunk_samples in args.chunk_samples:
        if chunk_samples <= 0:
            parser.error("every --chunk-samples value must be positive")
        for waveform in waveforms:
            engine.recognize_waveform(waveform, chunk_samples=chunk_samples)
        latencies = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            for waveform in waveforms:
                engine.recognize_waveform(waveform, chunk_samples=chunk_samples)
            latencies.append(time.perf_counter() - started)
        summary = summarize(latencies, audio_seconds)
        chunk_ms = chunk_samples / engine.frontend_config.sample_rate * 1000
        print(
            f"chunk={chunk_samples} samples ({chunk_ms:.1f} ms) "
            f"mean={summary.mean_seconds * 1000:.2f} ms "
            f"p50={summary.p50_seconds * 1000:.2f} ms "
            f"p95={summary.p95_seconds * 1000:.2f} ms "
            f"RTF={summary.real_time_factor:.4f}"
        )


if __name__ == "__main__":
    main()
