from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf
import torch

from acoustic_engine.features import LogMelFrontend
from acoustic_engine.streaming import StreamingAcousticEngine
from fsdd_generalization.training import (
    FeatureExample,
    SequenceSpec,
    build_sequence_specs,
    edit_distance,
    evaluate_model,
)
from .data import (
    AUDIOMNIST_REVISION,
    AudioMnistRecording,
    SpeakerMetadata,
    prepare_audiomnist,
    sha256,
)
from .final_fit import PREREGISTRATION_COMMIT
from .protocol import load_protocol, protocol_sha256


FROZEN_MODEL_COMMIT = "cac8aa682ed9f630623b9b627387cb651855b2f2"


def edit_operation_counts(reference: str, hypothesis: str) -> dict[str, int]:
    """Return one deterministic minimum-edit alignment's S/D/I counts."""
    n, m = len(reference), len(hypothesis)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0], back[i][0] = i, "deletion"
    for j in range(1, m + 1):
        costs[0][j], back[0][j] = j, "insertion"
    priority = {"correct": 0, "substitution": 1, "deletion": 2, "insertion": 3}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = "correct" if reference[i - 1] == hypothesis[j - 1] else "substitution"
            choices = [
                (costs[i - 1][j - 1] + (diagonal != "correct"), diagonal),
                (costs[i - 1][j] + 1, "deletion"),
                (costs[i][j - 1] + 1, "insertion"),
            ]
            costs[i][j], back[i][j] = min(
                choices, key=lambda item: (item[0], priority[item[1]])
            )
    counts = {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0}
    i, j = n, m
    while i or j:
        operation = back[i][j]
        if operation is None:
            raise AssertionError("edit backtrace ended before the origin")
        counts[operation] += 1
        if operation in {"correct", "substitution"}:
            i -= 1
            j -= 1
        elif operation == "deletion":
            i -= 1
        else:
            j -= 1
    if counts["substitution"] + counts["deletion"] + counts["insertion"] != costs[n][m]:
        raise AssertionError("edit operation counts disagree with edit distance")
    return counts


@lru_cache(maxsize=1_024)
def _resampled_waveform(path_text: str) -> torch.Tensor:
    audio, sample_rate = sf.read(path_text, dtype="float32")
    if sample_rate != 48_000 or audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"invalid frozen AudioMNIST waveform contract: {path_text}")
    resampled = librosa.resample(
        audio,
        orig_sr=48_000,
        target_sr=8_000,
        res_type="soxr_hq",
    ).astype(np.float32, copy=False)
    if resampled.size == 0 or not np.isfinite(resampled).all():
        raise ValueError(f"resampling produced invalid audio: {path_text}")
    return torch.from_numpy(resampled)


def load_external_waveform(recording: AudioMnistRecording) -> torch.Tensor:
    return _resampled_waveform(str(recording.path.resolve()))


def _decorate_evaluation(evaluation: dict[str, object]) -> dict[str, object]:
    predictions = []
    total_operations = {"correct": 0, "substitution": 0, "deletion": 0, "insertion": 0}
    for prediction in evaluation["predictions"]:  # type: ignore[index]
        row = dict(prediction)
        operations = edit_operation_counts(str(row["reference"]), str(row["hypothesis"]))
        row["edit_operations"] = operations
        predictions.append(row)
        for key, value in operations.items():
            total_operations[key] += value
    decorated = dict(evaluation)
    decorated["predictions"] = predictions
    decorated["edit_operations"] = total_operations
    return decorated


def _single_examples(
    rows: Iterable[AudioMnistRecording], frontend: LogMelFrontend
) -> list[FeatureExample]:
    examples = []
    for row in rows:
        waveform = load_external_waveform(row)
        examples.append(
            FeatureExample(
                features=frontend(waveform),
                targets=(int(row.digit) + 1,),
                text=row.digit,
                speaker=row.speaker,
                name=row.relative_name,
                augmented=False,
            )
        )
    return examples


def build_external_sequence_specs(
    recordings: list[AudioMnistRecording],
    count: int = 6_000,
    seed: int = 20_260_819,
) -> list[SequenceSpec]:
    specs = build_sequence_specs(
        recordings,
        count=count,
        min_digits=1,
        max_digits=4,
        seed=seed,
        prefix="audiomnist-external",
    )
    counts = defaultdict(int)
    for spec in specs:
        counts[spec.speaker] += 1
        if {row.speaker for row in spec.recordings} != {spec.speaker}:
            raise AssertionError("external multi-digit sequence crossed speaker boundary")
    if set(counts.values()) != {100} or len(counts) != 60:
        raise AssertionError("external sequence protocol requires 100 sequences per speaker")
    return specs


def _sequence_example(spec: SequenceSpec, frontend: LogMelFrontend) -> FeatureExample:
    pieces = []
    for index, row in enumerate(spec.recordings):
        pieces.append(load_external_waveform(row))  # type: ignore[arg-type]
        if index < len(spec.silence_samples):
            pieces.append(torch.zeros(spec.silence_samples[index]))
    waveform = torch.cat(pieces)
    return FeatureExample(
        features=frontend(waveform),
        targets=tuple(int(character) + 1 for character in spec.text),
        text=spec.text,
        speaker=spec.speaker,
        name=spec.name,
        augmented=False,
    )


def _speaker_interval(values: list[float], bounded: bool = False) -> dict[str, float | int]:
    if len(values) != 60:
        raise ValueError("primary external interval requires all 60 speakers")
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values)
    t_critical_df59 = 2.0009953780882674
    half = t_critical_df59 * sample_std / math.sqrt(60)
    low, high = mean - half, mean + half
    if bounded:
        low, high = max(0.0, low), min(1.0, high)
    return {
        "speaker_count": 60,
        "mean": mean,
        "sample_std": sample_std,
        "ci95_low": low,
        "ci95_high": high,
        "method": "two-sided t interval over 60 speaker-level metrics (df=59)",
    }


def aggregate_speakers(per_speaker: dict[str, dict[str, object]]) -> dict[str, object]:
    if len(per_speaker) != 60:
        raise ValueError("aggregate requires exactly 60 AudioMNIST speakers")
    rows = list(per_speaker.values())
    errors = sum(int(row["errors"]) for row in rows)
    characters = sum(int(row["reference_characters"]) for row in rows)
    exact = sum(int(row["exact"]) for row in rows)
    total = sum(int(row["total"]) for row in rows)
    operations = {
        key: sum(int(row["edit_operations"][key]) for row in rows)  # type: ignore[index]
        for key in ("correct", "substitution", "deletion", "insertion")
    }
    return {
        "micro_cer": errors / characters,
        "micro_exact_rate": exact / total,
        "total_errors": errors,
        "total_reference_characters": characters,
        "total_exact": exact,
        "total_utterances": total,
        "edit_operations": operations,
        "speaker_macro_cer": _speaker_interval([float(row["cer"]) for row in rows]),
        "speaker_macro_exact_rate": _speaker_interval(
            [float(row["exact_rate"]) for row in rows], bounded=True
        ),
    }


def _slice_summary(
    speaker_ids: list[str], per_speaker: dict[str, dict[str, object]]
) -> dict[str, object]:
    rows = [per_speaker[speaker] for speaker in speaker_ids]
    errors = sum(int(row["errors"]) for row in rows)
    characters = sum(int(row["reference_characters"]) for row in rows)
    exact = sum(int(row["exact"]) for row in rows)
    total = sum(int(row["total"]) for row in rows)
    return {
        "speaker_count": len(rows),
        "speakers": speaker_ids,
        "micro_cer": errors / characters,
        "micro_exact_rate": exact / total,
        "speaker_macro_cer": statistics.fmean(float(row["cer"]) for row in rows),
        "speaker_macro_exact_rate": statistics.fmean(
            float(row["exact_rate"]) for row in rows
        ),
    }


def _age_band(age: int) -> str:
    if age < 30:
        return "under_30"
    if age < 40:
        return "30_to_39"
    return "40_and_over"


def metadata_slices(
    metadata: dict[str, SpeakerMetadata],
    per_speaker: dict[str, dict[str, object]],
) -> dict[str, object]:
    dimensions = {
        "gender": lambda row: row.gender,
        "native_speaker": lambda row: str(row.native_speaker).lower(),
        "age_band": lambda row: _age_band(row.age),
        "accent_casefolded": lambda row: row.accent.casefold(),
    }
    output: dict[str, object] = {
        "note": "descriptive slices only; no causal or population-fairness claim",
        "age_band_definition": {"under_30": "age < 30", "30_to_39": "30 <= age < 40", "40_and_over": "age >= 40"},
    }
    for dimension, key_function in dimensions.items():
        groups: dict[str, list[str]] = defaultdict(list)
        for speaker, row in metadata.items():
            groups[str(key_function(row))].append(speaker)
        output[dimension] = {
            key: _slice_summary(sorted(speakers), per_speaker)
            for key, speakers in sorted(groups.items())
        }
    return output


def _evaluate_single_track(
    recordings: list[AudioMnistRecording],
    engine: StreamingAcousticEngine,
) -> dict[str, object]:
    frontend = LogMelFrontend(engine.frontend_config)
    groups: dict[str, list[AudioMnistRecording]] = defaultdict(list)
    for row in recordings:
        groups[row.speaker].append(row)
    per_speaker = {}
    for position, speaker in enumerate(sorted(groups), start=1):
        examples = _single_examples(groups[speaker], frontend)
        evaluation = evaluate_model(
            engine.model, examples, engine.feature_mean, engine.feature_std
        ).to_dict()
        per_speaker[speaker] = _decorate_evaluation(evaluation)
        if position % 5 == 0:
            print(f"single-digit scored speakers: {position}/60", flush=True)
    return {"aggregate": aggregate_speakers(per_speaker), "per_speaker": per_speaker}


def _evaluate_sequence_track(
    specs: list[SequenceSpec],
    engine: StreamingAcousticEngine,
) -> dict[str, object]:
    frontend = LogMelFrontend(engine.frontend_config)
    groups: dict[str, list[SequenceSpec]] = defaultdict(list)
    for spec in specs:
        groups[spec.speaker].append(spec)
    per_speaker = {}
    sequence_sources = {}
    for position, speaker in enumerate(sorted(groups), start=1):
        speaker_specs = groups[speaker]
        examples = [_sequence_example(spec, frontend) for spec in speaker_specs]
        evaluation = evaluate_model(
            engine.model, examples, engine.feature_mean, engine.feature_std
        ).to_dict()
        per_speaker[speaker] = _decorate_evaluation(evaluation)
        sequence_sources.update(
            {
                spec.name: {
                    "speaker": spec.speaker,
                    "reference": spec.text,
                    "files": [row.relative_name for row in spec.recordings],  # type: ignore[attr-defined]
                    "silence_samples": list(spec.silence_samples),
                }
                for spec in speaker_specs
            }
        )
        _resampled_waveform.cache_clear()
        if position % 5 == 0:
            print(f"multi-digit scored speakers: {position}/60", flush=True)
    return {
        "aggregate": aggregate_speakers(per_speaker),
        "per_speaker": per_speaker,
        "sequence_sources": sequence_sources,
    }


def _streaming_checks(
    recordings: list[AudioMnistRecording], engine: StreamingAcousticEngine
) -> dict[str, object]:
    lookup = {(row.speaker, row.digit, row.index): row for row in recordings}
    chunk_sizes = [1, 137, 400, 800, 1600]
    rows = []
    for speaker in sorted({row.speaker for row in recordings}):
        recording = lookup[(speaker, "7", 0)]
        waveform = load_external_waveform(recording)
        outputs = {
            str(size): engine.recognize_waveform(waveform, chunk_samples=size).text
            for size in chunk_sizes
        }
        rows.append(
            {
                "speaker": speaker,
                "file": recording.relative_name,
                "reference": "7",
                "outputs": outputs,
                "invariant": len(set(outputs.values())) == 1,
            }
        )
    invariant_count = sum(bool(row["invariant"]) for row in rows)
    if invariant_count != 60:
        raise AssertionError(f"streaming chunk invariance failed for {60-invariant_count} speakers")
    return {
        "chunk_samples": chunk_sizes,
        "checked_speakers": 60,
        "invariant_speakers": invariant_count,
        "rows": rows,
        "interpretation": "implementation consistency only; not recognition correctness",
    }


def _performance_checks(
    recordings: list[AudioMnistRecording], engine: StreamingAcousticEngine
) -> dict[str, object]:
    recording = next(
        row for row in recordings if row.speaker == "01" and row.digit == "7" and row.index == 0
    )
    waveform = load_external_waveform(recording)
    duration = waveform.numel() / 8_000
    rows = []
    for chunk_size in (400, 800, 1600):
        for _ in range(5):
            engine.recognize_waveform(waveform, chunk_samples=chunk_size)
        times = []
        for _ in range(30):
            started = time.perf_counter()
            engine.recognize_waveform(waveform, chunk_samples=chunk_size)
            times.append(time.perf_counter() - started)
        rows.append(
            {
                "chunk_samples": chunk_size,
                "chunk_ms": 1000 * chunk_size / 8_000,
                "latency_ms_p50": 1000 * float(np.percentile(times, 50)),
                "latency_ms_p95": 1000 * float(np.percentile(times, 95)),
                "mean_rtf": statistics.fmean(times) / duration,
            }
        )
    return {
        "input_file": recording.relative_name,
        "audio_seconds": duration,
        "threads": torch.get_num_threads(),
        "warmups": 5,
        "measured_runs": 30,
        "scope": "preloaded, already-resampled waveform; disk I/O and resampling excluded",
        "rows": rows,
    }


def _interpretation_band(single_exact: float) -> str:
    if single_exact >= 0.70:
        return "adequate_transfer_for_toy_digit_task_not_production_asr"
    if single_exact >= 0.40:
        return "weak_transfer_requires_separately_designed_adaptation"
    return "external_generalization_failure"


def _atomic_json_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def run_external_evaluation(
    cache_dir: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(
            "AudioMNIST has already been scored; refusing to overwrite the first untouched result"
        )
    protocol = load_protocol()
    torch.set_num_threads(int(protocol["performance_checks"]["threads"]))
    dataset_root, recordings, metadata = prepare_audiomnist(
        cache_dir, validate_audio=False
    )
    if len(recordings) != 30_000 or len(metadata) != 60:
        raise ValueError("external evaluation requires the full validated AudioMNIST dataset")
    engine = StreamingAcousticEngine.load(checkpoint_path)
    if sha256(checkpoint_path) != "5c74192bfa1b3bbddbf614a08a60b592082e7af0981af8facd3370e84d80e337":
        raise ValueError("frozen model checkpoint SHA256 changed after preregistration")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    print("FIRST EXTERNAL SCORE STARTED:", started_at, flush=True)
    single = _evaluate_single_track(recordings, engine)
    single["metadata_slices"] = metadata_slices(metadata, single["per_speaker"])  # type: ignore[arg-type]
    _resampled_waveform.cache_clear()
    specs = build_external_sequence_specs(recordings)
    sequence = _evaluate_sequence_track(specs, engine)
    streaming = _streaming_checks(recordings, engine)
    performance = _performance_checks(recordings, engine)
    exact = float(single["aggregate"]["micro_exact_rate"])  # type: ignore[index]

    result: dict[str, object] = {
        "format_version": 1,
        "status": "completed_first_and_only_untouched_audiomnist_score",
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256(),
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "frozen_model_commit": FROZEN_MODEL_COMMIT,
        "frozen_checkpoint": checkpoint_path.name,
        "frozen_checkpoint_sha256": sha256(checkpoint_path),
        "dataset": "AudioMNIST",
        "dataset_revision": AUDIOMNIST_REVISION,
        "dataset_root_not_published": dataset_root.name,
        "dataset_manifest": manifest_path.name,
        "dataset_manifest_sha256": sha256(manifest_path),
        "preprocessing": protocol["external_preprocessing"],
        "decoder": protocol["decoder"],
        "single_digit": single,
        "multi_digit": sequence,
        "streaming": streaming,
        "performance": performance,
        "preregistered_interpretation_band": _interpretation_band(exact),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "librosa": librosa.__version__,
            "soundfile": sf.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "disclosure": protocol["stopping_and_disclosure"],
    }
    _atomic_json_write(output_path, result)
    print("FIRST EXTERNAL SCORE COMPLETE", flush=True)
    print(
        "single exact/CER:",
        f"{single['aggregate']['micro_exact_rate']:.2%}",  # type: ignore[index]
        f"{single['aggregate']['micro_cer']:.2%}",  # type: ignore[index]
        flush=True,
    )
    print(
        "multi exact/CER:",
        f"{sequence['aggregate']['micro_exact_rate']:.2%}",  # type: ignore[index]
        f"{sequence['aggregate']['micro_cer']:.2%}",  # type: ignore[index]
        flush=True,
    )
    print("result:", output_path, flush=True)
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_external_evaluation(
        root / ".local_data",
        root / "artifacts" / "fsdd_final_external_frozen.pt",
        root / "artifacts" / "audiomnist_external_manifest.json",
        root / "artifacts" / "audiomnist_external_results.json",
    )


if __name__ == "__main__":
    main()
