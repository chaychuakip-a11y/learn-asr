from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import copy
from functools import lru_cache
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence

from acoustic_engine.decoder import ctc_collapse
from acoustic_engine.features import LogMelConfig, LogMelFrontend
from acoustic_engine.model import CausalConvCTCAcousticModel, CausalConvCTCConfig
from acoustic_engine.streaming import StreamingAcousticEngine
from .data import Recording


DIGIT_TOKENS = tuple("0123456789")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 17
    epochs: int = 24
    batch_size: int = 64
    learning_rate: float = 2e-3
    hidden_dim: int = 64
    num_layers: int = 6
    kernel_size: int = 5
    augmentation_copies: int = 1
    patience: int = 7
    train_sequences: int = 3_000
    dev_sequences: int = 500
    test_sequences: int = 500
    min_digits: int = 1
    max_digits: int = 4


@dataclass(frozen=True)
class SequenceSpec:
    name: str
    speaker: str
    text: str
    recordings: tuple[Recording, ...]
    silence_samples: tuple[int, ...]


@dataclass(frozen=True)
class FeatureExample:
    features: torch.Tensor
    targets: tuple[int, ...]
    text: str
    speaker: str
    name: str
    augmented: bool


@dataclass(frozen=True)
class PreparedFeatures:
    """Clean features shared by comparable training candidates in one process."""

    frontend: LogMelFrontend
    config_signature: tuple[int, ...]
    recording_signature: tuple[tuple[str, ...], ...]
    train_specs: tuple[SequenceSpec, ...]
    train_clean: tuple[FeatureExample, ...]
    dev_specs: tuple[SequenceSpec, ...]
    dev_raw: tuple[FeatureExample, ...]
    test_specs: tuple[SequenceSpec, ...]
    test_raw: tuple[FeatureExample, ...]
    test_single_raw: tuple[FeatureExample, ...]


@dataclass(frozen=True)
class PreparedTestFeatures:
    """Outer-test features constructed only after candidate selection."""

    specs: tuple[SequenceSpec, ...]
    raw: tuple[FeatureExample, ...]
    single_raw: tuple[FeatureExample, ...]


@dataclass(frozen=True)
class Evaluation:
    exact: int
    total: int
    errors: int
    reference_characters: int
    cer: float
    exact_rate: float
    predictions: tuple[dict[str, object], ...]
    per_length: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def frontend_config() -> LogMelConfig:
    return LogMelConfig(
        sample_rate=8_000,
        n_fft=256,
        win_length=200,
        hop_length=80,
        n_mels=24,
        peak_normalize=False,
    )


@lru_cache(maxsize=4_096)
def _load_waveform_cached(path: str) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != 8_000 or audio.ndim != 1:
        raise ValueError(f"unexpected audio contract: {path}")
    return torch.from_numpy(audio)


def load_waveform(recording: Recording) -> torch.Tensor:
    # Callers never mutate this tensor; caching avoids thousands of repeated WAV reads in LOSO.
    return _load_waveform_cached(str(recording.path.resolve()))


def augment_waveform(waveform: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    gain_db = float(torch.empty(1).uniform_(-8.0, 8.0, generator=generator))
    output = waveform * (10 ** (gain_db / 20))
    target_snr_db = float(torch.empty(1).uniform_(5.0, 25.0, generator=generator))
    noise = torch.randn(output.shape, generator=generator, dtype=output.dtype)
    signal_rms = output.square().mean().sqrt().clamp_min(1e-6)
    noise_rms = noise.square().mean().sqrt().clamp_min(1e-6)
    noise_scale = signal_rms / (noise_rms * 10 ** (target_snr_db / 20))
    return (output + noise * noise_scale).clamp(-1, 1)


def build_sequence_specs(
    recordings: Iterable[Recording],
    count: int,
    min_digits: int,
    max_digits: int,
    seed: int,
    prefix: str,
) -> list[SequenceSpec]:
    if count <= 0 or min_digits <= 0 or max_digits < min_digits:
        raise ValueError("invalid sequence generation limits")
    groups: dict[str, dict[str, list[Recording]]] = defaultdict(lambda: defaultdict(list))
    for recording in recordings:
        groups[recording.speaker][recording.digit].append(recording)
    speakers = sorted(groups)
    if not speakers:
        raise ValueError("cannot build sequences from an empty recording set")
    for speaker in speakers:
        if set(groups[speaker]) != set(DIGIT_TOKENS):
            raise ValueError(f"speaker {speaker} does not cover every digit")

    rng = random.Random(seed)
    specs = []
    for sequence_index in range(count):
        speaker = speakers[sequence_index % len(speakers)]
        length = rng.randint(min_digits, max_digits)
        # Force some adjacent repeats so CTC blank handling is genuinely exercised.
        if sequence_index % 10 == 0 and length >= 2:
            first = rng.choice(DIGIT_TOKENS)
            digits = [first, first] + [rng.choice(DIGIT_TOKENS) for _ in range(length - 2)]
        else:
            digits = [rng.choice(DIGIT_TOKENS) for _ in range(length)]
        chosen = tuple(rng.choice(groups[speaker][digit]) for digit in digits)
        silences = tuple(rng.randint(320, 800) for _ in range(length - 1))
        specs.append(
            SequenceSpec(
                name=f"{prefix}-{sequence_index:05d}",
                speaker=speaker,
                text="".join(digits),
                recordings=chosen,
                silence_samples=silences,
            )
        )
    return specs


def sequence_waveform(spec: SequenceSpec) -> torch.Tensor:
    pieces = []
    for index, recording in enumerate(spec.recordings):
        pieces.append(load_waveform(recording))
        if index < len(spec.silence_samples):
            pieces.append(torch.zeros(spec.silence_samples[index]))
    return torch.cat(pieces)


def single_recording_specs(recordings: Iterable[Recording], prefix: str) -> list[SequenceSpec]:
    return [
        SequenceSpec(
            name=f"{prefix}-{recording.path.stem}",
            speaker=recording.speaker,
            text=recording.digit,
            recordings=(recording,),
            silence_samples=(),
        )
        for recording in recordings
    ]


def build_feature_examples(
    specs: Iterable[SequenceSpec],
    frontend: LogMelFrontend,
    augmentation_copies: int,
    seed: int,
) -> list[FeatureExample]:
    if augmentation_copies < 0:
        raise ValueError("augmentation_copies cannot be negative")
    examples = []
    for position, spec in enumerate(specs):
        waveform = sequence_waveform(spec)
        targets = tuple(int(digit) + 1 for digit in spec.text)
        examples.append(
            FeatureExample(frontend(waveform), targets, spec.text, spec.speaker, spec.name, False)
        )
        for copy_index in range(augmentation_copies):
            augmented = augment_waveform(waveform, seed + position * 997 + copy_index * 17)
            examples.append(
                FeatureExample(frontend(augmented), targets, spec.text, spec.speaker, spec.name, True)
            )
    return examples


def _config_signature(config: ExperimentConfig) -> tuple[int, ...]:
    # Optimizer/model/augmentation fields do not affect the shared clean examples.
    return (
        config.seed,
        config.train_sequences,
        config.dev_sequences,
        config.test_sequences,
        config.min_digits,
        config.max_digits,
    )


def _recording_signature(*splits: Iterable[Recording]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(str(row.path.resolve()) for row in split) for split in splits)


def prepare_experiment_features(
    train_recordings: list[Recording],
    dev_recordings: list[Recording],
    test_recordings: list[Recording],
    config: ExperimentConfig,
) -> PreparedFeatures:
    """Compute candidate-independent sequence definitions and clean features once."""
    frontend = LogMelFrontend(frontend_config())
    train_specs = build_sequence_specs(
        train_recordings, config.train_sequences, config.min_digits, config.max_digits,
        config.seed + 101, "train",
    )
    dev_specs = build_sequence_specs(
        dev_recordings, config.dev_sequences, config.min_digits, config.max_digits,
        config.seed + 202, "dev",
    )
    test_specs = (
        build_sequence_specs(
            test_recordings, config.test_sequences, config.min_digits, config.max_digits,
            config.seed + 303, "test",
        )
        if test_recordings
        else []
    )
    return PreparedFeatures(
        frontend=frontend,
        config_signature=_config_signature(config),
        recording_signature=_recording_signature(
            train_recordings, dev_recordings, test_recordings
        ),
        train_specs=tuple(train_specs),
        train_clean=tuple(build_feature_examples(train_specs, frontend, 0, config.seed)),
        dev_specs=tuple(dev_specs),
        dev_raw=tuple(build_feature_examples(dev_specs, frontend, 0, config.seed)),
        test_specs=tuple(test_specs),
        test_raw=tuple(build_feature_examples(test_specs, frontend, 0, config.seed)),
        test_single_raw=tuple(
            build_feature_examples(
                single_recording_specs(test_recordings, "test-single"),
                frontend,
                0,
                config.seed,
            )
        ) if test_recordings else (),
    )


def prepare_test_features(
    test_recordings: list[Recording],
    config: ExperimentConfig,
    frontend: LogMelFrontend | None = None,
) -> PreparedTestFeatures:
    """Create outer-test labels/features after the dev-only decision is frozen."""
    if not test_recordings:
        raise ValueError("test recordings cannot be empty")
    frontend = frontend or LogMelFrontend(frontend_config())
    specs = build_sequence_specs(
        test_recordings,
        config.test_sequences,
        config.min_digits,
        config.max_digits,
        config.seed + 303,
        "test",
    )
    return PreparedTestFeatures(
        specs=tuple(specs),
        raw=tuple(build_feature_examples(specs, frontend, 0, config.seed)),
        single_raw=tuple(
            build_feature_examples(
                single_recording_specs(test_recordings, "test-single"),
                frontend,
                0,
                config.seed,
            )
        ),
    )


def _training_examples(
    prepared: PreparedFeatures,
    augmentation_copies: int,
    seed: int,
) -> list[FeatureExample]:
    if augmentation_copies < 0:
        raise ValueError("augmentation_copies cannot be negative")
    examples = []
    for position, (spec, clean) in enumerate(
        zip(prepared.train_specs, prepared.train_clean, strict=True)
    ):
        examples.append(clean)
        if augmentation_copies == 0:
            continue
        waveform = sequence_waveform(spec)
        for copy_index in range(augmentation_copies):
            augmented = augment_waveform(
                waveform, seed + position * 997 + copy_index * 17
            )
            examples.append(
                FeatureExample(
                    prepared.frontend(augmented),
                    clean.targets,
                    clean.text,
                    clean.speaker,
                    clean.name,
                    True,
                )
            )
    return examples


def feature_statistics(examples: Iterable[FeatureExample]) -> tuple[torch.Tensor, torch.Tensor]:
    frames = torch.cat([example.features for example in examples], dim=0)
    return frames.mean(dim=0), frames.std(dim=0).clamp_min(1e-5)


def _normalized(
    examples: Iterable[FeatureExample], mean: torch.Tensor, std: torch.Tensor
) -> list[FeatureExample]:
    return [
        FeatureExample(
            (example.features - mean) / std,
            example.targets,
            example.text,
            example.speaker,
            example.name,
            example.augmented,
        )
        for example in examples
    ]


def _batches(examples: list[FeatureExample], batch_size: int, seed: int):
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        rows = [examples[index] for index in order[start:start+batch_size]]
        lengths = torch.tensor([len(row.features) for row in rows], dtype=torch.long)
        padded = pad_sequence([row.features for row in rows], batch_first=True)
        targets = torch.tensor(
            [token for row in rows for token in row.targets], dtype=torch.long
        )
        target_lengths = torch.tensor([len(row.targets) for row in rows], dtype=torch.long)
        yield padded, lengths, targets, target_lengths


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_character in enumerate(reference, start=1):
        current = [i]
        for j, hyp_character in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j-1] + (ref_character != hyp_character),
                )
            )
        previous = current
    return previous[-1]


@torch.inference_mode()
def evaluate_model(
    model: CausalConvCTCAcousticModel,
    examples: list[FeatureExample],
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int = 128,
) -> Evaluation:
    model.eval()
    normalized = _normalized(examples, mean, std)
    predictions = []
    for start in range(0, len(normalized), batch_size):
        rows = normalized[start:start+batch_size]
        lengths = torch.tensor([len(row.features) for row in rows], dtype=torch.long)
        padded = pad_sequence([row.features for row in rows], batch_first=True)
        logits, output_lengths = model(padded, lengths)
        for row, row_logits, length in zip(rows, logits, output_lengths, strict=True):
            text, _ = ctc_collapse(
                row_logits[:int(length)].argmax(dim=-1).tolist(), DIGIT_TOKENS
            )
            predictions.append(
                {
                    "utterance": row.name,
                    "speaker": row.speaker,
                    "reference": row.text,
                    "hypothesis": text,
                    "reference_length": len(row.text),
                    "correct": text == row.text,
                }
            )
    errors = sum(
        edit_distance(str(row["reference"]), str(row["hypothesis"])) for row in predictions
    )
    characters = sum(len(str(row["reference"])) for row in predictions)
    exact = sum(bool(row["correct"]) for row in predictions)
    per_length = {}
    for length in sorted({int(row["reference_length"]) for row in predictions}):
        rows = [row for row in predictions if row["reference_length"] == length]
        row_errors = sum(
            edit_distance(str(row["reference"]), str(row["hypothesis"])) for row in rows
        )
        row_characters = sum(len(str(row["reference"])) for row in rows)
        row_exact = sum(bool(row["correct"]) for row in rows)
        per_length[str(length)] = {
            "exact": row_exact,
            "total": len(rows),
            "exact_rate": row_exact / len(rows),
            "cer": row_errors / row_characters,
        }
    return Evaluation(
        exact=exact,
        total=len(predictions),
        errors=errors,
        reference_characters=characters,
        cer=errors / characters,
        exact_rate=exact / len(predictions),
        predictions=tuple(predictions),
        per_length=per_length,
    )


def run_training(
    train_recordings: list[Recording],
    dev_recordings: list[Recording],
    test_recordings: list[Recording],
    config: ExperimentConfig,
    output_checkpoint: Path | None = None,
    prepared: PreparedFeatures | None = None,
    evaluate_test: bool = True,
) -> dict[str, object]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(1)
    if prepared is None:
        prepared = prepare_experiment_features(
            train_recordings, dev_recordings, test_recordings, config
        )
    if prepared.config_signature != _config_signature(config):
        raise ValueError("prepared features do not match the sequence configuration")
    if prepared.recording_signature != _recording_signature(
        train_recordings, dev_recordings, test_recordings
    ):
        raise ValueError("prepared features do not match the recording splits")
    if evaluate_test and (not prepared.test_raw or not prepared.test_single_raw):
        raise ValueError("evaluate_test=True requires prepared test features")
    frontend = prepared.frontend
    train_specs = prepared.train_specs
    dev_specs = prepared.dev_specs
    test_specs = prepared.test_specs
    train_raw = _training_examples(prepared, config.augmentation_copies, config.seed)
    dev_raw = list(prepared.dev_raw)
    test_raw = list(prepared.test_raw)
    test_single_raw = list(prepared.test_single_raw)
    mean, std = feature_statistics(train_raw)
    train = _normalized(train_raw, mean, std)

    model_config = CausalConvCTCConfig(
        feature_dim=frontend.config.n_mels,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        kernel_size=config.kernel_size,
        num_classes=11,
    )
    model = CausalConvCTCAcousticModel(model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-4
    )
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    history = []
    best_state = None
    best_epoch = 0
    best_dev = (float("inf"), -1.0)
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for padded, lengths, targets, target_lengths in _batches(
            train, config.batch_size, config.seed + epoch
        ):
            optimizer.zero_grad()
            logits, output_lengths = model(padded, lengths)
            loss = criterion(
                logits.log_softmax(dim=-1).transpose(0, 1),
                targets,
                output_lengths,
                target_lengths,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite CTC loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        dev = evaluate_model(model, dev_raw, mean, std)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "dev_cer": dev.cer,
            "dev_exact_rate": dev.exact_rate,
        }
        history.append(row)
        criterion_key = (dev.cer, -dev.exact_rate)
        if criterion_key < best_dev:
            best_dev = criterion_key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"dev_CER={dev.cer:.2%} dev_exact={dev.exact_rate:.2%}",
            flush=True,
        )
        if stale_epochs >= config.patience:
            print(
                f"early stop: no dev improvement for {config.patience} epochs",
                flush=True,
            )
            break
    if best_state is None:
        raise AssertionError("training produced no checkpoint")
    model.load_state_dict(best_state)
    final_dev = evaluate_model(model, dev_raw, mean, std)
    # The best checkpoint is frozen by dev before any optional test evaluation.
    engine = StreamingAcousticEngine(frontend.config, model, DIGIT_TOKENS, mean, std)
    if output_checkpoint is not None:
        engine.save(output_checkpoint)
    result: dict[str, object] = {
        "config": asdict(config),
        "model_config": model_config.to_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_recordings": len(train_recordings),
        "train_sequences": len(train_specs),
        "train_feature_examples": len(train_raw),
        "dev_sequences": len(dev_specs),
        "test_sequences": len(test_specs),
        "best_epoch": best_epoch,
        "history": history,
        "dev": final_dev.to_dict(),
        "test_evaluated": evaluate_test,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "checkpoint": output_checkpoint.name if output_checkpoint is not None else None,
    }
    if evaluate_test:
        # Test labels are touched only after dev model selection is complete.
        result["test"] = evaluate_model(model, test_raw, mean, std).to_dict()
        result["test_single_digit"] = evaluate_model(
            model, test_single_raw, mean, std
        ).to_dict()
    return result


def evaluate_checkpoint_on_test(
    checkpoint: Path,
    prepared_test: PreparedTestFeatures,
) -> dict[str, object]:
    """Evaluate one already-selected checkpoint on an outer test fold."""
    engine = StreamingAcousticEngine.load(checkpoint)
    sequence = evaluate_model(
        engine.model,
        list(prepared_test.raw),
        engine.feature_mean,
        engine.feature_std,
    )
    single = evaluate_model(
        engine.model,
        list(prepared_test.single_raw),
        engine.feature_mean,
        engine.feature_std,
    )
    return {
        "test_sequences": len(prepared_test.specs),
        "sequence": sequence.to_dict(),
        "single_digit": single.to_dict(),
    }
