from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import shutil
import statistics
from typing import Iterable

from .data import EXPECTED_SPEAKERS, Recording, SplitSpec, prepare_fsdd, scan_recordings, split_recordings
from .training import (
    ExperimentConfig,
    evaluate_checkpoint_on_test,
    prepare_experiment_features,
    prepare_test_features,
    run_training,
)


@dataclass(frozen=True)
class OuterFold:
    index: int
    train_speakers: tuple[str, ...]
    dev_speaker: str
    test_speaker: str

    def split_spec(self) -> SplitSpec:
        spec = SplitSpec(
            train_speakers=self.train_speakers,
            dev_speakers=(self.dev_speaker,),
            test_speakers=(self.test_speaker,),
        )
        spec.validate()
        return spec


@dataclass(frozen=True)
class Candidate:
    name: str
    augmentation_copies: int
    complexity_rank: int


DEFAULT_CANDIDATES = (
    Candidate("clean", augmentation_copies=0, complexity_rank=0),
    Candidate("gain_noise", augmentation_copies=1, complexity_rank=1),
)


def build_loso_folds(speakers: Iterable[str] = EXPECTED_SPEAKERS) -> tuple[OuterFold, ...]:
    ordered = tuple(speakers)
    if len(ordered) < 3 or len(set(ordered)) != len(ordered):
        raise ValueError("LOSO needs at least three unique speakers")
    if set(ordered) != set(EXPECTED_SPEAKERS):
        raise ValueError(f"folds must cover exactly {EXPECTED_SPEAKERS}")
    folds = []
    for index, test_speaker in enumerate(ordered):
        dev_speaker = ordered[(index + 1) % len(ordered)]
        train_speakers = tuple(
            speaker for speaker in ordered if speaker not in {test_speaker, dev_speaker}
        )
        fold = OuterFold(index, train_speakers, dev_speaker, test_speaker)
        fold.split_spec()
        folds.append(fold)
    if {fold.test_speaker for fold in folds} != set(ordered):
        raise AssertionError("each speaker must be outer test exactly once")
    if {fold.dev_speaker for fold in folds} != set(ordered):
        raise AssertionError("each speaker must be dev exactly once")
    return tuple(folds)


def select_candidate(candidate_results: dict[str, dict[str, object]]) -> str:
    """Select by dev CER, then dev exact, then pre-registered simplicity."""
    ranks = {candidate.name: candidate.complexity_rank for candidate in DEFAULT_CANDIDATES}
    if set(candidate_results) != set(ranks):
        raise ValueError(f"expected candidates {sorted(ranks)}, got {sorted(candidate_results)}")

    def key(name: str) -> tuple[float, float, int, str]:
        dev = candidate_results[name]["dev"]
        if not isinstance(dev, dict):
            raise TypeError("candidate dev result must be a dictionary")
        return (
            float(dev["cer"]),
            -float(dev["exact_rate"]),
            ranks[name],
            name,
        )

    return min(candidate_results, key=key)


def _mean_interval(values: list[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("speaker-level interval needs at least two folds")
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    # Two-sided 95% t critical value for df=5; this experiment always has six speakers.
    t_critical = 2.5705818366147395 if len(values) == 6 else 1.96
    half = t_critical * std / math.sqrt(len(values))
    return {
        "speaker_count": len(values),
        "mean": mean,
        "sample_std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def aggregate_outer_folds(folds: list[dict[str, object]]) -> dict[str, object]:
    if len(folds) != len(EXPECTED_SPEAKERS) or {
        str(row["test_speaker"]) for row in folds
    } != set(EXPECTED_SPEAKERS):
        raise ValueError("aggregate requires one outer result for every FSDD speaker")
    sequence_rows = [row["outer_test"]["sequence"] for row in folds]  # type: ignore[index]
    single_rows = [row["outer_test"]["single_digit"] for row in folds]  # type: ignore[index]

    def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
        errors = sum(int(row["errors"]) for row in rows)
        characters = sum(int(row["reference_characters"]) for row in rows)
        exact = sum(int(row["exact"]) for row in rows)
        total = sum(int(row["total"]) for row in rows)
        return {
            "micro_cer": errors / characters,
            "micro_exact_rate": exact / total,
            "macro_speaker_cer": _mean_interval([float(row["cer"]) for row in rows]),
            "macro_speaker_exact_rate": _mean_interval(
                [float(row["exact_rate"]) for row in rows]
            ),
            "total_errors": errors,
            "total_reference_characters": characters,
            "total_exact": exact,
            "total_utterances": total,
        }

    selected_counts = {
        candidate.name: sum(row["selected_candidate"] == candidate.name for row in folds)
        for candidate in DEFAULT_CANDIDATES
    }
    return {
        "sequence": summarize(sequence_rows),
        "single_digit": summarize(single_rows),
        "selected_candidate_counts": selected_counts,
    }


def default_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=41,
        epochs=14,
        batch_size=64,
        learning_rate=2e-3,
        hidden_dim=48,
        num_layers=5,
        kernel_size=5,
        augmentation_copies=0,
        patience=4,
        train_sequences=1_200,
        dev_sequences=300,
        test_sequences=300,
        min_digits=1,
        max_digits=4,
    )


def run_outer_fold(
    recordings: list[Recording],
    fold: OuterFold,
    base_config: ExperimentConfig,
    work_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, object]:
    splits = split_recordings(recordings, fold.split_spec())
    fold_config = replace(base_config, seed=base_config.seed + fold.index * 1_009)
    clean_config = replace(fold_config, augmentation_copies=0)
    # Deliberately pass no test recordings: candidates cannot construct outer-test labels.
    prepared = prepare_experiment_features(
        splits["train"], splits["dev"], [], clean_config
    )
    candidate_results: dict[str, dict[str, object]] = {}
    candidate_checkpoints: dict[str, Path] = {}
    for candidate in DEFAULT_CANDIDATES:
        print(
            f"\nfold={fold.index + 1}/6 test={fold.test_speaker} "
            f"dev={fold.dev_speaker} candidate={candidate.name}",
            flush=True,
        )
        config = replace(
            fold_config, augmentation_copies=candidate.augmentation_copies
        )
        checkpoint = work_dir / f"fold_{fold.index + 1}_{candidate.name}.pt"
        result = run_training(
            splits["train"],
            splits["dev"],
            [],
            config,
            checkpoint,
            prepared=prepared,
            evaluate_test=False,
        )
        if result.get("test_evaluated") is not False or "test" in result:
            raise AssertionError("candidate training touched outer test")
        candidate_results[candidate.name] = result
        candidate_checkpoints[candidate.name] = checkpoint

    selected = select_candidate(candidate_results)
    selected_config = replace(
        fold_config,
        augmentation_copies=next(
            item.augmentation_copies for item in DEFAULT_CANDIDATES if item.name == selected
        ),
    )
    print(
        f"selected={selected} using dev only; constructing outer test={fold.test_speaker}",
        flush=True,
    )
    prepared_test = prepare_test_features(
        splits["test"], selected_config, prepared.frontend
    )
    outer_test = evaluate_checkpoint_on_test(
        candidate_checkpoints[selected], prepared_test
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    published_checkpoint = checkpoint_dir / f"fold_{fold.index + 1}_{fold.test_speaker}.pt"
    shutil.copy2(candidate_checkpoints[selected], published_checkpoint)
    return {
        "fold": fold.index + 1,
        "train_speakers": list(fold.train_speakers),
        "dev_speaker": fold.dev_speaker,
        "test_speaker": fold.test_speaker,
        "candidate_results": candidate_results,
        "selection_rule": "min(dev_cer, -dev_exact_rate, complexity_rank, name)",
        "selected_candidate": selected,
        "outer_test_constructed_after_selection": True,
        "outer_test": outer_test,
        "selected_checkpoint": published_checkpoint.name,
    }


def run_loso(
    recordings: list[Recording],
    output_path: Path,
    work_dir: Path,
    checkpoint_dir: Path,
    config: ExperimentConfig | None = None,
) -> dict[str, object]:
    config = config or default_experiment_config()
    work_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    result: dict[str, object] = {
        "format_version": 1,
        "protocol": "six_outer_speaker_folds_with_one_inner_dev_speaker",
        "interpretation": "exploratory_resampling_estimate_not_a_new_untouched_benchmark",
        "prior_data_contact": (
            "The preceding fixed-split lesson already used all FSDD speakers and inspected "
            "yweweler test outcomes; LOSO estimates speaker variability but is not a fresh blind test."
        ),
        "candidate_registry": [asdict(candidate) for candidate in DEFAULT_CANDIDATES],
        "base_config": asdict(config),
        "folds": folds,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fold in build_loso_folds():
        folds.append(run_outer_fold(recordings, fold, config, work_dir, checkpoint_dir))
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    result["aggregate"] = aggregate_outer_folds(folds)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    defaults = default_experiment_config()
    parser = argparse.ArgumentParser(description="Run leakage-gated six-fold FSDD LOSO")
    parser.add_argument("--cache-dir", type=Path, default=root / ".local_data")
    parser.add_argument(
        "--output", type=Path, default=root / "artifacts" / "fsdd_loso_results.json"
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=root / "artifacts" / "fsdd_loso_checkpoints"
    )
    parser.add_argument(
        "--work-dir", type=Path, default=root / ".local_data" / "fsdd_loso_work"
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--train-sequences", type=int, default=defaults.train_sequences)
    parser.add_argument("--dev-sequences", type=int, default=defaults.dev_sequences)
    parser.add_argument("--test-sequences", type=int, default=defaults.test_sequences)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.train_sequences, args.dev_sequences, args.test_sequences) <= 0:
        raise ValueError("epochs and sequence counts must be positive")
    recordings_dir, _ = prepare_fsdd(args.cache_dir)
    recordings = scan_recordings(recordings_dir, validate_audio=False)
    config = replace(
        default_experiment_config(),
        epochs=args.epochs,
        train_sequences=args.train_sequences,
        dev_sequences=args.dev_sequences,
        test_sequences=args.test_sequences,
    )
    result = run_loso(recordings, args.output, args.work_dir, args.checkpoint_dir, config)
    aggregate = result["aggregate"]
    print("\nLOSO complete", flush=True)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    print("results:", args.output, flush=True)


if __name__ == "__main__":
    main()
