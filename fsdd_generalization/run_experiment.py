from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .data import SplitSpec, build_manifest, prepare_fsdd
from .training import ExperimentConfig, prepare_experiment_features, run_training


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run speaker-disjoint FSDD CTC experiments")
    parser.add_argument("--cache-dir", type=Path, default=root / ".local_data")
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch-size, and patience must be positive")
    recordings_dir, splits = prepare_fsdd(args.cache_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(splits, SplitSpec())
    manifest_path = args.output_dir / "fsdd_speaker_disjoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("recordings:", recordings_dir, flush=True)
    print("manifest:", manifest_path, flush=True)

    baseline_config = ExperimentConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        augmentation_copies=0,
    )
    augmented_config = ExperimentConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        augmentation_copies=1,
    )
    print("preparing shared clean features ...", flush=True)
    prepared = prepare_experiment_features(
        splits["train"], splits["dev"], splits["test"], baseline_config
    )

    results: dict[str, object] = {
        "format_version": 1,
        "manifest": manifest_path.name,
        "split_spec": asdict(SplitSpec()),
    }
    if not args.skip_baseline:
        print("\n=== baseline: clean train only ===", flush=True)
        results["baseline"] = run_training(
            splits["train"], splits["dev"], splits["test"], baseline_config,
            prepared=prepared,
        )

    print(
        "\n=== augmented: one deterministic noisy copy per train recording ===",
        flush=True,
    )
    artifact_suffix = "_augmented_only" if args.skip_baseline else ""
    checkpoint = args.output_dir / f"fsdd_speaker_disjoint_ctc{artifact_suffix}.pt"
    results["augmented"] = run_training(
        splits["train"], splits["dev"], splits["test"], augmented_config, checkpoint,
        prepared=prepared,
    )
    results_path = args.output_dir / f"fsdd_speaker_disjoint_results{artifact_suffix}.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("checkpoint:", checkpoint, flush=True)
    print("results:", results_path, flush=True)


if __name__ == "__main__":
    main()
