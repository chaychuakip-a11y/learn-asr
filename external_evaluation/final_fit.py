from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from acoustic_engine.features import LogMelFrontend
from acoustic_engine.model import CausalConvCTCAcousticModel, CausalConvCTCConfig
from acoustic_engine.streaming import StreamingAcousticEngine
from fsdd_generalization.data import FSDD_REVISION, prepare_fsdd
from fsdd_generalization.training import (
    DIGIT_TOKENS,
    ExperimentConfig,
    _batches,
    _normalized,
    build_feature_examples,
    build_sequence_specs,
    feature_statistics,
    frontend_config,
)
from .protocol import load_protocol, protocol_sha256


PREREGISTRATION_COMMIT = "0b303605ea02aa486ba4de36469d0ac42cef9966"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_final_config() -> ExperimentConfig:
    protocol = load_protocol()
    final_fit = protocol["final_fit"]
    return ExperimentConfig(
        seed=int(final_fit["seed"]),
        epochs=int(final_fit["epochs"]),
        batch_size=int(final_fit["optimizer"]["batch_size"]),
        learning_rate=float(final_fit["optimizer"]["learning_rate"]),
        hidden_dim=int(final_fit["model"]["hidden_dim"]),
        num_layers=int(final_fit["model"]["num_layers"]),
        kernel_size=int(final_fit["model"]["kernel_size"]),
        augmentation_copies=int(final_fit["augmentation"]["copies"]),
        patience=1,
        train_sequences=int(final_fit["train_sequences"]),
        dev_sequences=0,
        test_sequences=0,
        min_digits=int(final_fit["min_digits"]),
        max_digits=int(final_fit["max_digits"]),
    )


def run_final_fit(
    cache_dir: Path,
    checkpoint_path: Path,
    result_path: Path,
) -> dict[str, object]:
    """Fit the preregistered model using FSDD only and a fixed epoch count."""
    protocol = load_protocol()
    config = frozen_final_config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(1)

    _, split_rows = prepare_fsdd(cache_dir, validate_audio=False)
    recordings = [
        row for split in ("train", "dev", "test") for row in split_rows[split]
    ]
    if len(recordings) != 3_000 or len({row.speaker for row in recordings}) != 6:
        raise ValueError("final fit requires all 3,000 recordings from all six FSDD speakers")

    frontend = LogMelFrontend(frontend_config())
    specs = build_sequence_specs(
        recordings,
        config.train_sequences,
        config.min_digits,
        config.max_digits,
        config.seed + 101,
        "final-train",
    )
    print("building FSDD-only clean and augmented features ...", flush=True)
    raw_examples = build_feature_examples(
        specs, frontend, config.augmentation_copies, config.seed
    )
    mean, std = feature_statistics(raw_examples)
    examples = _normalized(raw_examples, mean, std)

    model_config = CausalConvCTCConfig(
        feature_dim=frontend.config.n_mels,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        kernel_size=config.kernel_size,
        num_classes=11,
    )
    model = CausalConvCTCAcousticModel(model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=float(protocol["final_fit"]["optimizer"]["weight_decay"]),
    )
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for padded, lengths, targets, target_lengths in _batches(
            examples, config.batch_size, config.seed + epoch
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
                raise FloatingPointError(f"non-finite final-fit CTC loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(protocol["final_fit"]["optimizer"]["gradient_clip_norm"]),
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        history.append(row)
        print(
            f"final_fit epoch={epoch:02d}/{config.epochs} train_loss={row['train_loss']:.4f}",
            flush=True,
        )

    engine = StreamingAcousticEngine(
        frontend.config, model, DIGIT_TOKENS, mean, std
    )
    engine.save(checkpoint_path)
    result: dict[str, object] = {
        "format_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256(),
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "audio_mnist_accessed_by_final_fit": False,
        "training_dataset": "Free Spoken Digit Dataset",
        "training_revision": FSDD_REVISION,
        "training_recordings": len(recordings),
        "training_speakers": sorted({row.speaker for row in recordings}),
        "training_sequences": len(specs),
        "training_feature_examples": len(raw_examples),
        "config": asdict(config),
        "model_config": model_config.to_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "checkpoint_selection": "fixed final epoch 10; no dev or external metric read",
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("checkpoint:", checkpoint_path, flush=True)
    print("fit evidence:", result_path, flush=True)
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_final_fit(
        root / ".local_data",
        root / "artifacts" / "fsdd_final_external_frozen.pt",
        root / "artifacts" / "fsdd_final_fit.json",
    )


if __name__ == "__main__":
    main()
