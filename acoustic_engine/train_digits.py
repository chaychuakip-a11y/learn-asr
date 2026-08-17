from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence

from .decoder import greedy_decode
from .engine import AcousticEngine
from .features import LogMelConfig, LogMelFrontend
from .model import TinyCTCAcousticModel, TinyCTCConfig


DIGIT_TOKENS = tuple("0123456789")
TRAIN_TEXTS = tuple(DIGIT_TOKENS) + (
    "12", "21", "34", "43", "56", "65", "78", "87", "90", "09", "11", "22"
)


def load_digit_parts(data_dir: Path) -> tuple[dict[str, torch.Tensor], int]:
    waveforms: dict[str, torch.Tensor] = {}
    expected_sample_rate: int | None = None
    for digit in DIGIT_TOKENS:
        path = data_dir / f"{digit}_jackson_0.wav"
        audio, sample_rate = sf.read(path, dtype="float32")
        if audio.ndim != 1:
            raise ValueError(f"expected mono audio: {path}")
        if expected_sample_rate is None:
            expected_sample_rate = sample_rate
        elif sample_rate != expected_sample_rate:
            raise ValueError("all digit parts must use the same sample rate")
        waveforms[digit] = torch.from_numpy(audio)
    assert expected_sample_rate is not None
    return waveforms, expected_sample_rate


def join_digits(
    text: str,
    parts: dict[str, torch.Tensor],
    sample_rate: int,
) -> torch.Tensor:
    silence = torch.zeros(round(sample_rate * 0.06))
    pieces: list[torch.Tensor] = []
    for index, digit in enumerate(text):
        pieces.append(parts[digit])
        if index + 1 < len(text):
            pieces.append(silence)
    return torch.cat(pieces)


def encode_targets(texts: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    flat = [DIGIT_TOKENS.index(character) + 1 for text in texts for character in text]
    return torch.tensor(flat, dtype=torch.long), torch.tensor([len(text) for text in texts])


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    parts, sample_rate = load_digit_parts(args.data_dir)
    frontend_config = LogMelConfig(
        sample_rate=sample_rate,
        n_fft=256,
        win_length=200,
        hop_length=80,
        n_mels=24,
    )
    frontend = LogMelFrontend(frontend_config)
    waveforms = [join_digits(text, parts, sample_rate) for text in TRAIN_TEXTS]
    raw_features = [frontend(waveform) for waveform in waveforms]
    stacked = torch.cat(raw_features, dim=0)
    feature_mean = stacked.mean(dim=0)
    feature_std = stacked.std(dim=0).clamp_min(1e-5)
    features = [(item - feature_mean) / feature_std for item in raw_features]
    lengths = torch.tensor([item.shape[0] for item in features], dtype=torch.long)
    padded = pad_sequence(features, batch_first=True)
    targets, target_lengths = encode_targets(TRAIN_TEXTS)

    model_config = TinyCTCConfig(feature_dim=24, hidden_dim=48, num_classes=11)
    model = TinyCTCAcousticModel(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        logits, output_lengths = model(padded, lengths)
        loss = criterion(
            logits.log_softmax(dim=-1).transpose(0, 1),
            targets,
            output_lengths,
            target_lengths,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} ctc_loss={loss.item():.4f}")

    model.eval()
    with torch.inference_mode():
        logits, output_lengths = model(padded, lengths)
    predictions = greedy_decode(logits, output_lengths, DIGIT_TOKENS)
    exact = sum(prediction == target for prediction, target in zip(predictions, TRAIN_TEXTS, strict=True))
    print(f"training-set exact matches: {exact}/{len(TRAIN_TEXTS)}")
    for target, prediction in zip(TRAIN_TEXTS, predictions, strict=True):
        print(f"target={target:>2} prediction={prediction or '<empty>':>8}")

    engine = AcousticEngine(frontend, model, DIGIT_TOKENS, feature_mean, feature_std)
    saved = engine.save(args.output)
    print(f"saved checkpoint: {saved}")
    print("boundary: this tiny checkpoint is an overfitting lab, not a general ASR model")
    return saved


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train the capstone digit CTC engine")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data" / "spoken_digits_parts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts" / "tiny_digit_ctc.pt",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=8e-3)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
