from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .decoder import greedy_decode
from .features import LogMelConfig, LogMelFrontend
from .model import CausalConvCTCAcousticModel, CausalConvCTCConfig
from .streaming import StreamingAcousticEngine
from .train_digits import DIGIT_TOKENS, TRAIN_TEXTS, encode_targets, join_digits, load_digit_parts


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    parts, sample_rate = load_digit_parts(args.data_dir)
    frontend_config = LogMelConfig(
        sample_rate=sample_rate,
        n_fft=256,
        win_length=200,
        hop_length=160,
        n_mels=24,
        peak_normalize=False,
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

    model = CausalConvCTCAcousticModel(
        CausalConvCTCConfig(
            feature_dim=24,
            hidden_dim=48,
            num_layers=4,
            kernel_size=5,
            num_classes=11,
        )
    )
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} ctc_loss={loss.item():.4f}")

    model.eval()
    with torch.inference_mode():
        logits, output_lengths = model(padded, lengths)
    predictions = greedy_decode(logits, output_lengths, DIGIT_TOKENS)
    batch_exact = sum(a == b for a, b in zip(predictions, TRAIN_TEXTS, strict=True))
    print(f"full-utterance exact matches: {batch_exact}/{len(TRAIN_TEXTS)}")

    engine = StreamingAcousticEngine(
        frontend_config,
        model,
        DIGIT_TOKENS,
        feature_mean,
        feature_std,
    )
    stream_predictions = [
        engine.recognize_waveform(waveform, chunk_samples=args.chunk_samples).text
        for waveform in waveforms
    ]
    stream_exact = sum(a == b for a, b in zip(stream_predictions, TRAIN_TEXTS, strict=True))
    print(f"chunked-stream exact matches: {stream_exact}/{len(TRAIN_TEXTS)}")
    for target, prediction in zip(TRAIN_TEXTS, stream_predictions, strict=True):
        print(f"target={target:>2} streaming={prediction or '<empty>':>8}")
    saved = engine.save(args.output)
    print(f"saved checkpoint: {saved}")
    print("boundary: this is a causal overfitting lab; speaker-disjoint quality is unproven")
    return saved


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train the stateful streaming digit CTC engine")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "spoken_digits_parts")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "streaming_digit_ctc.pt")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--chunk-samples", type=int, default=800)
    parser.add_argument("--seed", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
