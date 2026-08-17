from __future__ import annotations

import argparse
from pathlib import Path

from .streaming import StreamingAcousticEngine
from .language_model import AddKBigramLanguageModel, ShallowFusionScorer
from .recognize import encode_hotwords


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a WAV through a trained causal engine")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--chunk-samples", type=int, default=800)
    parser.add_argument("--decoder", choices=("greedy", "prefix_beam"), default="greedy")
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--lm-corpus", type=Path)
    parser.add_argument("--lm-weight", type=float, default=0.0)
    parser.add_argument("--token-bonus", type=float, default=0.0)
    parser.add_argument("--hotword", action="append", default=[])
    parser.add_argument("--hotword-bonus", type=float, default=0.0)
    args = parser.parse_args()
    engine = StreamingAcousticEngine.load(args.checkpoint)
    language_model = None
    if args.lm_corpus is not None:
        corpus = [line.strip() for line in args.lm_corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
        language_model = AddKBigramLanguageModel.fit(corpus, engine.tokens)
    scorer = ShallowFusionScorer(
        language_model=language_model,
        lm_weight=args.lm_weight,
        token_bonus=args.token_bonus,
        hotwords=encode_hotwords(args.hotword, engine.tokens),
        hotword_bonus=args.hotword_bonus,
    )
    use_scorer = language_model is not None or bool(args.hotword) or args.token_bonus != 0.0
    update = engine.recognize_file(
        args.audio,
        chunk_samples=args.chunk_samples,
        decoder=args.decoder,
        beam_size=args.beam_size,
        extension_scorer=scorer if use_scorer else None,
    )
    print(f"text={update.text or '<empty>'}")
    print(f"decoder={args.decoder}")
    print(f"processed_frames={update.total_frames}")
    print(f"chunk_samples={args.chunk_samples}")


if __name__ == "__main__":
    main()
