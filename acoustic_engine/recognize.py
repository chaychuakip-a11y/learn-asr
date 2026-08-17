from __future__ import annotations

import argparse
from pathlib import Path

from .engine import AcousticEngine
from .language_model import AddKBigramLanguageModel, ShallowFusionScorer


def encode_hotwords(hotwords: list[str], tokens: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    token_to_id = {token: index for index, token in enumerate(tokens)}
    encoded = []
    for hotword in hotwords:
        try:
            encoded.append(tuple(token_to_id[character] for character in hotword))
        except KeyError as exc:
            raise ValueError(f"hotword token is outside the engine vocabulary: {exc.args[0]}") from exc
    return tuple(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize one WAV with a trained checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--decoder", choices=("greedy", "prefix_beam"), default="greedy")
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--lm-corpus", type=Path, help="UTF-8 text file with one token sequence per line")
    parser.add_argument("--lm-weight", type=float, default=0.0)
    parser.add_argument("--token-bonus", type=float, default=0.0)
    parser.add_argument("--hotword", action="append", default=[])
    parser.add_argument("--hotword-bonus", type=float, default=0.0)
    args = parser.parse_args()
    engine = AcousticEngine.load(args.checkpoint)
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
    use_scorer = language_model is not None or bool(args.hotword)
    result = engine.recognize_file(
        args.audio,
        decoder=args.decoder,
        beam_size=args.beam_size,
        extension_scorer=scorer if use_scorer else None,
    )
    print(f"text={result.text or '<empty>'}")
    print(f"decoder={result.decoder}")
    if result.score is not None:
        print(f"score={result.score:.4f}")
    print(f"feature_frames={result.feature_frames}")
    print(f"ctc_frame_ids={result.frame_ids}")


if __name__ == "__main__":
    main()
