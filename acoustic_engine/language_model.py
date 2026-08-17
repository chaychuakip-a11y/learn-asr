from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Protocol


class ExtensionScorer(Protocol):
    """Extra log score added only when CTC extends a token prefix."""

    def score_extension(self, prefix: tuple[int, ...], token_id: int) -> float: ...


class AddKBigramLanguageModel:
    """Small teaching LM matching lessons 19-20; not a production LM."""

    BOS = -1

    def __init__(
        self,
        tokens: Sequence[str],
        context_counts: dict[int, int],
        bigram_counts: dict[tuple[int, int], int],
        add_k: float = 0.1,
    ):
        if not tokens:
            raise ValueError("language-model vocabulary cannot be empty")
        if add_k <= 0:
            raise ValueError("add_k must be positive")
        self.tokens = tuple(tokens)
        self.context_counts = Counter(context_counts)
        self.bigram_counts = Counter(bigram_counts)
        self.add_k = float(add_k)

    @classmethod
    def fit(
        cls,
        corpus: Iterable[str | Sequence[str]],
        tokens: Sequence[str],
        add_k: float = 0.1,
    ) -> "AddKBigramLanguageModel":
        vocabulary = tuple(tokens)
        token_to_id = {token: index for index, token in enumerate(vocabulary)}
        if len(token_to_id) != len(vocabulary):
            raise ValueError("tokens must be unique")
        contexts: Counter[int] = Counter()
        bigrams: Counter[tuple[int, int]] = Counter()
        seen = 0
        for sample in corpus:
            pieces = list(sample) if isinstance(sample, str) else list(sample)
            try:
                ids = [token_to_id[piece] for piece in pieces]
            except KeyError as exc:
                raise ValueError(f"LM corpus token is outside the vocabulary: {exc.args[0]}") from exc
            previous = cls.BOS
            for token_id in ids:
                contexts[previous] += 1
                bigrams[previous, token_id] += 1
                previous = token_id
                seen += 1
        if seen == 0:
            raise ValueError("language-model corpus contains no tokens")
        return cls(vocabulary, dict(contexts), dict(bigrams), add_k)

    def score_extension(self, prefix: tuple[int, ...], token_id: int) -> float:
        if token_id < 0 or token_id >= len(self.tokens):
            raise ValueError(f"token id is outside LM vocabulary: {token_id}")
        previous = prefix[-1] if prefix else self.BOS
        numerator = self.bigram_counts[previous, token_id] + self.add_k
        denominator = self.context_counts[previous] + self.add_k * len(self.tokens)
        return math.log(numerator / denominator)

    def sequence_log_probability(self, token_ids: Sequence[int]) -> float:
        prefix: tuple[int, ...] = ()
        total = 0.0
        for token_id in token_ids:
            total += self.score_extension(prefix, int(token_id))
            prefix += (int(token_id),)
        return total

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "tokens": list(self.tokens),
            "add_k": self.add_k,
            "context_counts": {str(key): value for key, value in self.context_counts.items()},
            "bigram_counts": {
                f"{previous},{current}": value
                for (previous, current), value in self.bigram_counts.items()
            },
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "AddKBigramLanguageModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("unsupported language-model format")
        contexts = {int(key): int(value) for key, value in payload["context_counts"].items()}
        bigrams = {}
        for key, value in payload["bigram_counts"].items():
            previous, current = key.split(",", maxsplit=1)
            bigrams[int(previous), int(current)] = int(value)
        return cls(payload["tokens"], contexts, bigrams, float(payload["add_k"]))


@dataclass(frozen=True)
class ShallowFusionScorer:
    language_model: AddKBigramLanguageModel | None = None
    lm_weight: float = 0.0
    token_bonus: float = 0.0
    hotwords: tuple[tuple[int, ...], ...] = ()
    hotword_bonus: float = 0.0

    def score_extension(self, prefix: tuple[int, ...], token_id: int) -> float:
        score = self.token_bonus
        if self.language_model is not None:
            score += self.lm_weight * self.language_model.score_extension(prefix, token_id)
        extended = prefix + (token_id,)
        if self.hotword_bonus and any(
            len(extended) >= len(hotword) and extended[-len(hotword) :] == hotword
            for hotword in self.hotwords
            if hotword
        ):
            score += self.hotword_bonus
        return score
