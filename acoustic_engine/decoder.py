from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch

from .language_model import ExtensionScorer


NEG_INF = float("-inf")


def _log_add(*values: float) -> float:
    finite = [value for value in values if value != NEG_INF]
    if not finite:
        return NEG_INF
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


@dataclass(frozen=True)
class BeamHypothesis:
    text: str
    token_ids: tuple[int, ...]
    score: float
    blank_score: float
    nonblank_score: float


class StreamingGreedyDecoder:
    """Carry the previous frame class so CTC collapse is correct across chunks."""

    def __init__(self, tokens: Sequence[str], blank_id: int = 0):
        self.tokens = tuple(tokens)
        self.blank_id = blank_id
        self.reset()

    def reset(self) -> None:
        self.previous_class: int | None = None
        self.token_ids: list[int] = []

    @property
    def text(self) -> str:
        return "".join(self.tokens[token_id] for token_id in self.token_ids)

    def accept(self, frame_ids: Sequence[int]) -> str:
        for class_id in frame_ids:
            class_id = int(class_id)
            if class_id != self.blank_id and class_id != self.previous_class:
                token_id = class_id if class_id < self.blank_id else class_id - 1
                if token_id < 0 or token_id >= len(self.tokens):
                    raise ValueError(f"class id {class_id} has no matching token")
                self.token_ids.append(token_id)
            self.previous_class = class_id
        return self.text


class StreamingPrefixBeamDecoder:
    """Carry CTC prefix probabilities across arbitrary logit chunks."""

    def __init__(
        self,
        tokens: Sequence[str],
        beam_size: int = 10,
        blank_id: int = 0,
        extension_scorer: ExtensionScorer | None = None,
    ):
        if not tokens:
            raise ValueError("tokens cannot be empty")
        if beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if blank_id < 0 or blank_id > len(tokens):
            raise ValueError("blank_id is outside the class dimension")
        self.tokens = tuple(tokens)
        self.beam_size = beam_size
        self.blank_id = blank_id
        self.extension_scorer = extension_scorer
        self.reset()

    def reset(self) -> None:
        self.beam: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}

    @property
    def text(self) -> str:
        return self.hypotheses()[0].text

    def accept_logits(self, logits: torch.Tensor) -> list[BeamHypothesis]:
        if logits.ndim != 2:
            raise ValueError("single-utterance logits must have shape [time, class]")
        if logits.shape[1] != len(self.tokens) + 1:
            raise ValueError("class dimension must equal len(tokens) + one CTC blank")
        for frame in logits.log_softmax(dim=-1).detach().cpu():
            next_beam: dict[tuple[int, ...], tuple[float, float]] = {}

            def update(
                prefix: tuple[int, ...],
                blank: float = NEG_INF,
                nonblank: float = NEG_INF,
            ) -> None:
                old_blank, old_nonblank = next_beam.get(prefix, (NEG_INF, NEG_INF))
                next_beam[prefix] = (
                    _log_add(old_blank, blank),
                    _log_add(old_nonblank, nonblank),
                )

            for prefix, (p_blank, p_nonblank) in self.beam.items():
                blank_logp = float(frame[self.blank_id])
                update(
                    prefix,
                    blank=_log_add(p_blank + blank_logp, p_nonblank + blank_logp),
                )
                for class_id in range(logits.shape[1]):
                    if class_id == self.blank_id:
                        continue
                    token_id = class_id if class_id < self.blank_id else class_id - 1
                    token_logp = float(frame[class_id])
                    extra = (
                        self.extension_scorer.score_extension(prefix, token_id)
                        if self.extension_scorer is not None
                        else 0.0
                    )
                    if prefix and token_id == prefix[-1]:
                        update(prefix, nonblank=p_nonblank + token_logp)
                        update(
                            prefix + (token_id,),
                            nonblank=p_blank + token_logp + extra,
                        )
                    else:
                        update(
                            prefix + (token_id,),
                            nonblank=_log_add(p_blank, p_nonblank) + token_logp + extra,
                        )
            ranked = sorted(
                next_beam.items(),
                key=lambda item: _log_add(*item[1]),
                reverse=True,
            )
            self.beam = dict(ranked[: self.beam_size])
        return self.hypotheses()

    def hypotheses(self) -> list[BeamHypothesis]:
        return [
            BeamHypothesis(
                text="".join(self.tokens[token_id] for token_id in token_ids),
                token_ids=token_ids,
                score=_log_add(p_blank, p_nonblank),
                blank_score=p_blank,
                nonblank_score=p_nonblank,
            )
            for token_ids, (p_blank, p_nonblank) in sorted(
                self.beam.items(),
                key=lambda item: _log_add(*item[1]),
                reverse=True,
            )
        ]


def ctc_collapse(
    frame_ids: Sequence[int],
    tokens: Sequence[str],
    blank_id: int = 0,
) -> tuple[str, list[int]]:
    """Merge adjacent repeats, remove blank, and map class ids to tokens."""

    token_ids: list[int] = []
    previous: int | None = None
    for class_id in frame_ids:
        class_id = int(class_id)
        if class_id != blank_id and class_id != previous:
            token_id = class_id - 1 if class_id > blank_id else class_id
            if token_id < 0 or token_id >= len(tokens):
                raise ValueError(f"class id {class_id} has no matching token")
            token_ids.append(token_id)
        previous = class_id
    return "".join(tokens[index] for index in token_ids), token_ids


def greedy_decode(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    tokens: Sequence[str],
    blank_id: int = 0,
) -> list[str]:
    """Decode logits [batch, time, class] with CTC greedy search."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, time, class]")
    if lengths.ndim != 1 or lengths.numel() != logits.shape[0]:
        raise ValueError("lengths must have shape [batch]")
    predictions = logits.argmax(dim=-1)
    results = []
    for row, length in zip(predictions, lengths, strict=True):
        text, _ = ctc_collapse(row[: int(length)].tolist(), tokens, blank_id)
        results.append(text)
    return results


def prefix_beam_search(
    logits: torch.Tensor,
    tokens: Sequence[str],
    beam_size: int = 10,
    blank_id: int = 0,
    extension_scorer: ExtensionScorer | None = None,
) -> list[BeamHypothesis]:
    """Exact CTC prefix recurrence with beam pruning and optional shallow fusion."""

    decoder = StreamingPrefixBeamDecoder(
        tokens,
        beam_size=beam_size,
        blank_id=blank_id,
        extension_scorer=extension_scorer,
    )
    return decoder.accept_logits(logits)
