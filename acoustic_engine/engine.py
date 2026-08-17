from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch

from .decoder import ctc_collapse, prefix_beam_search
from .features import LogMelConfig, LogMelFrontend, load_mono_audio
from .model import TinyCTCAcousticModel, TinyCTCConfig
from .language_model import ExtensionScorer


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    token_ids: list[int]
    frame_ids: list[int]
    feature_frames: int
    decoder: str = "greedy"
    score: float | None = None


class AcousticEngine:
    """Own the frontend, normalization, acoustic model, and CTC decoding contract."""

    def __init__(
        self,
        frontend: LogMelFrontend,
        model: TinyCTCAcousticModel,
        tokens: Sequence[str],
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ):
        if model.config.num_classes != len(tokens) + 1:
            raise ValueError("CTC model needs one blank class plus one class per token")
        self.frontend = frontend
        self.model = model.eval()
        self.tokens = tuple(tokens)
        feature_dim = model.config.feature_dim
        self.feature_mean = (
            torch.zeros(feature_dim) if feature_mean is None else feature_mean.float()
        )
        self.feature_std = (
            torch.ones(feature_dim) if feature_std is None else feature_std.float()
        ).clamp_min(1e-5)
        if self.feature_mean.shape != (feature_dim,) or self.feature_std.shape != (feature_dim,):
            raise ValueError("feature statistics must have shape [feature_dim]")

    @torch.inference_mode()
    def recognize_waveform(
        self,
        waveform: torch.Tensor,
        decoder: Literal["greedy", "prefix_beam"] = "greedy",
        beam_size: int = 10,
        extension_scorer: ExtensionScorer | None = None,
    ) -> RecognitionResult:
        features = self.frontend(waveform)
        normalized = (features - self.feature_mean) / self.feature_std
        lengths = torch.tensor([normalized.shape[0]], dtype=torch.long)
        logits, output_lengths = self.model(normalized.unsqueeze(0), lengths)
        frame_ids = logits[0, : int(output_lengths[0])].argmax(dim=-1).tolist()
        if decoder == "greedy":
            text, token_ids = ctc_collapse(frame_ids, self.tokens)
            return RecognitionResult(
                text=text,
                token_ids=token_ids,
                frame_ids=frame_ids,
                feature_frames=normalized.shape[0],
                decoder=decoder,
            )
        if decoder == "prefix_beam":
            best = prefix_beam_search(
                logits[0, : int(output_lengths[0])],
                self.tokens,
                beam_size=beam_size,
                extension_scorer=extension_scorer,
            )[0]
            return RecognitionResult(
                text=best.text,
                token_ids=list(best.token_ids),
                frame_ids=frame_ids,
                feature_frames=normalized.shape[0],
                decoder=decoder,
                score=best.score,
            )
        raise ValueError(f"unknown decoder: {decoder}")

    def recognize_file(self, path: str | Path, **decode_options: object) -> RecognitionResult:
        waveform = load_mono_audio(path, self.frontend.config.sample_rate)
        return self.recognize_waveform(waveform, **decode_options)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "frontend_config": self.frontend.config.to_dict(),
                "model_config": self.model.config.to_dict(),
                "model_state": self.model.state_dict(),
                "tokens": list(self.tokens),
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
            },
            output,
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "AcousticEngine":
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported acoustic engine checkpoint format")
        frontend = LogMelFrontend(LogMelConfig(**checkpoint["frontend_config"]))
        model = TinyCTCAcousticModel(TinyCTCConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        return cls(
            frontend,
            model,
            checkpoint["tokens"],
            checkpoint["feature_mean"],
            checkpoint["feature_std"],
        )
