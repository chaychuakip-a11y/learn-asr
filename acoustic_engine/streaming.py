from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import torch

from .decoder import StreamingGreedyDecoder, StreamingPrefixBeamDecoder
from .features import LogMelConfig, StreamingLogMelFrontend, load_mono_audio
from .model import (
    CausalConvCTCAcousticModel,
    CausalConvCTCConfig,
    StreamingCTCAcousticModel,
    StreamingCTCConfig,
)
from .language_model import ExtensionScorer


@dataclass(frozen=True)
class StreamingUpdate:
    text: str
    new_text: str
    chunk_frames: int
    total_frames: int
    is_final: bool
    revised: bool = False


class StreamingSession:
    def __init__(
        self,
        engine: "StreamingAcousticEngine",
        decoder: str = "greedy",
        beam_size: int = 10,
        extension_scorer: ExtensionScorer | None = None,
    ):
        self.engine = engine
        self.frontend = StreamingLogMelFrontend(engine.frontend_config)
        if decoder == "greedy":
            self.decoder = StreamingGreedyDecoder(engine.tokens)
        elif decoder == "prefix_beam":
            self.decoder = StreamingPrefixBeamDecoder(
                engine.tokens,
                beam_size=beam_size,
                extension_scorer=extension_scorer,
            )
        else:
            raise ValueError(f"unknown decoder: {decoder}")
        self.model_state: object | None = None
        self.total_frames = 0
        self.closed = False

    @torch.inference_mode()
    def accept_audio(self, chunk: torch.Tensor, final: bool = False) -> StreamingUpdate:
        if self.closed:
            raise RuntimeError("streaming recognition session is already finalized")
        previous_text = self.decoder.text
        features = self.frontend.accept(chunk, final=final)
        if features.shape[0]:
            normalized = (features - self.engine.feature_mean) / self.engine.feature_std
            logits, self.model_state = self.engine.model.forward_chunk(
                normalized.unsqueeze(0),
                self.model_state,
            )
            if isinstance(self.decoder, StreamingGreedyDecoder):
                frame_ids = logits[0].argmax(dim=-1).tolist()
                self.decoder.accept(frame_ids)
            else:
                self.decoder.accept_logits(logits[0])
            self.total_frames += features.shape[0]
        if final:
            self.closed = True
        text = self.decoder.text
        revised = not text.startswith(previous_text)
        return StreamingUpdate(
            text=text,
            new_text=text[len(previous_text) :] if not revised else "",
            chunk_frames=features.shape[0],
            total_frames=self.total_frames,
            is_final=final,
            revised=revised,
        )


class StreamingAcousticEngine:
    """Raw-audio streaming frontend + recurrent cache + cross-chunk CTC state."""

    def __init__(
        self,
        frontend_config: LogMelConfig,
        model: StreamingCTCAcousticModel | CausalConvCTCAcousticModel,
        tokens: Sequence[str],
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ):
        if frontend_config.peak_normalize:
            raise ValueError("streaming engine requires peak_normalize=False")
        if model.config.num_classes != len(tokens) + 1:
            raise ValueError("CTC model needs one blank class plus one class per token")
        self.frontend_config = frontend_config
        self.model = model.eval()
        self.tokens = tuple(tokens)
        self.feature_mean = feature_mean.float()
        self.feature_std = feature_std.float().clamp_min(1e-5)
        expected = (model.config.feature_dim,)
        if self.feature_mean.shape != expected or self.feature_std.shape != expected:
            raise ValueError("feature statistics must have shape [feature_dim]")

    def start_session(
        self,
        decoder: str = "greedy",
        beam_size: int = 10,
        extension_scorer: ExtensionScorer | None = None,
    ) -> StreamingSession:
        return StreamingSession(self, decoder, beam_size, extension_scorer)

    def recognize_waveform(
        self,
        waveform: torch.Tensor,
        chunk_samples: int = 800,
        decoder: str = "greedy",
        beam_size: int = 10,
        extension_scorer: ExtensionScorer | None = None,
    ) -> StreamingUpdate:
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")
        session = self.start_session(decoder, beam_size, extension_scorer)
        final_update: StreamingUpdate | None = None
        for start in range(0, waveform.numel(), chunk_samples):
            end = min(start + chunk_samples, waveform.numel())
            final_update = session.accept_audio(waveform[start:end], final=end == waveform.numel())
        if final_update is None:
            raise ValueError("waveform is empty")
        return final_update

    def recognize_file(self, path: str | Path, chunk_samples: int = 800, **decode_options: object) -> StreamingUpdate:
        waveform = load_mono_audio(path, self.frontend_config.sample_rate)
        return self.recognize_waveform(waveform, chunk_samples, **decode_options)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "engine_type": (
                    "causal_conv_ctc"
                    if isinstance(self.model, CausalConvCTCAcousticModel)
                    else "streaming_gru_ctc"
                ),
                "frontend_config": self.frontend_config.to_dict(),
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
    def load(cls, path: str | Path) -> "StreamingAcousticEngine":
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported streaming acoustic engine checkpoint")
        frontend_config = LogMelConfig(**checkpoint["frontend_config"])
        if checkpoint.get("engine_type") == "streaming_gru_ctc":
            model = StreamingCTCAcousticModel(StreamingCTCConfig(**checkpoint["model_config"]))
        elif checkpoint.get("engine_type") == "causal_conv_ctc":
            model = CausalConvCTCAcousticModel(CausalConvCTCConfig(**checkpoint["model_config"]))
        else:
            raise ValueError("unsupported streaming acoustic engine checkpoint")
        model.load_state_dict(checkpoint["model_state"])
        return cls(
            frontend_config,
            model,
            checkpoint["tokens"],
            checkpoint["feature_mean"],
            checkpoint["feature_std"],
        )
