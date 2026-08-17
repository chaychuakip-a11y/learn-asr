from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


@dataclass(frozen=True)
class LogMelConfig:
    """All units are explicit so the frontend contract is easy to inspect."""

    sample_rate: int = 16_000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 40
    f_min: float = 20.0
    f_max: float | None = None
    log_floor: float = 1e-6
    peak_normalize: bool = True

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _mel_filterbank(config: LogMelConfig) -> torch.Tensor:
    max_hz = config.sample_rate / 2 if config.f_max is None else config.f_max
    if not 0 <= config.f_min < max_hz <= config.sample_rate / 2:
        raise ValueError("expected 0 <= f_min < f_max <= Nyquist frequency")

    mel_edges = torch.linspace(
        _hz_to_mel(torch.tensor(config.f_min)),
        _hz_to_mel(torch.tensor(max_hz)),
        config.n_mels + 2,
    )
    hz_edges = _mel_to_hz(mel_edges)
    fft_hz = torch.linspace(0.0, config.sample_rate / 2, config.n_fft // 2 + 1)
    lower = hz_edges[:-2, None]
    center = hz_edges[1:-1, None]
    upper = hz_edges[2:, None]
    rising = (fft_hz[None, :] - lower) / (center - lower).clamp_min(1e-12)
    falling = (upper - fft_hz[None, :]) / (upper - center).clamp_min(1e-12)
    return torch.minimum(rising, falling).clamp(0.0, 1.0)


class LogMelFrontend:
    """Convert a mono waveform [samples] to Log-Mel features [time, mel]."""

    def __init__(self, config: LogMelConfig):
        if config.n_fft < config.win_length:
            raise ValueError("n_fft must be at least win_length")
        if config.hop_length <= 0 or config.win_length <= 0:
            raise ValueError("window and hop lengths must be positive")
        self.config = config
        self.window = torch.hann_window(config.win_length)
        self.mel_filters = _mel_filterbank(config)

    def __call__(self, waveform: torch.Tensor | np.ndarray) -> torch.Tensor:
        audio = torch.as_tensor(waveform, dtype=torch.float32).flatten()
        if audio.numel() == 0:
            raise ValueError("waveform is empty")
        if self.config.peak_normalize:
            audio = audio / audio.abs().max().clamp_min(1e-8)
        if audio.numel() < self.config.n_fft:
            audio = torch.nn.functional.pad(audio, (0, self.config.n_fft - audio.numel()))

        spectrum = torch.stft(
            audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = spectrum.abs().square()
        mel_power = self.mel_filters @ power
        return torch.log(mel_power.clamp_min(self.config.log_floor)).transpose(0, 1)


class StreamingLogMelFrontend:
    """Stateful center=False frontend that preserves overlap across audio chunks."""

    def __init__(self, config: LogMelConfig):
        if config.peak_normalize:
            raise ValueError("streaming frontend cannot use utterance-level peak normalization")
        self.frontend = LogMelFrontend(config)
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._buffer = torch.empty(0, dtype=torch.float32)
        self._emitted_frames = 0
        self._closed = False

    def accept(self, chunk: torch.Tensor, final: bool = False) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("streaming frontend is already finalized")
        audio = torch.as_tensor(chunk, dtype=torch.float32).flatten()
        self._buffer = torch.cat([self._buffer, audio])
        available = 0
        if self._buffer.numel() >= self.config.n_fft:
            available = 1 + (self._buffer.numel() - self.config.n_fft) // self.config.hop_length

        if available:
            process_length = self.config.n_fft + (available - 1) * self.config.hop_length
            features = self.frontend(self._buffer[:process_length])
            self._buffer = self._buffer[available * self.config.hop_length :]
            self._emitted_frames += available
        elif final and self._emitted_frames == 0 and self._buffer.numel() > 0:
            features = self.frontend(self._buffer)
            self._emitted_frames = features.shape[0]
        else:
            features = torch.empty((0, self.config.n_mels), dtype=torch.float32)

        if final:
            self._buffer = torch.empty(0, dtype=torch.float32)
            self._closed = True
        return features


def load_mono_audio(path: str | Path, target_sample_rate: int) -> torch.Tensor:
    """Load a WAV-like file, mix channels, and resample when necessary."""

    audio, sample_rate = sf.read(Path(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    if sample_rate != target_sample_rate:
        mono = librosa.resample(
            mono,
            orig_sr=sample_rate,
            target_sr=target_sample_rate,
        )
    return torch.from_numpy(np.asarray(mono, dtype=np.float32))
