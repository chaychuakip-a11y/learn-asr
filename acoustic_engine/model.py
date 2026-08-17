from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True)
class TinyCTCConfig:
    feature_dim: int = 40
    hidden_dim: int = 48
    num_layers: int = 1
    num_classes: int = 11

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class TinyCTCAcousticModel(nn.Module):
    """Offline teaching model: local convolution + bidirectional GRU + CTC head."""

    def __init__(self, config: TinyCTCConfig):
        super().__init__()
        self.config = config
        self.conv = nn.Conv1d(
            config.feature_dim,
            config.hidden_dim,
            kernel_size=5,
            padding=2,
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.encoder = nn.GRU(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(config.hidden_dim * 2, config.num_classes)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature]")
        if features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.config.feature_dim}, got {features.shape[-1]}"
            )
        if lengths.ndim != 1 or lengths.numel() != features.shape[0]:
            raise ValueError("lengths must have shape [batch]")
        hidden = torch.relu(self.conv(features.transpose(1, 2)).transpose(1, 2))
        hidden = self.norm(hidden)
        if torch.any(lengths <= 0) or torch.any(lengths > hidden.shape[1]):
            raise ValueError("every length must be in the range 1..time")
        packed = pack_padded_sequence(
            hidden,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_encoded, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            packed_encoded,
            batch_first=True,
            total_length=hidden.shape[1],
        )
        return self.head(encoded), lengths


@dataclass(frozen=True)
class StreamingCTCConfig:
    feature_dim: int = 40
    hidden_dim: int = 64
    num_layers: int = 2
    num_classes: int = 11

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class StreamingCTCAcousticModel(nn.Module):
    """Unidirectional GRU whose hidden state is the explicit chunk cache."""

    def __init__(self, config: StreamingCTCConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.feature_dim, config.hidden_dim)
        self.input_norm = nn.LayerNorm(config.hidden_dim)
        self.encoder = nn.GRU(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(config.hidden_dim, config.num_classes)

    def _validate(self, features: torch.Tensor) -> None:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature]")
        if features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.config.feature_dim}, got {features.shape[-1]}"
            )

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(features)
        if lengths.ndim != 1 or lengths.numel() != features.shape[0]:
            raise ValueError("lengths must have shape [batch]")
        if torch.any(lengths <= 0) or torch.any(lengths > features.shape[1]):
            raise ValueError("every length must be in the range 1..time")
        hidden = self.input_norm(torch.relu(self.input_projection(features)))
        packed = pack_padded_sequence(
            hidden,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_encoded, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            packed_encoded,
            batch_first=True,
            total_length=features.shape[1],
        )
        return self.head(encoded), lengths

    def initial_state(self, batch_size: int = 1) -> torch.Tensor:
        parameter = next(self.parameters())
        return parameter.new_zeros(self.config.num_layers, batch_size, self.config.hidden_dim)

    def forward_chunk(
        self,
        features: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(features)
        hidden = self.input_norm(torch.relu(self.input_projection(features)))
        encoded, new_state = self.encoder(hidden, state)
        return self.head(encoded), new_state


@dataclass(frozen=True)
class CausalConvCTCConfig:
    feature_dim: int = 40
    hidden_dim: int = 48
    num_layers: int = 8
    kernel_size: int = 5
    num_classes: int = 11

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class CausalConvCTCAcousticModel(nn.Module):
    """Finite-left-context encoder with one explicit cache per convolution layer."""

    def __init__(self, config: CausalConvCTCConfig):
        super().__init__()
        if config.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if config.num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.config = config
        self.input_projection = nn.Linear(config.feature_dim, config.hidden_dim)
        self.input_norm = nn.LayerNorm(config.hidden_dim)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(config.hidden_dim, config.hidden_dim, config.kernel_size)
            for _ in range(config.num_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)
        )
        self.head = nn.Linear(config.hidden_dim, config.num_classes)

    def _validate(self, features: torch.Tensor) -> None:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature]")
        if features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.config.feature_dim}, got {features.shape[-1]}"
            )

    def _input(self, features: torch.Tensor) -> torch.Tensor:
        return self.input_norm(torch.relu(self.input_projection(features)))

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(features)
        if lengths.ndim != 1 or lengths.numel() != features.shape[0]:
            raise ValueError("lengths must have shape [batch]")
        hidden = self._input(features)
        left = self.config.kernel_size - 1
        for convolution, norm in zip(self.convolutions, self.norms, strict=True):
            channel_first = F.pad(hidden.transpose(1, 2), (left, 0))
            update = torch.relu(convolution(channel_first).transpose(1, 2))
            hidden = norm(hidden + update)
        return self.head(hidden), lengths

    def initial_state(self, batch_size: int = 1) -> tuple[torch.Tensor, ...]:
        parameter = next(self.parameters())
        return tuple(
            parameter.new_zeros(batch_size, self.config.hidden_dim, self.config.kernel_size - 1)
            for _ in range(self.config.num_layers)
        )

    def forward_chunk(
        self,
        features: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        self._validate(features)
        caches = self.initial_state(features.shape[0]) if state is None else state
        if len(caches) != self.config.num_layers:
            raise ValueError("causal convolution cache count does not match num_layers")
        hidden = self._input(features)
        new_caches = []
        for convolution, norm, cache in zip(
            self.convolutions,
            self.norms,
            caches,
            strict=True,
        ):
            channel_first = hidden.transpose(1, 2)
            joined = torch.cat([cache, channel_first], dim=2)
            update = torch.relu(convolution(joined).transpose(1, 2))
            hidden = norm(hidden + update)
            new_caches.append(joined[:, :, -(self.config.kernel_size - 1) :])
        return self.head(hidden), tuple(new_caches)
