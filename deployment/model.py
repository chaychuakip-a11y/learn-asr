from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn


FEATURE_DIM = 24
NUM_CLASSES = 11
CHUNK_FRAMES = 8
CACHE_FRAMES = 2


class StreamingCTCDemo(nn.Module):
    """Small causal acoustic model with an explicit input/output cache."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(FEATURE_DIM, 32, kernel_size=3)
        self.head = nn.Linear(32, NUM_CLASSES)

    def forward(self, frames: torch.Tensor, cache: torch.Tensor):
        joined = torch.cat([cache, frames], dim=1)
        hidden = torch.relu(self.conv(joined.transpose(1, 2)).transpose(1, 2))
        logits = self.head(hidden)
        new_cache = joined[:, -CACHE_FRAMES:, :]
        return logits, new_cache


def build_model(seed: int = 17) -> StreamingCTCDemo:
    torch.manual_seed(seed)
    return StreamingCTCDemo().eval()


def export_onnx(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()
    frames = torch.randn(1, CHUNK_FRAMES, FEATURE_DIM)
    cache = torch.zeros(1, CACHE_FRAMES, FEATURE_DIM)
    torch.onnx.export(
        model,
        (frames, cache),
        path,
        input_names=["frames", "cache"],
        output_names=["logits", "new_cache"],
        opset_version=18,
        dynamo=True,
        verbose=False,
    )
    return path


def compare_torch_onnx(path: Path) -> dict[str, float]:
    rng = np.random.default_rng(21)
    frames = rng.normal(size=(1, CHUNK_FRAMES, FEATURE_DIM)).astype(np.float32)
    cache = rng.normal(size=(1, CACHE_FRAMES, FEATURE_DIM)).astype(np.float32)
    model = build_model()
    with torch.inference_mode():
        expected = model(torch.from_numpy(frames), torch.from_numpy(cache))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"frames": frames, "cache": cache})
    return {
        "logits_max_abs_error": float(np.max(np.abs(expected[0].numpy() - actual[0]))),
        "cache_max_abs_error": float(np.max(np.abs(expected[1].numpy() - actual[1]))),
    }
