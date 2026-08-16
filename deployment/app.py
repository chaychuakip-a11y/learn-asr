import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from deployment.model import CACHE_FRAMES, CHUNK_FRAMES, FEATURE_DIM


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INT8 = ROOT / "artifacts" / "streaming_ctc_demo.int8.onnx"
DEFAULT_FP32 = ROOT / "artifacts" / "streaming_ctc_demo.onnx"
MODEL_PATH = Path(os.environ.get("ASR_MODEL_PATH", DEFAULT_INT8 if DEFAULT_INT8.exists() else DEFAULT_FP32))
LABELS = ["∅", *list("0123456789")]

app = FastAPI(title="Streaming CTC teaching service", version="0.1.0")
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


class InferRequest(BaseModel):
    frames: list[list[float]] = Field(description=f"Exactly {CHUNK_FRAMES} frames × {FEATURE_DIM} features")
    cache: list[list[float]] | None = None


def run_chunk(frames, cache=None):
    x = np.asarray(frames, dtype=np.float32)
    if x.shape != (CHUNK_FRAMES, FEATURE_DIM):
        raise ValueError(f"frames shape must be {(CHUNK_FRAMES, FEATURE_DIM)}, got {x.shape}")
    if cache is None:
        c = np.zeros((CACHE_FRAMES, FEATURE_DIM), dtype=np.float32)
    else:
        c = np.asarray(cache, dtype=np.float32)
    if c.shape != (CACHE_FRAMES, FEATURE_DIM):
        raise ValueError(f"cache shape must be {(CACHE_FRAMES, FEATURE_DIM)}, got {c.shape}")
    logits, new_cache = session.run(None, {"frames": x[None], "cache": c[None]})
    return logits[0], new_cache[0]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "provider": session.get_providers()[0]}


@app.post("/infer")
def infer(request: InferRequest):
    logits, new_cache = run_chunk(request.frames, request.cache)
    return {"logits": logits.tolist(), "cache": new_cache.tolist()}


@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    cache = np.zeros((CACHE_FRAMES, FEATURE_DIM), dtype=np.float32)
    pending: list[list[float]] = []
    previous = 0
    transcript = ""
    sn = 0
    try:
        while True:
            message = await websocket.receive_json()
            if message.get("eof"):
                await websocket.send_json({"sn": sn + 1, "pgs": "apd", "text": "", "full": transcript, "ls": True})
                break
            pending.extend(message.get("features", []))
            while len(pending) >= CHUNK_FRAMES:
                chunk, pending = pending[:CHUNK_FRAMES], pending[CHUNK_FRAMES:]
                logits, cache = run_chunk(chunk, cache)
                added = []
                for token in logits.argmax(axis=-1):
                    token = int(token)
                    if token != 0 and token != previous:
                        added.append(LABELS[token])
                    previous = token
                text = "".join(added)
                transcript += text
                sn += 1
                await websocket.send_json({"sn": sn, "pgs": "apd", "text": text, "full": transcript, "ls": False})
    except WebSocketDisconnect:
        return
