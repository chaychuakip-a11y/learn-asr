from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import torch

from .streaming import StreamingAcousticEngine


class AudioRequest(BaseModel):
    samples: list[float] = Field(min_length=1)
    sample_rate: int
    chunk_samples: int = Field(default=800, gt=0)


def create_app(engine_or_checkpoint: StreamingAcousticEngine | str | Path) -> FastAPI:
    engine = (
        engine_or_checkpoint
        if isinstance(engine_or_checkpoint, StreamingAcousticEngine)
        else StreamingAcousticEngine.load(engine_or_checkpoint)
    )
    api = FastAPI(title="Learn ASR streaming acoustic engine", version="0.1.0")

    @api.get("/health")
    def health() -> dict[str, object]:
        return {
            "ready": True,
            "sample_rate": engine.frontend_config.sample_rate,
            "tokens": list(engine.tokens),
        }

    @api.post("/recognize")
    def recognize(request: AudioRequest) -> dict[str, object]:
        if request.sample_rate != engine.frontend_config.sample_rate:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"sample_rate must be {engine.frontend_config.sample_rate}; "
                    "resample at the client or use the file CLI"
                ),
            )
        update = engine.recognize_waveform(
            torch.tensor(request.samples, dtype=torch.float32),
            chunk_samples=request.chunk_samples,
        )
        return asdict(update)

    @api.websocket("/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        session = engine.start_session()
        try:
            while True:
                message = await websocket.receive_json()
                sample_rate = int(message.get("sample_rate", engine.frontend_config.sample_rate))
                if sample_rate != engine.frontend_config.sample_rate:
                    await websocket.send_json(
                        {"error": f"sample_rate must be {engine.frontend_config.sample_rate}"}
                    )
                    await websocket.close(code=1003)
                    return
                samples = message.get("samples", [])
                final = bool(message.get("final", False))
                update = session.accept_audio(
                    torch.tensor(samples, dtype=torch.float32),
                    final=final,
                )
                await websocket.send_json(asdict(update))
                if final:
                    return
        except WebSocketDisconnect:
            return

    return api


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "artifacts" / "streaming_digit_ctc.pt"
if DEFAULT_CHECKPOINT.exists():
    app = create_app(DEFAULT_CHECKPOINT)
else:
    app = FastAPI(title="Learn ASR streaming acoustic engine", version="0.1.0")

    @app.get("/health")
    def checkpoint_missing() -> dict[str, object]:
        return {"ready": False, "missing_checkpoint": str(DEFAULT_CHECKPOINT)}
