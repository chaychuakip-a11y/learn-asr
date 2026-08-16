# Streaming CTC teaching deployment

The service accepts feature chunks rather than raw audio so that model serving,
cache ownership, WebSocket state, and PGS-style events remain easy to inspect.
The model has deterministic random weights and is for deployment mechanics, not
recognition accuracy.

Generate the ONNX artifacts by running lessons 25 and 27, then start the server:

```powershell
uv run uvicorn deployment.app:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `POST /infer`: one fixed feature chunk plus optional cache
- `WS /stream`: stateful feature chunks and PGS-style incremental responses
