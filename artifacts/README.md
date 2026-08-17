# Teaching model artifacts

- `tiny_digit_ctc.pt`: offline bidirectional-GRU CTC teaching checkpoint;
- `streaming_digit_ctc.pt`: causal streaming CTC teaching checkpoint;
- `streaming_ctc_demo.onnx` and `streaming_ctc_demo.int8.onnx`: deterministic deployment lab artifacts.

The PyTorch digit checkpoints are intentionally tiny overfitting demonstrations trained from repository copies or adaptations of Free Spoken Digit Dataset recordings. They prove serialization and inference contracts, not speaker-independent ASR quality.

FSDD recordings and course adaptations are under the upstream CC BY-SA 4.0 license. Keep `DATA_SOURCES.md` and its attribution when redistributing FSDD-derived teaching artifacts. Original project code is Apache-2.0 and original course prose is CC BY 4.0 as described in `LICENSE-SCOPE.md`.
