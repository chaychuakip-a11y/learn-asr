# 实时数字 CTC 声学引擎结课项目

主 Notebook：`notebooks/结课项目_实时数字CTC声学引擎_从WAV到流式文本.ipynb`。

建议学习顺序：

1. 第一次：checkpoint 契约、WAV/Log-Mel/logits shape、CTC collapse；
2. 第二次：training-style 与 speaker-disjoint 评测、CER 和混淆矩阵；
3. 第三次：三类流式状态、50 组随机 chunk 数值等价、partial/PGS；
4. 第四次：RTF、LM/热词、故障注入、测试和 64 分答辩。

运行：

```powershell
uv sync
uv run jupyter lab
```

自动回归：

```powershell
uv run python -m unittest discover -s tests -p "test_acoustic_engine.py"
```

基础概念交互教练：

```powershell
uv run python -m acoustic_engine.tutor
uv run python -m acoustic_engine.tutor --status
```

重新训练时先使用新输出文件，不要直接覆盖已验收 checkpoint：

```powershell
uv run python -m acoustic_engine.train_streaming_digits --epochs 300 --output artifacts/my_streaming_digit_ctc.pt
```

通过标准：所有 Notebook 断言及声学引擎测试通过；能解释训练风格 10/10 与跨说话人低准确率的证据边界；能独立画出三类流式状态；答辩至少 52/64。

当前两个 `.pt` 教学 checkpoint 由仓库内 FSDD 派生录音训练。公开再分发时保留 `DATA_SOURCES.md` 中的上游署名和 CC BY-SA 4.0 说明。完整 FSDD 实验应固定数据 revision、按 speaker 划分，并单独保存数据清单和评测报告。
