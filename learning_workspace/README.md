# ASR 亲手编码区

这里只放你亲自完成的练习，不要从 `acoustic_engine/` 复制整段实现。

按顺序编辑 `asr_practice.py`，每次只完成一个函数，然后运行：

```powershell
uv run python -m acoustic_engine.challenge --list
uv run python -m acoustic_engine.challenge --hint sampling
uv run python -m acoustic_engine.challenge --check sampling
uv run python -m acoustic_engine.challenge --status
```

六关顺序：

1. `sampling`：采样率和时长换算为采样点；
2. `framing`：`center=False` 分帧数量；
3. `ctc`：合并相邻重复并删除 blank；
4. `streaming-ctc`：跨 chunk 保存上一帧类别；
5. `bigram`：Add-k Bigram 条件概率；
6. `rtf`：处理耗时与音频时长的比值。

验证通过只代表函数行为正确。随后还要闭卷解释、故障注入并把函数迁移到新输入。
