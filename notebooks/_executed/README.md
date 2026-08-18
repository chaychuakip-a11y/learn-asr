# 已运行 Notebook 对照区

这里保存课程的执行结果快照，供卡住时核对图形、数值、shape 和预期输出。它们不是第二套课程，也不是推荐的学习入口。

正确使用方式：

1. 从 [`../README.md`](../README.md) 打开根目录中的学习版；
2. 先预测结果、亲手运行并尝试修改；
3. 只有在无法定位问题时，才打开同名的运行对照查看当前小节；
4. 不直接编辑这里的文件。修改学习版后，通过课程执行脚本重新生成对照。

分类如下：

| 目录 | 内容 |
|---|---|
| [`asr_core/`](asr_core/) | ASR 主线第 01～46 课 |
| [`pytorch_foundations/`](pytorch_foundations/) | Python/PyTorch 桥梁课与代码伴读 |
| [`audio_foundations/`](audio_foundations/) | 音频零基础 01～06 的完整运行对照 |
| [`language_models/`](language_models/) | 语言模型专修第 01～09 课 |
| [`labs/`](labs/) | 学习中枢与专题实验室 |
| [`capstone/`](capstone/) | 实时数字 CTC 结课项目 |

维护命令：

```powershell
uv run python scripts/execute_course_labs.py --write-executed
uv run python scripts/validate_course.py
```

具体输出路径由 `scripts/notebook_layout.py` 统一决定，构建脚本不应自行拼接 `_已运行.ipynb` 路径。
