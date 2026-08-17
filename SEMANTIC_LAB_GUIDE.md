# 语义后处理专题学习顺序

打开 `notebooks/专题_语义后处理实验室_时间戳ITN置信度与安全执行.ipynb`。

建议分四次学习：

1. 第一次（约 75 分钟）：时间坐标映射、真实 FSDD speaker embedding 和 DER；
2. 第二次（约 75 分钟）：标点/ITN、原始证据与黄金集测试；
3. 第三次（约 90 分钟）：置信度校准、reliability diagram、N-best oracle 与语义重排；
4. 第四次（约 90 分钟）：NLU、业务校验、安全门、LLM grounding、验收矩阵和 56 分测试。

通过标准：

- Notebook 从头执行且所有断言通过；
- 能从 encoder step 正确映射到带流式 offset 的输入时间；
- 能解释 speaker recognition 与 diarization 的边界，以及 DER 的三类错误和分母；
- 教学 ITN 黄金集 4/4 通过，同时能主动给出至少 3 个失败例；
- temperature scaling 只在 validation 上拟合，并使独立 test ECE 明显下降；
- 能解释 1-best WER、N-best oracle WER、重排后 WER 的区别；
- 能让低声学置信、高风险动作和不合法槽位分别进入正确的 clarify/reject 路径；
- 闭卷测试至少 45/56，完成末尾 5 分钟口述题。

本实验里的 speaker embedding、ITN、NLU 和 grounding 都是可解释教学基线，不代表生产能力。真实上线必须换成目标场景数据和模型，并联合报告 ASR、后处理、任务成功、误执行、延迟、资源与回滚证据。
