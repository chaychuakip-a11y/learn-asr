# FSDD 说话人泛化实验学习指南

主 Notebook：`notebooks/专题_FSDD说话人泛化实验_数据划分增强与盲测.ipynb`。

建议分五次学习：

1. 数据版本、SHA256、manifest、许可与 speaker-disjoint split；
2. 真人录音差异、多 token utterance 构造与变长 CTC batch；
3. baseline/augmentation、train/dev/test 职责与 early stopping；
4. CER、exact、长度分桶、插删替错误和冻结样例；
5. 流式数值等价、partial、RTF、统计边界和 72 分答辩。

准备固定 FSDD 数据：

```powershell
uv run python -m fsdd_generalization.data
```

重现实验（首次运行需要数分钟）：

```powershell
uv run python -m fsdd_generalization.run_experiment --epochs 24
```

快速只重训增强版本：

```powershell
uv run python -m fsdd_generalization.run_experiment --epochs 24 --skip-baseline
```

快速模式写入独立的 `*_augmented_only` checkpoint 和 results，不会覆盖正式的 baseline/augmented 对照产物。

本实验固定 train speakers 为 george/jackson/lucas/nicolas，dev 为 theo，test 为 yweweler。看到当前 test 结果后，不应继续用 yweweler 调下一版并仍称其为盲测；下一次模型改动应使用预先定义的新外层 speaker fold 或新数据。

通过标准：数据和单元测试全部通过；能从空白写出 split 泄漏断言与 CTC collate；能解释增强为什么改善但仍未解决泛化；72 分测试至少 58 分，并完成末尾 10 分钟实验答辩。
