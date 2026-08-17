# FSDD 六折 LOSO 学习指南

主 Notebook：`notebooks/专题_FSDD六折LOSO_嵌套选择与说话人统计.ipynb`。

这节紧接固定 speaker-disjoint 实验，解决“只有一位 test speaker，结果能否代表新人”的问题。六位 FSDD speaker 轮流做外层 test；每折用另一位 dev 在 clean 与 gain+noise 间选择候选，其余四位训练。候选阶段不接收 test 录音，选择完成后才构造外测特征。

## 推荐顺序

1. 先闭卷回答课前 6 题；
2. 运行折叠矩阵和 test gate 断言；
3. 逐折看训练曲线，只依据 dev 解释选择；
4. 看 speaker 柱图、dev-test 散点、micro/macro 与 cluster 区间；
5. 听困难 speaker，做错误对齐、长度分桶与 chunk 实验；
6. 完成 96 分测试、6 个迁移任务和 10 分钟答辩。

## 重现实验

```powershell
uv run python -m fsdd_generalization.loso
```

默认训练 12 个候选（6 folds × 2 candidates），会逐折输出 dev 曲线并保存增量结果。最终产物：

- `artifacts/fsdd_loso_results.json`
- `artifacts/fsdd_loso_checkpoints/fold_*.pt`

## 当前结论边界

六折结果用于描述 FSDD 内部的 speaker variability，并评估当前 dev-only 选择程序。上一实验已经使用或查看全部 FSDD speaker，因此这不是新的 untouched benchmark。下一阶段必须先冻结协议，再使用从未参与课程设计的新 speaker / 新语料一次性外测。
