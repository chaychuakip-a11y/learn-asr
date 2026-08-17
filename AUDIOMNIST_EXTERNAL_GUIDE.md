# AudioMNIST 外部盲测学习指南

主 Notebook：`notebooks/专题_AudioMNIST外部盲测_冻结协议跨域失败与适配边界.ipynb`。

这节课连接三份可审计证据：

1. `0b30360`：接触音频前冻结 protocol；
2. `cac8aa6`：评分前冻结最终 FSDD model/checkpoint；
3. `artifacts/audiomnist_external_results.json`：首次且唯一一次全量外测。

## 学习顺序

1. 先写 6 个课前预测；
2. 看 Git 时间线与 hash 断言；
3. 听 48 kHz 真人音频并观察 48→8 kHz 重采样；
4. 看主分数、60 位 speaker 分布和区间；
5. 分析 S/D/I、混淆矩阵、错误音频和多数字长度；
6. 学会正确解释 metadata slices、chunk invariance 和 RTF；
7. 完成 8 个编程任务、120 分测试和 12 分钟答辩。

## 数据与复现

完整数据约需 1.85 GB 解压空间：

```powershell
uv run python -m external_evaluation.data
```

首次评分已经完成。评分器现在会拒绝覆盖现有结果，防止把第二次运行冒充首次盲测：

```powershell
uv run python -m external_evaluation.evaluate
```

若只是学习，请直接加载已发布 JSON，不要删除结果文件重跑。

## 结论边界

单数字 exact 59.30%、CER 40.85%，按预注册定义属于 weak transfer。AudioMNIST 现已是 contacted evaluation data；下一步适配必须另立协议，最终结论还需要另一份独立来源确认。
