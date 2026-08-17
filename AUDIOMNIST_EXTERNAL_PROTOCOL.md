# AudioMNIST 外部盲测预注册协议

机器可检验的冻结协议：`external_evaluation/audiomnist_protocol.json`。

本文件和 JSON 必须在下载、试听或评分 AudioMNIST 音频之前进入 Git 历史。提交之后只允许补充下载归档 SHA256、文件 manifest 和运行环境证据，不允许根据 AudioMNIST 结果改变模型、epoch、增强、预处理、decoder、指标、分桶或停止规则后仍称其为同一次 untouched evaluation。

## 为什么选择 AudioMNIST

AudioMNIST 与 FSDD 都是英语数字 0～9，因此没有更换标签任务；但 AudioMNIST 来自独立项目，包含 60 位不同 speaker、30,000 条录音，原始采样率与录音设备也不同。官方仓库采用 MIT 许可。本实验固定官方 revision：`630d7dab4c040882834de6fa21baf9a60372accd`。

## 在接触外测之前冻结的模型

- 训练数据：完整 FSDD 六位 speaker、3,000 条录音；
- 模型：48 hidden、5 层、kernel 5 的 causal Conv CTC；
- 增强：每条合成训练序列增加一份随机增益与白噪声副本；
- epoch：已发布 FSDD LOSO 六折选中模型的最佳 epoch 为 `[7, 11, 11, 8, 11, 7]`，中位数 9.5，按 round-half-up 固定为 10；
- 最终拟合不再使用 dev，也不读取任何 AudioMNIST 内容；
- 解码：greedy CTC，无语言模型。

## 一次性外测

1. 对全部 30,000 条单数字录音评分；
2. 每位 speaker 固定构造 100 条 1～4 位数字序列，共 6,000 条；
3. 报告 micro 与 speaker-macro、每位 speaker、长度、S/D/I、元数据描述性分桶；
4. 对 60 条固定代表录音验证多种 chunk 的流式一致性；
5. 在单线程 CPU 上报告预加载波形的 P50/P95/RTF；
6. 不因结果难看而删除、修改或重跑模型。

## 解释边界

这仍是孤立数字任务，不代表开放词汇、长语音、多语言或生产环境 ASR。AudioMNIST 一旦评分，就成为已接触数据；后续适配必须单独建立 train/dev，并寻找另一份从未接触的外部 test 才能再次声称盲测。
