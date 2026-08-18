# PyTorch 零基础课程：进入 ASR 主线前的 6 节桥梁课

这条路线面向几乎没有 Python/PyTorch 基础的学习者。先完成 10 分钟导学，再按 1～6 顺序学习；每节都有无输出学习版、已运行对照版、课前诊断、10 道练习、离场票和间隔复习。

如果你看到代码经常不知道每行在做什么，先使用 [零基础逐行代码伴读](代码伴读_零基础逐行理解ASR.ipynb)｜[已运行对照](_executed/pytorch_foundations/代码伴读_零基础逐行理解ASR_已运行.ipynb)。其中 14 个可运行小节专门解释变量、函数、点号调用、shape/axis、dtype/device、`nn.Module`、训练循环、padding、声学前端、CTC、流式 cache 和系统化排错。题目可以之后再做。

## 先做代码伴读与导学

- [零基础预备课：Python 与 PyTorch 的第一步](零基础预备课_Python与PyTorch.ipynb)：变量、Tensor、shape、最小训练循环和即时判题。

## 6 节基础主课

| 课次 | 学习版 | 已运行对照 | 核心能力 | 通关证据 |
|---:|---|---|---|---|
| 1 | [Python 最小语法与函数](基础_01_Python最小语法与函数.ipynb) | [查看运行输出](_executed/pytorch_foundations/基础_01_Python最小语法与函数_已运行.ipynb) | 变量、列表、循环、函数、异常、断言 | 独立实现并测试单位换算函数 |
| 2 | [Tensor 创建、索引与 Shape](基础_02_Tensor创建索引与Shape.ipynb) | [查看运行输出](_executed/pytorch_foundations/基础_02_Tensor创建索引与Shape_已运行.ipynb) | dtype、device、`[B,T,F]`、索引、维度 | 对任意 Tensor 解释每个轴 |
| 3 | [Tensor 运算、广播与维度变换](基础_03_Tensor运算广播与维度变换.ipynb) | [查看运行输出](_executed/pytorch_foundations/基础_03_Tensor运算广播与维度变换_已运行.ipynb) | 归约、广播、reshape、permute、矩阵乘法 | 预测每步 shape 并排查维度错误 |
| 4 | [Autograd、损失与优化器](基础_04_Autograd损失与优化器.ipynb) | [查看运行输出](_executed/pytorch_foundations/基础_04_Autograd损失与优化器_已运行.ipynb) | loss、梯度、backward、zero_grad、step | 从空白重写最小训练循环 |
| 5 | [nn.Module 与训练验证循环](基础_05_nnModule与训练验证循环.ipynb) | [查看运行输出](_executed/pytorch_foundations/基础_05_nnModule与训练验证循环_已运行.ipynb) | forward、参数注册、train/eval、state_dict | 实现训练/验证步骤并保存加载 |
| 6 | [Dataset、DataLoader 与变长语音 Batch](基础_06_Dataset_DataLoader与变长语音Batch.ipynb) | [查看运行输出](_executed/pytorch_foundations/基础_06_Dataset_DataLoader与变长语音Batch_已运行.ipynb) | collate、padding、length、mask、batch contract | 构造变长 batch 并验证有效区域 |

## 学习规则

1. 第一次只打开学习版；运行前写下输出或 shape 预测。
2. 每次只修改一个变量，记录“预测—结果—解释”。
3. 每课 10 道练习按 0～2 分评分，达到 16/20 再继续。
4. 已运行版只用于核对，不替代自己执行。
5. 错题记录到根目录 `LEARNING_LOG.md`，并在第 1、7、30 天复习。

## 进入 ASR 主线的门槛

满足以下条件后进入 [第 1～41 课核心索引](核心课程索引_第01到41课.md)：

- 能解释 `[B,T,F]`、dtype 和 device；
- 能从空白写出预测、loss、`zero_grad()`、`backward()`、`step()`；
- 能使用 `Dataset`/`DataLoader` 构造带 lengths 与 mask 的变长 batch；
- 能根据报错定位 Python 语法、shape、dtype、device 或梯度中的哪一类问题。

基础 6 完成后从第 1 课学习声音与采样；到第 7～9 课时，会把基础能力用于真正的声学编码器。
