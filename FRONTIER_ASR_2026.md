# 2026 前沿 ASR 学习地图

更新时间：2026-08-17

## 先说结论

“声学模型”这个词来自传统流水线。现代端到端 ASR 中，声学、对齐和一部分语言建模已经联合训练；到了 Large Audio-Language Model（LALM），音频编码器还会连接大语言模型。因此没有脱离场景的唯一“最前沿模型”。

- 实时、端侧、可控延迟：主线仍是流式 Conformer/FastConformer 编码器 + CTC、RNN-T 或 TDT。
- 离线、多语种、方言、噪声、长音频和上下文：前沿是音频编码器 + projector + LLM 的 LALM。
- 初学顺序：声音与 Log-Mel → 张量与编码器 → CTC → Conformer → 流式 → RNN-T/TDT → 自监督预训练 → LALM。

## 一条语音怎样变成文字

```text
16 kHz 波形
  → 分帧、加窗、FFT、Mel 滤波、取 log
  → Log-Mel 特征 [时间 T, 频率 F]
  → 声学编码器（Conv / Conformer / Transformer）
  → 更短、更有语义的隐藏序列 [T', D]
  → 解码目标（CTC / RNN-T / AED / LLM next-token）
  → token 序列
  → 文本规范化、时间戳、置信度
```

模型真正学习的是：哪些短时频谱变化对应音素/字词，以及在上下文中哪串文字最可能。训练时反向传播会调整数百万到数十亿个参数，使正确转录的损失变小。

## 四类必须分清的架构

| 架构 | 核心思想 | 优点 | 主要代价 | 典型用途 |
|---|---|---|---|---|
| CTC | 每帧预测 token/blank，再合并重复并删除 blank | 简单、并行、易训练、易流式 | 输出之间依赖弱，常需 LM/热词 | 教学、端侧、快速离线 |
| RNN-T | 声学编码器 + 已输出 token 的预测网络 + joiner | 原生流式、输出依赖更强 | loss/解码和状态管理更复杂 | 手机、实时语音助手 |
| AED | 编码器读取音频，注意力解码器逐 token 生成 | 离线准确率强、多任务自然 | 通常非流式，可能漏段或幻觉 | 离线转录、翻译 |
| LALM | 音频编码器把连续语音表示接入 LLM | 上下文、方言、多语种、长音频、知识能力强 | 数据/算力大，生成式幻觉与延迟需治理 | 现代通用 ASR、语音理解 |

TDT 是 Transducer 的高效变体：同时预测 token 和持续时长，跳过大量 blank 帧，从而减少解码步骤。

## 2026 年代表性前沿

### Qwen3-ASR：开放 LALM 代表

Qwen3-ASR-0.6B/1.7B 共支持 52 种语言和方言（官方拆分为 30 种语言、22 种中文方言）。其结构是：

```text
128 维 Fbank
  → AuT 音频编码器（8 倍下采样，12.5 Hz 输出）
  → projector
  → Qwen3-0.6B 或 Qwen3-1.7B
  → 自回归文字 token
```

动态注意力窗在 1～8 秒之间，使同一模型可以兼顾短 chunk 流式推理和长输入离线推理。训练分为 AuT 预训练、Qwen3-Omni 多模态预训练、ASR SFT 和 GSPO 强化学习。官方报告称 AuT 预训练使用约 4,000 万小时伪标注数据，所以本课程不会假装可以从零复现同等基础模型；我们会先复现结构和损失，再用开源权重做推理/微调。

### FastConformer + CTC/RNN-T/TDT：工业实时代表

FastConformer 用卷积捕获局部声学模式，用自注意力建模长上下文，并通过更激进的下采样降低时间序列长度。CTC 最简单；RNN-T 原生流式；TDT 进一步预测持续时长以跳过冗余 blank。它们通常比通用 LALM 更容易达到可预测的低延迟和稳定增量输出。

## 我们怎样“搭建”，而不是只会调用

### 第 1 层：亲手实现最小系统

1. 从 WAV 得到 Log-Mel。
2. 实现小型卷积/Conformer 编码器。
3. 用 CTC loss 在小数据上过拟合，确认数据、shape、对齐和梯度全通。
4. 实现 greedy 与 prefix beam search，用 CER/WER 评估。

### 第 2 层：做成真正流式

1. 音频按 chunk 输入，保留特征尾巴。
2. 编码器使用因果卷积、有限右上下文和 KV/cache。
3. 解码器跨 chunk 保存 prefix/state。
4. 同时测 CER/WER、RTF、首字延迟、尾延迟和 partial 稳定性。

### 第 3 层：进入前沿

1. 学 Conformer、RNN-T 和 TDT，比较质量—延迟曲线。
2. 学 wav2vec 2.0/自监督预训练，理解无标注音频的价值。
3. 搭建迷你 LALM：预训练音频编码器 → 下采样/projector → 小语言模型 → ASR SFT。
4. 使用 Qwen3-ASR 等开放权重做领域微调，并专门评估幻觉、专名、数字、噪声和长音频。

## 本仓库的实际学习顺序

1. `notebooks/零基础预备课_Python与PyTorch.ipynb`
2. `notebooks/01_声音与采样_从零开始.ipynb` 到第 14 课
3. `notebooks/专题_CTC可视化实验室_从路径到流式解码.ipynb`
4. 第 15～18 课和流式专题实验室
5. 第 19～30 课：语言模型、WFST、系统与部署
6. 第 42～46 课：Conformer、RNN-T/TDT、自监督预训练、迷你 LALM 与 Qwen3-ASR 真实项目验收

第 46 课提供 Qwen3-ASR 开放权重的推理、微调与验收路线；实际运行仍需单独下载模型、准备许可清晰的数据并按目标硬件评估，课程不伪造未发生的大模型训练结果。

学习标准不是“运行成功”，而是能闭卷解释、从空白实现、定位失败并把方法迁移到新音频。

## 主要资料

- Qwen3-ASR Technical Report: https://arxiv.org/abs/2601.21337
- Qwen3-ASR 官方代码: https://github.com/QwenLM/Qwen3-ASR
- Fast Conformer: https://arxiv.org/abs/2305.05084
- Seed-ASR: https://arxiv.org/abs/2407.04675
- FunASR: https://github.com/modelscope/FunASR

注意：厂商的“最佳”结论往往依赖其数据、硬件和评测规范。任何模型选型都必须在自己的中文、方言、噪声、设备和延迟约束上重新测 CER/WER 与性能。
