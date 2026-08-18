# OpenFst、KenLM 与 ASR 语言模型速查表

这份速查表用于做实验和排错，不替代 9 课 Notebook 中的推导。

## 1. N-gram 最小公式集

链式法则：

```text
P(w1…wT) = ∏t P(wt | w1…w(t-1))
```

N-gram 近似只保留最近 `N-1` 个词：

```text
Bigram:  P(wt | w(t-1))
Trigram: P(wt | w(t-2), w(t-1))
```

最大似然 Bigram：

```text
P(w | h) = count(h, w) / count(h)
```

Add-k 平滑：

```text
P(w | h) = (count(h, w) + k) / (count(h) + k|V|)
```

困惑度：

```text
PPL = exp(-1/T · Σt ln P(wt | history))
```

只在相同 tokenization、词表、边界符和测试集上比较 PPL。PPL 更低通常表示文本预测更好，不保证最终 WER 一定更低。

## 2. 四个容易混淆的词

| 名称 | 解决什么 | 核心动作 |
|---|---|---|
| Smoothing | 见过的和没见过的事件怎样重新分配概率质量 | 修改概率估计 |
| Backoff | 高阶 N-gram 缺失时怎么办 | 乘回退权重，再查询低阶项 |
| Interpolation | 是否同时使用多阶模型 | 对多阶概率加权求和 |
| OOV / `<unk>` | 测试词不在固定词表中怎么办 | 训练时建立 `<unk>` 类，并保持训练/测试映射一致 |

## 3. 概率、ARPA 和 FST 权重方向

KenLM ARPA 通常保存 `log10(P)`；OpenFst tropical 图常保存自然对数下的负对数代价：

```text
fst_cost = -ln(P) = -log10(P) × ln(10)
```

因此：

- 概率越大，代价越小；
- 一条路径的弧代价相加，等价于路径概率相乘后取负对数；
- 最短路径对应最高概率路径；
- 不要直接把 ARPA 的负 `log10(P)` 当成 OpenFst 的自然对数代价。

常见融合形式，如果所有量都写成“越小越好”的代价：

```text
total_cost = acoustic_cost + α · lm_cost + β · word_count + γ · bias_cost
```

如果某个库给的是“越大越好”的 log score，先确认符号再融合。LM scale `α` 和插词项 `β` 必须在开发集上选择，然后冻结。

## 4. OpenFst 文本格式

FST 弧：

```text
源状态 目标状态 输入标签 输出标签 [权重]
0       1       ni       你       0.2
```

FSA/acceptor 弧：

```text
源状态 目标状态 标签 [权重]
0       1       你   0.2
```

终止状态：

```text
状态 [终止权重]
1     0.0
```

注意：状态编号和 symbol id 是两套编号；OpenFst symbol table 必须把 `<eps>` 放在 id 0。

## 5. 常用命令

```bash
# 文本 → 二进制 FST
fstcompile \
  --isymbols=words.txt --osymbols=words.txt \
  --keep_isymbols=true --keep_osymbols=true \
  G.txt G.fst

# 打印、查看统计和画图
fstprint G.fst
fstinfo G.fst
fstdraw --portrait=true G.fst G.dot
dot -Tsvg G.dot -o G.svg

# 为 composition 排序：左图输出，右图输入
fstarcsort --sort_type=olabel L.fst L.sorted.fst
fstarcsort --sort_type=ilabel G.fst G.sorted.fst
fstcompose L.sorted.fst G.sorted.fst LG.fst

# 图变换
fstrmepsilon input.fst noeps.fst
fstdeterminize input.fst det.fst
fstminimize det.fst min.fst
fstproject --project_type=output input.fst output.fst

# 最短路径 / N-best
fstshortestpath input.fst best.fst
fstshortestpath --nshortest=5 --unique=true input.fst best5.fst
fstprint best.fst
```

高频诊断顺序：`fstinfo` 看图是否为空，再 `fstprint` 看标签和权重，最后才跑 shortest path。`fstequivalent` 可用于满足其前提的等价性检查，但不能代替 symbol table、随机性、路径覆盖率和数值容差检查。

## 6. Composition 的接口规则

```text
左图：input → middle
右图：middle → output
组合：input → output
```

`L : token/phone → word`，`G : word → word`，所以 `L∘G : token/phone → word`。要连接的“左输出标签”和“右输入标签”必须是同一套符号及 id。

```text
L --按 olabel 排序--┐
                    ├─ compose → LG
G --按 ilabel 排序--┘
```

图为空时优先检查中间字母表，不要先怀疑最短路径算法。

## 7. 关键符号

| 符号 | 含义 | 常见位置 |
|---|---|---|
| `<eps>` | epsilon：不消费或不输出普通符号 | 所有 FST；id 必须为 0 |
| `<s>` | 句首历史 | ARPA/G |
| `</s>` | 句尾事件 | ARPA/G |
| `<unk>` | 固定词表之外的词类 | LM 训练和测试 |
| `#0` | 常用于 G 的 backoff/disambiguation 接口 | G 及 L 上配套自环 |
| `#1`, `#2`… | 区分词典中会造成非确定性的发音路径 | L 的词典消歧 |

`#0` 和 `#1` 不能因为都以 `#` 开头就混用。课程第 6 课会删除 `#0:#0` 自环，直接观察合法路径怎样消失。

## 8. HCLG 与 CTC-TLG

| 图 | 输入 | 输出 | 职责 |
|---|---|---|---|
| `G` | word | word | N-gram 语法和 LM 权重 |
| `L` | phone/token | word | 发音词典和词边界 |
| `C` | context-dependent phone | phone | 音素上下文依赖 |
| `H` | transition-id | context-dependent phone | HMM 状态与声学模型接口 |
| `T` | CTC label/frame transition | token | blank、重复和 CTC 拓扑 |

传统 HMM-DNN 解码常写：

```text
HCLG = H ∘ C ∘ L ∘ G
```

CTC 系统常写：

```text
TLG = T ∘ L ∘ G
```

不同工具对 `T` 的具体定义可能不同；不要只看图名，要检查每一层输入/输出符号的接口。

## 9. N-best、lattice 和二遍模型

- Beam search 状态：解码过程中的活动假设，不等于最终 N-best。
- N-best：完整候选句子的有限列表。
- Lattice：共享前缀/后缀的紧凑候选图，通常能保留更多路径。
- Oracle WER@K：前 K 个候选中，与参考文本最接近者的 WER。
- 二遍重排只能从保留下来的候选中选择；正确路径已经被剪掉时，再强的 LLM 也无法“只重排”救回。

先报告 oracle WER 和候选覆盖率，再报告重排后的 1-best WER。

## 10. 十步排错清单

1. `fstinfo` 的状态数和弧数是否为 0？
2. `<eps>` 是否为 symbol id 0？
3. compose 的左输出表与右输入表是否逐项同 id？
4. 左图是否按 `olabel`、右图是否按 `ilabel` 排序？
5. `<s>`、`</s>`、`<unk>` 是否在训练和评分阶段一致？
6. ARPA 的 `log10` 是否正确转换为 `-ln`？
7. backoff 权重是否只在高阶项缺失时使用？
8. `#0` 自环和词典消歧符号是否正确保留到该删除的阶段？
9. LM scale、插词项和 beam 是否只在开发集调整？
10. 1-best 变差时，oracle WER 是否说明正确路径仍在候选中？

## 11. 现代系统的最小安全结构

```text
流式 CTC/RNN-T 声学模型
        ↓
N-gram/WFST 一遍解码（低延迟、确定性保底）
        ↓ 保留 lattice/N-best/后验/时间戳
检索相关联系人、术语和文档上下文
        ↓
音频条件的二遍纠错或 SpeechLM 重排
        ↓
置信度、幻觉与覆盖风险门控
        ├─ 可信：采用二遍结果
        └─ 不可信：回退一遍结果或请求确认
```

“语言更流畅”不是唯一目标。前沿系统至少同时看 WER、实体准确率、延迟/RTF、上下文误偏置、幻觉率与拒绝覆盖率。
