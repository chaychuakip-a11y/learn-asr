from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "专题_WFST实验室_从L与G到流式TokenPassing.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
    md(
        """
# WFST 实验室：从 L 与 G 到流式 Token Passing

这是第 19～23 课的贯通专题。我们不用大型 OpenFst 图，而用几十条弧的纯 Python 小图，把每个状态和分数都看清楚。

完成后你应该能够：

1. 区分 acceptor、transducer、输入标签、输出标签和权重；
2. 解释概率乘法为什么变成负对数代价加法；
3. 构造词典 transducer `L` 与 Bigram acceptor `G`；
4. 实际执行 `L ∘ G`，检查中间符号表；
5. 用最短路径在 `night/knight` 同音词之间选择；
6. 解释 LM scale、插词项、热词偏置和剪枝的风险；
7. 生成 N-best/lattice，并跨 chunk 保存 token-passing 状态；
8. 说清传统 HMM `HCLG` 与 CTC 解码图的关系和区别。

本实现只处理教学所需的输出 epsilon，不替代 OpenFst/k2 的完整 epsilon filter、semiring、determinize 和 minimize 实现。
"""
    ),
    md(
        """
## 0. 先把五个对象分开

| 对象 | 输入 | 输出 | 作用 |
|---|---|---|---|
| FSA/acceptor | token | 同一 token | 判断序列是否被接受并累计代价 |
| FST/transducer | 输入 token | 输出 token | 把一种符号序列映射成另一种 |
| L | phone/token | word | 发音词典 |
| G | word | word | 语言模型约束和代价 |
| L∘G | phone/token | word | 同时满足词典和语言模型 |

传统 Kaldi HMM 系统常见 `H ∘ C ∘ L ∘ G`。CTC 系统通常用 CTC topology/token graph 替代传统 H/C 的部分职责，具体名字取决于工具链；不能看到“CTC head”就把所有图都称为 HCLG。
"""
    ),
    code(
        """
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
import math

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from IPython.display import clear_output, display

EPS = "<eps>"

@dataclass(frozen=True)
class Arc:
    src: int
    dst: int
    ilabel: str
    olabel: str
    weight: float = 0.0

class WFST:
    def __init__(self, start=0):
        self.start = start
        self.arcs = []
        self.finals = {}

    def add_arc(self, src, dst, ilabel, olabel, weight=0.0):
        self.arcs.append(Arc(src, dst, ilabel, olabel, float(weight)))

    def set_final(self, state, weight=0.0):
        self.finals[int(state)] = float(weight)

    def outgoing(self, state):
        return [arc for arc in self.arcs if arc.src == state]

    @property
    def states(self):
        result = {self.start, *self.finals}
        for arc in self.arcs:
            result.update([arc.src, arc.dst])
        return sorted(result)

    def summary(self, name="FST"):
        print(f"{name}: states={len(self.states)}, arcs={len(self.arcs)}, start={self.start}, finals={self.finals}")

print("最小 WFST 数据结构准备完成")
"""
    ),
    md(
        """
## 1. 概率域、tropical semiring 与 log semiring

一条路径上的概率相乘。使用代价 `cost = -log(probability)` 后，路径代价可以相加，较高概率对应较低代价。

- **Viterbi/最短路径**关心最好的一条路径：多个候选之间取 `min`；
- **forward/总概率**关心所有路径之和：在 log 域使用 LogSumExp；
- 二者不能混用。WFST 的 semiring 决定“沿路径”和“路径之间”分别怎样组合。
"""
    ),
    code(
        """
path_probabilities = np.array([0.30, 0.20])
path_costs = -np.log(path_probabilities)
viterbi_cost = path_costs.min()
total_probability = path_probabilities.sum()
forward_cost = -math.log(total_probability)

print("两条路径概率：", path_probabilities)
print("对应代价：", path_costs)
print("Viterbi 只保留最好路径：P=", math.exp(-viterbi_cost))
print("Forward 合并两条路径：P=", math.exp(-forward_cost))
assert np.isclose(math.exp(-forward_cost), 0.5)
"""
    ),
    md(
        """
## 2. 构造 L：phone 到 word 的词典 transducer

我们使用四个词：

- `GOOD → G UH D`
- `MEDIEVAL → M EH D IY V AH L`
- `NIGHT → N AY T`
- `KNIGHT → N AY T`

`NIGHT` 与 `KNIGHT` 的输入 phone 完全相同，所以 L 会产生两个候选。声学输入本身无法决定拼写，需要 G 的上下文。
"""
    ),
    code(
        """
LEXICON = {
    "GOOD": ["G", "UH", "D"],
    "MEDIEVAL": ["M", "EH", "D", "IY", "V", "AH", "L"],
    "NIGHT": ["N", "AY", "T"],
    "KNIGHT": ["N", "AY", "T"],
}

def build_lexicon_fst(lexicon):
    fst = WFST(start=0)
    fst.set_final(0, 0.0)
    next_state = 1
    for word, phones in lexicon.items():
        state = 0
        for index, phone in enumerate(phones):
            last = index == len(phones) - 1
            destination = 0 if last else next_state
            if not last:
                next_state += 1
            output = word if last else EPS
            fst.add_arc(state, destination, phone, output, 0.0)
            state = destination
    return fst

L = build_lexicon_fst(LEXICON)
L.summary("L")
print("NIGHT/KNIGHT 的词典路径具有相同输入，但最后输出词不同。")
for arc in L.arcs:
    if arc.ilabel in {"N", "AY", "T"}:
        print(arc)
"""
    ),
    md(
        """
## 3. 构造 G：带 Bigram 代价的 word acceptor

这个 G 只接受两词短句：

- `GOOD NIGHT` 概率高，`GOOD KNIGHT` 概率低；
- `MEDIEVAL KNIGHT` 概率高，`MEDIEVAL NIGHT` 概率低。

G 的状态可理解为语言模型历史。弧权重使用 `-log P(word | history)`。
"""
    ),
    code(
        """
def neglog(probability):
    if not 0 < probability <= 1:
        raise ValueError("概率必须位于 (0,1]")
    return -math.log(probability)

def build_bigram_g():
    fst = WFST(start=0)
    # 0=<s>, 1=GOOD, 2=MEDIEVAL, 3/4=句末候选
    fst.add_arc(0, 1, "GOOD", "GOOD", neglog(0.55))
    fst.add_arc(0, 2, "MEDIEVAL", "MEDIEVAL", neglog(0.45))
    fst.add_arc(1, 3, "NIGHT", "NIGHT", neglog(0.90))
    fst.add_arc(1, 3, "KNIGHT", "KNIGHT", neglog(0.10))
    fst.add_arc(2, 4, "NIGHT", "NIGHT", neglog(0.08))
    fst.add_arc(2, 4, "KNIGHT", "KNIGHT", neglog(0.92))
    fst.set_final(3, 0.0)
    fst.set_final(4, 0.0)
    return fst

G = build_bigram_g()
G.summary("G")
for arc in G.arcs:
    print(f"{arc.src} -> {arc.dst}  {arc.ilabel:9s}  cost={arc.weight:.3f}  P={math.exp(-arc.weight):.2f}")
"""
    ),
    code(
        """
def draw_fst(fst, positions, title):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for state, (x, y) in positions.items():
        final = state in fst.finals
        ax.scatter([x], [y], s=900, facecolor="white", edgecolor="black", linewidth=2)
        if final:
            ax.scatter([x], [y], s=700, facecolor="none", edgecolor="black", linewidth=1)
        ax.text(x, y, str(state), ha="center", va="center")
    parallel_count = defaultdict(int)
    for arc in fst.arcs:
        key = (arc.src, arc.dst)
        parallel_count[key] += 1
        rad = 0.18 * (parallel_count[key] - 1)
        x1, y1 = positions[arc.src]
        x2, y2 = positions[arc.dst]
        arrow = FancyArrowPatch((x1 + 0.12, y1), (x2 - 0.12, y2), arrowstyle="->",
                                mutation_scale=14, connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(arrow)
        ax.text((x1+x2)/2, (y1+y2)/2 + rad + 0.08,
                f"{arc.ilabel}:{arc.olabel}/{arc.weight:.2f}", ha="center", fontsize=9)
    sx, sy = positions[fst.start]
    ax.annotate("start", xy=(sx-0.28, sy), xytext=(sx-0.9, sy), arrowprops={"arrowstyle": "->"})
    ax.set(title=title, xlim=(-1, 5), ylim=(-1.3, 1.4), aspect="equal")
    ax.axis("off")
    plt.show()

G_POSITIONS = {0: (0, 0), 1: (1.6, 0.7), 2: (1.6, -0.7), 3: (3.7, 0.7), 4: (3.7, -0.7)}
draw_fst(G, G_POSITIONS, "Bigram language-model acceptor G（标签格式 input:output/cost）")
"""
    ),
    md(
        """
## 4. Composition：L 的输出必须匹配 G 的输入

组合状态是状态对 `(l_state, g_state)`。当 L 输出普通 word 时，只能与 G 中输入标签相同的弧配对；当 L 输出 epsilon 时，只推进 L，G 状态保持不动。

完整 OpenFst composition 还要处理双方 epsilon、过滤器、排序和 semiring。本实验只实现当前 L/G 结构所需的安全子集。
"""
    ),
    code(
        """
def interface_report(left, right):
    left_outputs = {arc.olabel for arc in left.arcs if arc.olabel != EPS}
    right_inputs = {arc.ilabel for arc in right.arcs if arc.ilabel != EPS}
    return {
        "left_outputs": left_outputs,
        "right_inputs": right_inputs,
        "missing_in_right": left_outputs - right_inputs,
        "unused_in_right": right_inputs - left_outputs,
    }

def compose_left_output_epsilon(left, right, lm_scale=1.0, word_boosts=None):
    word_boosts = {} if word_boosts is None else dict(word_boosts)
    composed = WFST(start=0)
    pair_to_state = {(left.start, right.start): 0}
    queue = deque([(left.start, right.start)])

    def state_for(pair):
        if pair not in pair_to_state:
            pair_to_state[pair] = len(pair_to_state)
            queue.append(pair)
        return pair_to_state[pair]

    while queue:
        left_state, right_state = queue.popleft()
        source = pair_to_state[(left_state, right_state)]
        if left_state in left.finals and right_state in right.finals:
            composed.set_final(source, left.finals[left_state] + lm_scale * right.finals[right_state])
        for left_arc in left.outgoing(left_state):
            if left_arc.olabel == EPS:
                pair = (left_arc.dst, right_state)
                composed.add_arc(source, state_for(pair), left_arc.ilabel, EPS, left_arc.weight)
                continue
            for right_arc in right.outgoing(right_state):
                if left_arc.olabel != right_arc.ilabel:
                    continue
                pair = (left_arc.dst, right_arc.dst)
                boost = float(word_boosts.get(right_arc.olabel, 0.0))
                weight = left_arc.weight + lm_scale * right_arc.weight - boost
                composed.add_arc(source, state_for(pair), left_arc.ilabel, right_arc.olabel, weight)
    composed.state_pairs = {state: pair for pair, state in pair_to_state.items()}
    return composed

print("符号接口检查：", interface_report(L, G))
LG = compose_left_output_epsilon(L, G)
LG.summary("L∘G")
print("组合状态示例：", list(LG.state_pairs.items())[:8])
"""
    ),
    md(
        """
## 5. Token passing：给定 phone 输入搜索最低代价 word 路径

一个 token/hypothesis 至少携带当前图状态、累计代价、输出词序列和回溯信息。这里为便于观察直接保存整条弧；生产解码器通常使用紧凑 backpointer。
"""
    ),
    code(
        """
@dataclass(frozen=True)
class Hypothesis:
    cost: float
    state: int
    words: tuple[str, ...]
    arcs: tuple[Arc, ...]

def start_hypotheses(fst):
    return [Hypothesis(0.0, fst.start, (), ())]

def advance_hypotheses(fst, hypotheses, input_labels, beam_size=100):
    active = list(hypotheses)
    for label in input_labels:
        next_active = []
        for hypothesis in active:
            for arc in fst.outgoing(hypothesis.state):
                if arc.ilabel != label:
                    continue
                words = hypothesis.words + (() if arc.olabel == EPS else (arc.olabel,))
                next_active.append(Hypothesis(
                    hypothesis.cost + arc.weight, arc.dst, words, hypothesis.arcs + (arc,)
                ))
        active = sorted(next_active, key=lambda h: h.cost)[:beam_size]
        if not active:
            break
    return active

def finalize_hypotheses(fst, hypotheses, nbest=10):
    completed = [
        Hypothesis(h.cost + fst.finals[h.state], h.state, h.words, h.arcs)
        for h in hypotheses if h.state in fst.finals
    ]
    return sorted(completed, key=lambda h: h.cost)[:nbest]

def decode_nbest(fst, phones, beam_size=100, nbest=10):
    active = advance_hypotheses(fst, start_hypotheses(fst), phones, beam_size)
    return finalize_hypotheses(fst, active, nbest)

good_night_phones = LEXICON["GOOD"] + LEXICON["NIGHT"]
medieval_knight_phones = LEXICON["MEDIEVAL"] + LEXICON["KNIGHT"]

for name, phones in [("GOOD + 同音词", good_night_phones), ("MEDIEVAL + 同音词", medieval_knight_phones)]:
    print()
    print(name, phones)
    for hypothesis in decode_nbest(LG, phones):
        print(" ".join(hypothesis.words), f"cost={hypothesis.cost:.4f}", f"score≈{math.exp(-hypothesis.cost):.4f}")
"""
    ),
    md(
        """
### 怎样理解上面的分数

词典路径权重设为 0，所以总代价来自 G。`GOOD NIGHT` 的近似路径概率是 `0.55×0.90`；`GOOD KNIGHT` 是 `0.55×0.10`。对 `MEDIEVAL` 上下文，排序反过来。

这里不是说语言模型永远正确，而是说明同音拼写必须借助上下文。若领域发生变化，G 的统计和热词策略也必须更新。
"""
    ),
    md(
        """
## 6. LM scale、插词项与热词：都是代价，不是字符串替换

常见融合形式可写成：

`total_cost = acoustic_cost + alpha × lm_cost + beta × word_count - hotword_bonus`

代价越低越好。不同代码库的符号约定可能相反，调参前必须确认使用的是 score 还是 cost。
"""
    ),
    code(
        """
CANDIDATES = [
    {"text": "GOOD NIGHT", "acoustic": 2.40, "lm": neglog(0.55 * 0.90), "words": 2, "hot": False},
    {"text": "GOOD KNIGHT", "acoustic": 1.80, "lm": neglog(0.55 * 0.10), "words": 2, "hot": True},
    {"text": "GOODNIGHT", "acoustic": 2.20, "lm": neglog(0.05), "words": 1, "hot": False},
]

alpha_widget = widgets.FloatSlider(min=0, max=2.5, step=0.1, value=0.8, description="LM scale", continuous_update=False)
beta_widget = widgets.FloatSlider(min=-1, max=1, step=0.1, value=0.0, description="插词项", continuous_update=False)
boost_widget = widgets.FloatSlider(min=0, max=3, step=0.1, value=0.0, description="KNIGHT热词", continuous_update=False)
fusion_output = widgets.Output()

def fusion_cost(candidate, alpha, beta, boost):
    return candidate["acoustic"] + alpha * candidate["lm"] + beta * candidate["words"] - (boost if candidate["hot"] else 0.0)

def show_fusion(*_):
    costs = [fusion_cost(c, alpha_widget.value, beta_widget.value, boost_widget.value) for c in CANDIDATES]
    best = int(np.argmin(costs))
    with fusion_output:
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar([c["text"] for c in CANDIDATES], costs)
        bars[best].set_hatch("//")
        ax.set(ylabel="总代价（越低越好）", title="声学、LM、插词项与热词的竞争")
        for bar, value in zip(bars, costs):
            ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.2f}", ha="center", va="bottom")
        plt.show()
        print("当前最佳：", CANDIDATES[best]["text"])
        print("热词 bonus 过大时，即使声学与上下文都不支持，也会制造误触发。")

for control in [alpha_widget, beta_widget, boost_widget]:
    control.observe(show_fusion, names="value")
display(widgets.VBox([alpha_widget, beta_widget, boost_widget]), fusion_output)
show_fusion()
"""
    ),
    code(
        """
def decode_with_graph_settings(phones, lm_scale=1.0, knight_boost=0.0):
    graph = compose_left_output_epsilon(L, G, lm_scale=lm_scale, word_boosts={"KNIGHT": knight_boost})
    return decode_nbest(graph, phones, nbest=5)

for boost in [0.0, 0.5, 1.5, 3.0]:
    result = decode_with_graph_settings(good_night_phones, lm_scale=1.0, knight_boost=boost)
    print(f"boost={boost:.1f} →", [(" ".join(h.words), round(h.cost, 3)) for h in result])
"""
    ),
    md(
        """
## 7. N-best 与 lattice：不要只留下 1-best

N-best 是若干完整候选列表；lattice 是共享前缀/后缀的紧凑搜索图，能保留更多替代路径和时间信息。N-best 可以从 lattice 中提取，但 lattice 不只是一个 Python list。

保留替代假设有利于二遍 LM、热词重打分、置信度和语义重排。剪枝过早会使后处理无候选可救。
"""
    ),
    code(
        """
def common_word_prefix(word_sequences):
    if not word_sequences:
        return ()
    result = []
    for items in zip(*word_sequences):
        if len(set(items)) != 1:
            break
        result.append(items[0])
    return tuple(result)

active = start_hypotheses(LG)
print("逐 phone 推进 token passing：")
for index, phone in enumerate(good_night_phones, start=1):
    active = advance_hypotheses(LG, active, [phone], beam_size=20)
    top = sorted(active, key=lambda h: h.cost)[:5]
    stable = common_word_prefix([h.words for h in top])
    print(f"t={index:2d} phone={phone:>2s} active={len(active):2d} stable={' '.join(stable)!r} top={[(' '.join(h.words), round(h.cost,2)) for h in top[:3]]}")

completed = finalize_hypotheses(LG, active, nbest=5)
print("最终 N-best：", [(" ".join(h.words), round(h.cost, 4)) for h in completed])
"""
    ),
    md(
        """
## 8. 流式跨 chunk：保存 active tokens，不是只保存最佳字符串

如果在 chunk 边界重置图状态，第二个 chunk 的 phone 会被当成新句开头；若只保留当前 1-best，也可能剪掉之后会翻盘的路径。正确状态至少包含多个 active graph states、累计代价、输出历史和 backpointer。
"""
    ),
    code(
        """
offline_active = advance_hypotheses(LG, start_hypotheses(LG), good_night_phones, beam_size=100)
offline_nbest = finalize_hypotheses(LG, offline_active, nbest=10)

split = 4
chunk1 = good_night_phones[:split]
chunk2 = good_night_phones[split:]
stream_active = advance_hypotheses(LG, start_hypotheses(LG), chunk1, beam_size=100)
stream_active = advance_hypotheses(LG, stream_active, chunk2, beam_size=100)
stream_nbest = finalize_hypotheses(LG, stream_active, nbest=10)

offline_signature = [(h.words, h.cost) for h in offline_nbest]
stream_signature = [(h.words, h.cost) for h in stream_nbest]
assert offline_signature == stream_signature
print("离线与跨 chunk 保存 token state 的 N-best 完全一致")

reset_active = advance_hypotheses(LG, start_hypotheses(LG), chunk2, beam_size=100)
reset_nbest = finalize_hypotheses(LG, reset_active, nbest=10)
print("错误地在 chunk2 重置后：", [(h.words, h.cost) for h in reset_nbest])
"""
    ),
    md(
        """
## 9. 最常见的 WFST 工程错误

1. **符号表 id 不一致**：L 输出 `NIGHT=42`，G 却把 42 当成别的词；图可能能组合但语义错误。
2. **score/cost 方向相反**：把奖励加到负对数代价上，反而惩罚热词。
3. **epsilon 处理错误**：漏路径、重复路径或 composition 爆炸。
4. **LM scale 重复应用**：建图时缩放一次，解码时又缩放。
5. **过度剪枝**：正确路径早期声学分低，被永久删掉。
6. **图版本不一致**：词表、L、G、token topology 或模型输出 id 不匹配。
7. **负环**：动态 boost 产生可重复获得奖励的环，最短路径失去定义。
8. **流式 reset 错误**：chunk 间丢失 active tokens、LM state 或 backpointer。
"""
    ),
    code(
        """
# 大小写不一致的 G：接口报告会直接暴露问题。
bad_G = WFST(start=0)
bad_G.add_arc(0, 1, "good", "good", neglog(0.5))
bad_G.set_final(1)
bad_report = interface_report(L, bad_G)
print(bad_report)
assert "GOOD" in bad_report["missing_in_right"]

def assert_graph_sane(fst):
    assert fst.start in fst.states
    for arc in fst.arcs:
        assert np.isfinite(arc.weight), arc
        assert arc.ilabel != "", arc
        assert arc.olabel != "", arc
    assert fst.finals, "图没有 final state"
    return True

assert assert_graph_sane(L)
assert assert_graph_sane(G)
assert assert_graph_sane(LG)
print("结构检查通过；生产图还需可达性、coaccessibility、epsilon cycle 和符号表 checksum 检查。")
"""
    ),
    md(
        """
## 10. Determinize、minimize、epsilon removal 与 pruning 在做什么

- **determinize**：让同一状态对同一输入尽量只有一条确定转移，同时保持加权关系；
- **minimize**：合并等价状态，减小图；
- **epsilon removal**：消除可安全消除的 epsilon 转移并重新分配权重；
- **pruning**：删除相对最佳路径差太多的候选，换取速度和内存。

这些操作不是“让图看起来整齐”。必须在正确 semiring 和前置条件下保持语言/权重等价。大型图应交给 OpenFst、Kaldi、k2 等成熟实现，并做随机路径等价测试。
"""
    ),
    md(
        """
## 11. 从本实验走向 CTC 解码图

本实验直接把 phone 序列送入 `L∘G`。真实 CTC 系统前面还需要一个 token topology，把逐帧 blank/repeat 路径映射到 token 序列，再与 lexicon 和 LM 连接：

```text
frame-level CTC labels → CTC topology T → token/phone → L → word → G
```

若使用字符或词片直接作为语言模型单元，L 可能很薄甚至省略；若使用 phone、词典和上下文依赖，图会更接近传统链路。关键不是背图名字，而是逐层检查输入符号、输出符号、权重语义和状态生命周期。
"""
    ),
    md(
        """
## 12. 最终闭卷测试（40 分）

每题 0～2 分。达到 **32/40**，且代码题实际运行，才算通过。

### 概念与手算

1. 用一句话分别定义 FSA、FST、L、G。
2. 概率 0.8 和 0.2 的弧代价分别是多少？哪条更好？
3. 两条路径概率 0.3 与 0.2：Viterbi 概率和 forward 总概率分别是多少？
4. composition 为什么要求左图输出标签与右图输入标签匹配？
5. L 中 phone 弧输出 epsilon 的作用是什么？
6. `GOOD NIGHT` 与 `GOOD KNIGHT` phone 相同时，G 如何改变排序？
7. tropical 与 log semiring 在合并多条路径时有什么区别？
8. N-best 与 lattice 有什么区别？

### 编程与排错

9. 从空白实现 Arc/WFST，并验证起点、终点和弧权重。
10. 为五个词构造 L，其中至少包含一组同音词。
11. 从语料计数构造一个带平滑的 Bigram G。
12. 实现本专题的 epsilon-output composition，并与手算路径核对。
13. 实现 token passing 与 N-best，证明代价最低路径正确。
14. 故意打乱符号 id，设计 checksum 或符号表检查阻止加载。
15. 调节 LM scale，找出两个候选发生翻转的精确阈值。
16. 加热词奖励，画出召回提升与误触发的权衡。
17. 随机切分输入，证明跨 chunk 保存 active tokens 与离线 N-best 一致。
18. 只保存 1-best，构造后续无法翻盘的反例。

### 系统与表达

19. 画传统 HCLG 和 CTC TLG 两条链路，标出每层输入/输出符号。
20. 用 5 分钟向同事解释“WFST 不是语言模型本身”，必须包含 L、G、composition、shortest path 和 streaming state。
"""
    ),
    md(
        """
<details><summary>展开关键答案与评分锚点</summary>

1. FSA 接受序列；FST 映射序列；L 映射发音到词；G 编码词序列语言约束。
2. 代价约 0.223 与 1.609，0.8 对应的代价更低。
3. Viterbi 取 0.3，forward 总概率为 0.5。
4. 中间标签代表同一种符号，才能把两段路径连接。
5. 一个词的内部 phone 不应重复输出 word，通常只在词边界输出。
6. 通过上下文状态上的 Bigram 代价让合适拼写路径更短。
7. tropical 在候选路径间取 min，log semiring 聚合路径总概率。
8. N-best 是若干完整候选；lattice 是共享结构的紧凑搜索图。

代码题必须有手算小图、自动断言、错误输入和跨 chunk 测试。只会调用现成库而不能解释弧与状态，最多得一半分。

</details>
"""
    ),
    md(
        """
## 13. 离场票

- [ ] 我能从概率手算负对数路径代价。
- [ ] 我能画出一个小 L、一个小 G，并解释每条弧的输入/输出。
- [ ] 我能实现并调试 `L∘G`。
- [ ] 我能从 lattice/N-best 解释 1-best 为什么可能不可靠。
- [ ] 我能解释 LM scale、插词项、热词奖励的方向和风险。
- [ ] 我能跨 chunk 保存 active graph tokens，并证明结果与离线一致。
- [ ] 我能说明传统 HCLG 与 CTC 解码图不是同一个固定模板。

达到这些证据后，再进入量化与部署；否则模型部署成功也无法判断解码图是否在悄悄损害结果。
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "course": {"kind": "wfst-intensive-lab", "version": 1, "related_lessons": [19, 20, 21, 22, 23]},
    },
)

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUT)
print(f"wrote {OUT}")
