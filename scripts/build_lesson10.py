from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "10_从未知对齐到CTC_兼顾流式.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (learn-asr)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}


def md(text: str):
    nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str):
    nb.cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# 第 10 课：从“未知对齐”走到 CTC，并埋下流式 ASR 的主线

这一课先不急着调用 `torch.nn.CTCLoss`。我们要亲眼看到 CTC 为什么必须存在。

学完以后，你应该能够回答：

1. 为什么音频帧不能直接和文字一一对应？
2. CTC 的 `blank` 到底做了什么？
3. 一条 frame-level path 怎样折叠成最终文字？
4. 为什么目标 `11` 比目标 `12` 需要更多时间步？
5. 为什么“用了 CTC”不等于“模型天然支持流式”？

> 本课原则：先看图、再动手、最后才总结规则。代码里的小矩阵都是故意做得很小，让我们能看懂每一个格子。
""")

code(r"""
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from IPython.display import Audio, display
from ipywidgets import interact, IntSlider, Dropdown

plt.rcParams["figure.figsize"] = (11, 4)
plt.rcParams["axes.grid"] = False

def find_root():
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("找不到项目根目录，请从 learn_asr 目录或 notebooks 目录启动 Jupyter。")

ROOT = find_root()
print("项目根目录:", ROOT)
""")

md(r"""
## 1. 先看真实语音：几十帧只对应一个标签

下面使用开源 Free Spoken Digit Dataset 中的一条真实录音。文件名里的 `1` 是说话内容，而模型看到的是一串采样点。
""")

code(r"""
audio_path = ROOT / "data" / "spoken_digits_parts" / "1_jackson_0.wav"
y, sr = sf.read(audio_path)
if y.ndim > 1:
    y = y.mean(axis=1)

display(Audio(y, rate=sr))
print(f"采样率: {sr} Hz | 时长: {len(y)/sr:.3f} 秒 | 采样点: {len(y)} | 标签: '1'")
""")

code(r"""
hop_length = max(1, round(sr * 0.010))       # 10 ms
win_length = max(2, round(sr * 0.025))       # 25 ms
n_fft = 1 << (win_length - 1).bit_length()

mel = librosa.feature.melspectrogram(
    y=y.astype(np.float32), sr=sr, n_fft=n_fft,
    hop_length=hop_length, win_length=win_length,
    n_mels=40, fmin=20, fmax=sr/2, power=2.0,
)
logmel = librosa.power_to_db(mel, ref=np.max)
frame_times = librosa.frames_to_time(np.arange(logmel.shape[1]), sr=sr, hop_length=hop_length)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
times = np.arange(len(y)) / sr
axes[0].plot(times, y, linewidth=0.8)
axes[0].set(title="Real waveform: spoken digit '1'", xlabel="Time (s)", ylabel="Amplitude")
img = librosa.display.specshow(logmel, x_axis="time", y_axis="mel", sr=sr,
                               hop_length=hop_length, ax=axes[1], cmap="magma")
axes[1].set(title=f"Log-Mel: {logmel.shape[1]} frames but only one target token '1'")
fig.colorbar(img, ax=axes[1], label="dB relative to max")
plt.show()

print("Log-Mel shape [mel_bins, frames] =", logmel.shape)
print(f"大约每 10 ms 前进一步，所以模型要处理 {logmel.shape[1]} 个时间位置。")
""")

md(r"""
### 观察题 1

标签只有一个字符 `1`，但上图有许多帧。请先想一想：

- `1` 应该贴在哪一帧？
- 发音开始前和结束后的帧贴什么？
- `1` 的发音持续很多帧，这些帧全贴 `1` 吗？

<details><summary>展开参考思路</summary>

我们没有人工提供的逐帧边界，因此并不知道字符应该贴在哪一帧。静音和持续发音也让“一帧一个字符”的规则失效。CTC 的目标就是：**只给最终文字，不要求人工给出精确对齐。**

</details>
""")

md(r"""
## 2. 把困难缩小：8 帧语音，要识别成 `12`

假设编码器产生 8 个时间步，每一步都要在 `{blank, 1, 2}` 中选择一个符号。

我们用 `∅` 表示 CTC blank。它不是空格，也不是静音标签，而是“这个时间步不输出普通 token”的特殊类别。
""")

code(r"""
def draw_alignment(path, title="A frame-level path"):
    fig, ax = plt.subplots(figsize=(11, 2.5))
    symbols = list(path)
    colors = ["C0" if s == "1" else "C1" if s == "2" else "0.82" for s in symbols]
    ax.bar(np.arange(len(symbols)), np.ones(len(symbols)), color=colors, width=0.86)
    for i, symbol in enumerate(symbols):
        ax.text(i, 0.5, symbol, ha="center", va="center", fontsize=16)
    ax.set(xticks=np.arange(len(symbols)), xticklabels=np.arange(1, len(symbols)+1),
           xlabel="Encoder time step", yticks=[], ylim=(0, 1), title=title)
    plt.show()

draw_alignment(["∅", "1", "1", "∅", "∅", "2", "2", "∅"])
""")

md(r"""
这条路径的 8 个符号并不是最终识别结果。CTC 定义了一个折叠函数 $B$：

1. **先合并连续重复**：`∅ 1 1 ∅ ∅ 2 2 ∅` → `∅ 1 ∅ 2 ∅`
2. **再删除 blank**：`∅ 1 ∅ 2 ∅` → `12`

顺序非常重要。先删 blank 会在重复字符处得到错误答案。
""")

code(r"""
BLANK = "∅"

def ctc_collapse(path, blank=BLANK):
    merged = []
    previous = None
    for symbol in path:
        if symbol != previous:
            merged.append(symbol)
        previous = symbol
    return "".join(symbol for symbol in merged if symbol != blank)

examples = [
    ["∅", "1", "1", "∅", "2", "2", "∅"],
    ["1", "∅", "1"],
    ["1", "1"],
    ["∅", "1", "∅", "∅", "2"],
]
for path in examples:
    print(" ".join(path), " -> ", repr(ctc_collapse(path)))
""")

md(r"""
## 3. 交互实验：亲手编辑一条 CTC 路径

修改 6 个时间步的预测。每改一个格子，下方都会重新显示折叠结果。

请依次尝试构造：`12`、`11`、空字符串，以及一个会被错误合并成 `1` 的路径。
""")

code(r"""
choices = [BLANK, "1", "2"]
controls = {
    f"t{i+1}": Dropdown(options=choices, value=v, description=f"t{i+1}")
    for i, v in enumerate([BLANK, "1", "1", BLANK, "2", BLANK])
}

@interact(**controls)
def explore_path(**kwargs):
    path = [kwargs[f"t{i+1}"] for i in range(6)]
    draw_alignment(path, title=f"B(path) = {ctc_collapse(path)!r}")
    print("路径:", " ".join(path))
    print("折叠结果:", repr(ctc_collapse(path)))
""")

md(r"""
### 练习 2：不运行代码，先判断

写出下列路径的折叠结果：

1. `∅ A A ∅ B B ∅`
2. `A ∅ A`
3. `A A ∅`
4. `∅ ∅ ∅`
5. `A ∅ ∅ A`

<details><summary>查看答案</summary>

1. `AB`
2. `AA`
3. `A`
4. 空字符串
5. `AA`。两个 `A` 中间只要有至少一个 blank，就不会作为连续重复被合并。

</details>
""")

md(r"""
## 4. 同一个文字可以由很多条路径产生

目标文本是 `12`，以下路径都合法：

- `1 2`
- `∅ 1 2`
- `1 1 2`
- `1 ∅ 2`
- `∅ 1 1 ∅ 2`

CTC 不强迫模型选中某一条“标准对齐路径”。它把所有能折叠为目标文本的路径概率加起来。
""")

code(r"""
def all_paths(alphabet, time_steps):
    return list(product(alphabet, repeat=time_steps))

alphabet = [BLANK, "1", "2"]
for T in range(2, 7):
    paths = all_paths(alphabet, T)
    valid = [p for p in paths if ctc_collapse(p) == "12"]
    print(f"T={T}: 全部路径 {len(paths):4d} 条，其中折叠为 '12' 的有 {len(valid):3d} 条")
""")

code(r"""
@interact(time_steps=IntSlider(min=2, max=7, step=1, value=4, description="T"))
def show_valid_paths(time_steps=4):
    valid = [p for p in all_paths(alphabet, time_steps) if ctc_collapse(p) == "12"]
    print(f"T={time_steps}，共有 {len(valid)} 条合法路径。显示前 30 条：")
    for path in valid[:30]:
        print(" ".join(path), " -> 12")
""")

md(r"""
### 关键直觉

训练时，我们不知道正确边界究竟在哪里，所以不应该只奖励一条路径。CTC 要最大化的是：

$$P(\text{目标文字}\mid X)=\sum_{\pi:B(\pi)=\text{目标文字}}P(\pi\mid X)$$

- $X$：输入音频特征
- $\pi$：一条逐时间步路径
- $B(\pi)$：折叠后的文字

现在不用害怕求和符号。它只是在说：**把所有“最终答案正确”的对齐方式都算作成功。**
""")

md(r"""
## 5. 用一张概率热力图观察 CTC 输出

假设模型在 6 个时间步输出 `{blank, 1, 2}` 的概率。每一列加起来都是 1。
""")

code(r"""
probs = np.array([
    [0.70, 0.15, 0.10, 0.65, 0.15, 0.75],  # blank
    [0.25, 0.75, 0.80, 0.25, 0.05, 0.10],  # 1
    [0.05, 0.10, 0.10, 0.10, 0.80, 0.15],  # 2
])
symbols = [BLANK, "1", "2"]

fig, ax = plt.subplots(figsize=(10, 3.6))
im = ax.imshow(probs, aspect="auto", cmap="viridis", vmin=0, vmax=1)
for row in range(probs.shape[0]):
    for col in range(probs.shape[1]):
        ax.text(col, row, f"{probs[row, col]:.2f}", ha="center", va="center",
                color="white" if probs[row, col] < 0.45 else "black")
ax.set(xticks=np.arange(6), xticklabels=[f"t{i+1}" for i in range(6)],
       yticks=np.arange(3), yticklabels=symbols,
       xlabel="Encoder time step", ylabel="CTC class", title="Per-frame CTC probabilities")
fig.colorbar(im, ax=ax, label="Probability")
plt.show()

greedy_ids = probs.argmax(axis=0)
greedy_path = [symbols[i] for i in greedy_ids]
print("逐帧最大概率路径:", " ".join(greedy_path))
print("Greedy CTC 结果:", ctc_collapse(greedy_path))
""")

md(r"""
## 6. Greedy 最好路径，不一定等于概率最大的文字

Greedy decoding 每个时间步只取最大概率类别，它找的是一条很强的路径。

但 CTC 文本概率会把多条路径相加。很多条“各自不是第一名”的路径，相加后可能超过 greedy 路径所属的文本。这正是后面学习 Beam Search 的原因。
""")

code(r"""
def path_probability(path, probabilities, labels):
    index = {label: i for i, label in enumerate(labels)}
    return float(np.prod([probabilities[index[s], t] for t, s in enumerate(path)]))

text_probabilities = {}
ranked_paths = []
for path in all_paths(symbols, probs.shape[1]):
    p = path_probability(path, probs, symbols)
    text = ctc_collapse(path)
    text_probabilities[text] = text_probabilities.get(text, 0.0) + p
    ranked_paths.append((p, path, text))

print("概率最高的 8 条单独路径：")
for p, path, text in sorted(ranked_paths, reverse=True)[:8]:
    print(f"P={p:.6f} | {' '.join(path)} -> {text!r}")

print("\n把同一文本的路径求和后，概率最高的 8 个文本：")
for text, p in sorted(text_probabilities.items(), key=lambda x: x[1], reverse=True)[:8]:
    print(f"P={p:.6f} | text={text!r}")
""")

md(r"""
### 思考题 3

如果某条路径的每一步概率都比较高，它就一定对应概率最高的文本吗？

<details><summary>查看答案</summary>

不一定。一条路径的概率是各时间步概率的乘积；而一个文本的 CTC 概率是所有合法路径概率之和。路径最多和文本概率最大是两个不同问题。

</details>
""")

md(r"""
## 7. 最容易出错的地方：连续重复字符

目标是 `11` 时，路径 `1 1` 会先合并为一个 `1`，因此它并不能表示 `11`。

要保留两个相邻且相同的字符，中间必须有 blank：

`1 ∅ 1` → `11`
""")

code(r"""
targets = ["1", "12", "11", "112", "121", "111"]

def minimum_ctc_steps(target):
    adjacent_repeats = sum(a == b for a, b in zip(target, target[1:]))
    return len(target) + adjacent_repeats

for target in targets:
    repeats = sum(a == b for a, b in zip(target, target[1:]))
    print(f"target={target!r}: token 数={len(target)}, 相邻重复={repeats}, 最少时间步={minimum_ctc_steps(target)}")
""")

md(r"""
CTC 的最低时间长度不是永远等于标签长度：

$$T_{min}=U+\text{相邻重复 token 的数量}$$

其中 $T$ 是编码器输出时间步数，$U$ 是标签 token 数。这会直接影响卷积下采样：如果把时间轴压得太短，某些样本将不存在合法 CTC 路径，loss 可能变成无穷大。

### 练习 4

不运行代码，计算以下目标至少需要多少个编码器时间步：

1. `BOOK`
2. `CAT`
3. `COFFEE`
4. `1001`

<details><summary>查看答案</summary>

1. `BOOK`：5（`OO` 之间需要一个 blank）
2. `CAT`：3
3. `COFFEE`：8（`FF` 和 `EE` 各多需要一步）
4. `1001`：5（`00` 多需要一步）

</details>
""")

md(r"""
## 8. 流式 ASR：CTC 为什么很合适，但又为什么不够

CTC head 会在每个编码器时间步输出类别概率，因此很适合边收到音频边解码。但是系统能否流式工作，取决于整条链路：

```text
麦克风音频块
   ↓
在线特征提取（保留上一块的尾部）
   ↓
因果或分块编码器（不能无限偷看未来）
   ↓
逐帧 CTC 概率
   ↓
增量解码与结果稳定策略
```

如果编码器使用双向 LSTM，或者普通 Self-Attention 可以看到整段未来，即使最后一层使用 CTC，也仍然不是严格流式模型。
""")

code(r"""
stream_probs = np.array([
    [0.75, 0.20, 0.10, 0.65, 0.15, 0.70, 0.15, 0.70],
    [0.20, 0.70, 0.80, 0.25, 0.10, 0.10, 0.05, 0.10],
    [0.05, 0.10, 0.10, 0.10, 0.75, 0.20, 0.80, 0.20],
])

@interact(received=IntSlider(min=1, max=8, value=1, step=1, description="收到帧数"))
def stream_greedy_demo(received=1):
    partial = stream_probs[:, :received]
    path = [symbols[i] for i in partial.argmax(axis=0)]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(partial, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set(xticks=np.arange(received), xticklabels=[f"t{i+1}" for i in range(received)],
           yticks=np.arange(3), yticklabels=symbols,
           xlabel="Frames received so far", ylabel="CTC class",
           title=f"Partial greedy result: {ctc_collapse(path)!r}")
    fig.colorbar(im, ax=ax, label="Probability")
    plt.show()
    print("当前路径:", " ".join(path))
    print("当前临时文本:", repr(ctc_collapse(path)))
""")

md(r"""
### 流式学习主线（后续会逐项实现）

我们之后不会只做“把整段 wav 一次塞进模型”的离线实验，而会额外研究：

1. **Chunk（音频块）**：一次送入多少毫秒？
2. **左上下文缓存**：怎样复用过去，不重复计算？
3. **右上下文**：允许看一点未来能提高准确率，但会增加延迟。
4. **实时率 RTF**：处理 1 秒音频是否少于 1 秒？
5. **首字延迟与最终延迟**：何时第一次显示字，何时确认不再修改？
6. **Partial / Stable transcript**：临时结果为什么会抖动，怎样提交稳定前缀？
7. **流式 CTC Prefix Beam Search**：怎样跨 chunk 保存解码状态？

这里先记住一句话：

> CTC 提供了逐帧输出接口；因果/分块编码器和有状态解码器，才把它变成完整流式 ASR。
""")

md(r"""
## 9. 本课综合测试

请先把答案写在新的 Markdown cell 中，再展开答案。

### 基础题

1. 一条语音有 80 帧，标签有 5 个 token。为什么不能直接把第 1～5 帧分别贴上标签？
2. CTC blank 是不是“静音”？
3. 路径 `A A ∅ B B` 折叠成什么？
4. 路径 `A ∅ A` 折叠成什么？
5. CTC 为什么需要把多条路径的概率相加？

### 进阶题

6. 目标 `LETTER` 至少需要多少时间步？
7. 编码器输出只有 4 步，目标是 `BOOK`，CTC 是否存在合法路径？为什么？
8. Greedy path 概率最高，为什么对应文本仍可能不是概率最高文本？
9. 模型使用双向 LSTM + CTC，是否属于严格流式？
10. 若 chunk 为 320 ms，同时需要 160 ms 右上下文，仅这两项会带来怎样的等待直觉？

### 编程题

11. 修改 `ctc_collapse`，让它返回“合并连续重复后的序列”和“删除 blank 后的序列”两个结果。
12. 枚举长度为 5 的所有 `{blank, A, B}` 路径，统计能折叠成 `AB` 的路径数量。
13. 写函数检查一个 target 在给定 `T` 下是否可能存在合法路径。
""")

md(r"""
<details><summary>展开综合测试参考答案</summary>

1. 没有逐帧边界，而且发音、静音会占用很多帧；标签数与帧数也不匹配。
2. 不是。blank 是 CTC 的特殊“不输出普通 token”类别；模型可能在静音处输出 blank，也可能在字符内部输出 blank。
3. `AB`
4. `AA`
5. 因为训练数据只给最终文字，没有指定唯一对齐；所有能得到目标文字的对齐都应该贡献概率。
6. `LETTER` 有 `TT` 一个相邻重复，最少 $6+1=7$ 步。
7. 不存在。`BOOK` 至少需要 5 步，因为 `OO` 中间需要 blank。
8. 文本概率是属于该文本的所有路径概率之和，而 greedy 只选择一条路径。
9. 不是。双向 LSTM 的当前输出依赖未来帧。
10. 系统通常要先收集当前块，还要等待右上下文；这会贡献大约 480 ms 的算法等待直觉，但实际首字/最终延迟还包括特征、计算、解码和稳定策略，不能简单宣称总延迟恰好为 480 ms。
11～13 见下一代码单元的参考实现，建议自己先写。

</details>
""")

code(r"""
# 编程题参考实现：先折叠这个 cell，自己完成后再对照。

def ctc_collapse_two_steps(path, blank=BLANK):
    merged = []
    previous = None
    for symbol in path:
        if symbol != previous:
            merged.append(symbol)
        previous = symbol
    without_blank = [s for s in merged if s != blank]
    return merged, without_blank

valid_ab = [p for p in all_paths([BLANK, "A", "B"], 5) if ctc_collapse(p) == "AB"]

def ctc_target_is_possible(target, time_steps):
    return time_steps >= minimum_ctc_steps(target)

print("长度为 5、可折叠成 AB 的路径数:", len(valid_ab))
for target, T in [("BOOK", 4), ("BOOK", 5), ("CAT", 3)]:
    print(target, T, ctc_target_is_possible(target, T))
""")

md(r"""
## 10. 本课小结与下一课

今天最重要的不是背公式，而是建立四个直觉：

1. **未知对齐**：我们只有整句标签，没有逐帧标签。
2. **CTC path**：模型每个时间步输出一个类别。
3. **折叠规则**：先合并连续重复，再删除 blank。
4. **对齐求和**：同一文本对应许多路径，训练时要把它们的概率加起来。

流式方面也要牢牢记住：**CTC head 适合在线输出，但编码器、特征提取和解码器也必须按流式方式设计。**

下一课将进入 CTC 最核心也最容易卡住的部分：

> **第 11 课：不用枚举指数级路径——用动态规划手算 CTC 前向概率**

我们会从一个只有 3 个时间步的小矩阵开始，先用枚举验证，再画成状态图，最后一步步推导 forward algorithm 和 log-space 计算。
""")

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
