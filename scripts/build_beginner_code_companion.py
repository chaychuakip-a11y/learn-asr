from __future__ import annotations

import copy
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import nbformat as nbf
from nbclient import NotebookClient

from notebook_layout import (
    ensure_executed_directories,
    executed_path,
    sanitize_notebook_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
STEM = "代码伴读_零基础逐行理解ASR"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def lesson(
    number: int,
    title: str,
    goal: str,
    before: str,
    source: str,
    after: str,
    mistakes: str,
    experiment: str,
):
    return [
        md(
            f"""
## {number}. {title}

**这一段只学一件事：**{goal}

### 运行前，先这样读

{before}
"""
        ),
        code(source),
        md(
            f"""
### 运行后，逐项核对

{after}

### 常见错误

{mistakes}

### 只改一处的小实验（现在可以先跳过）

{experiment}
"""
        ),
    ]


cells = [
    md(
        """
# 代码伴读：零基础逐行理解 ASR

这不是考试，也不是另一套需要背诵的课程。它是整套 ASR 课程的“代码翻译器”：当你看到一段代码，不知道变量、括号、shape、类、训练循环或 CTC 参数在做什么，就回到这里找相同结构。

本 Notebook 特意遵守四条规则：

1. 一个代码单元只引入少量新概念；
2. 关键行都有中文注释；
3. 运行后打印值、类型和 shape，不让数据偷偷流过；
4. 每节都告诉你常见错误，以及只改一处就能观察到的变化。

你现在可以只按顺序运行，不回答任何题。带 `_已运行` 的同名 Notebook 保存了完整输出，卡住时可以逐单元对照。
"""
    ),
    md(
        """
## 使用方法：一个代码单元应该怎样读

看到代码时依次问五个问题：

1. **输入是谁？** 变量从哪里来，类型是什么？
2. **形状是什么？** 对语音代码尤其要写出每个轴的含义。
3. **这一行改变了什么？** 是创建新值、原地修改、还是只打印？
4. **输出给谁？** 下一层期望什么 dtype、shape 和单位？
5. **怎样证明没理解错？** 打印、断言、画图，或者故意给错误输入。

ASR 中最常见的 shape 记号：

| 记号 | 含义 | 例子 |
|---|---|---|
| `B` | 一个 batch 中有几条语音 | 4 条语音 |
| `S` | 原始采样点数 | 16 kHz 下 1 秒是 16000 |
| `T` | 特征或编码器的时间帧数 | 100 帧约 1 秒（10 ms 帧移） |
| `F` | 每帧特征维度 | 80 维 Log-Mel |
| `C` | 类别数，包含 CTC blank | 词表大小 + 1 |

不要只说“这是三维张量”，要说“这是 `[B,T,F]`，即批次、时间、特征”。
"""
    ),
]

cells += lesson(
    1,
    "Notebook、变量、类型与 print",
    "知道一行代码从右向左执行，并能检查变量的值与类型。",
    """
- `=` 不是数学等号，而是把右侧结果绑定给左侧名字。
- `16_000` 与 `16000` 数值相同，下划线只是方便阅读。
- `f\"...{name}...\"` 会把大括号里的变量放进字符串。
""",
    """
# 采样率：每秒包含多少个采样点。这里是整数 int。
sample_rate = 16_000

# 时长：2.5 秒。带小数点的数字通常是 float。
duration_seconds = 2.5

# 先做乘法，再用 int(...) 转成整数，最后保存到 num_samples。
num_samples = int(sample_rate * duration_seconds)

# print 只是把信息显示出来，不会改变变量。
print("采样率 =", sample_rate, "类型 =", type(sample_rate))
print("时长 =", duration_seconds, "类型 =", type(duration_seconds))
print(f"采样点总数 = {num_samples}")

# assert 是可执行的检查：条件为 False 时立刻报错。
assert num_samples == 40_000
""",
    """
- `sample_rate` 应是 `16000`，类型是 `int`。
- `duration_seconds` 应是 `2.5`，类型是 `float`。
- `16000 × 2.5 = 40000`，所以断言通过且没有额外输出。
- “断言没有输出”通常代表检查通过，不代表代码没有运行。
""",
    """
- 把 `=` 误读成“左右永远相等”；实际上以后可以重新绑定变量。
- 忘记字符串需要引号，例如写 `unit = seconds` 会把 `seconds` 当变量名。
- 看到没有输出就反复运行；先看左侧执行编号是否出现。
""",
    "把 `duration_seconds` 改成 `1.0`，先预测断言为什么失败；再把断言中的期望值改成正确结果。",
)

cells += lesson(
    2,
    "列表、索引、循环和函数",
    "看懂数据如何进入函数、经过循环，并通过 return 离开函数。",
    """
- `def` 开始定义函数，缩进的行属于函数体。
- 参数是调用者传进来的值；`return` 才是函数的正式输出。
- 列表索引从 0 开始，`values[0]` 是第一个元素。
""",
    """
def samples_to_seconds(num_samples: int, sample_rate: int) -> float:
    # 先检查错误边界，避免在后面得到没有意义的结果。
    if num_samples < 0:
        raise ValueError("num_samples 不能为负数")
    if sample_rate <= 0:
        raise ValueError("sample_rate 必须大于 0")

    # return 把计算结果交还给调用函数的那一行。
    return num_samples / sample_rate


# 列表中有三段语音的采样点数。
sample_counts = [8_000, 16_000, 32_000]
durations = []

# 循环每次从列表取一个值，放入 count。
for count in sample_counts:
    seconds = samples_to_seconds(count, 16_000)
    durations.append(seconds)  # append 把结果添加到列表末尾。
    print(count, "个采样点 ->", seconds, "秒")

print("全部时长：", durations)
assert durations == [0.5, 1.0, 2.0]
""",
    """
- 函数定义本身不会立刻计算；运行到 `samples_to_seconds(...)` 才调用。
- 循环运行 3 次，所以打印 3 行换算结果。
- `append` 改变的是 `durations` 列表，最终得到 `[0.5, 1.0, 2.0]`。
""",
    """
- 用 `print(result)` 代替 `return result`：屏幕上看似有结果，但调用者实际得到 `None`。
- 缩进不一致会产生 `IndentationError`，或者让某行意外跑到循环外。
- 写 `range(len(values))` 后混淆“索引”和“元素”；初学时优先直接遍历元素。
""",
    "调用 `samples_to_seconds(100, 0)`，观察 `ValueError` 最后一行；之后撤销这一改动，让 Notebook 可以继续运行。",
)

cells += lesson(
    3,
    "import、模块、对象和方法调用",
    "区分模块函数、对象方法和属性，读懂常见的点号语法。",
    """
- `import numpy as np` 给模块取简称，之后用 `np.函数名(...)`。
- `waveform.shape` 是读取属性；`waveform.mean()` 是调用方法，括号表示执行。
- 方法属于点号左边的对象，所以先确认左边是什么类型。
""",
    """
import numpy as np

# np.array 是模块中的函数；它返回 ndarray 对象。
waveform = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)

# shape、dtype 是属性，不加括号。
print("shape 属性：", waveform.shape)
print("dtype 属性：", waveform.dtype)

# mean、max 是 ndarray 对象的方法，要加括号才会执行。
print("均值方法：", waveform.mean())
print("最大绝对幅值：", np.abs(waveform).max())

assert waveform.shape == (4,)
assert waveform.dtype == np.float32
""",
    """
- `(4,)` 表示一维数组有 4 个元素；逗号是 Python 单元素 tuple 的写法。
- `np.abs(waveform)` 创建绝对值数组，随后 `.max()` 取最大值。
- `waveform.mean` 若不加括号，得到的是“方法对象”，不是均值数字。
""",
    """
- `ModuleNotFoundError`：环境中没有该包，先确认 Notebook 的 kernel 是否来自本项目。
- `AttributeError`：点号左边的对象没有这个属性，检查变量类型和拼写。
- 把 Python 列表当 NumPy 数组调用 `.shape`；列表没有 shape 属性。
""",
    "把 `print(waveform.mean)` 临时加到代码末尾，对比它与 `print(waveform.mean())` 的输出。",
)

cells += lesson(
    4,
    "数组的 shape、axis、索引与切片",
    "不靠猜测，逐轴说明 `[B,T,F]` 数据。",
    """
- 下面创建 `B=2, T=4, F=3` 的假特征。
- `features[b, t, f]` 依次选择批次、时间、特征。
- `:` 表示该轴全部保留；切片右端不包含在结果中。
""",
    """
import numpy as np

B, T, F = 2, 4, 3

# arange 先产生 0~23 共 24 个数；reshape 只重排形状，不改变元素总数。
features = np.arange(B * T * F, dtype=np.float32).reshape(B, T, F)

print("全部特征 [B,T,F]：", features.shape)
print("第 0 条语音 [T,F]：", features[0].shape)
print("第 0 条语音第 1 帧 [F]：", features[0, 1].shape)
print("所有语音的前两帧 [B,2,F]：", features[:, :2, :].shape)

# axis=1 沿时间轴求平均，时间轴消失，剩下 [B,F]。
time_average = features.mean(axis=1)
print("时间平均 [B,F]：", time_average.shape)

assert features.shape == (2, 4, 3)
assert time_average.shape == (2, 3)
""",
    """
- `features[0]` 消掉 B 轴；`features[:, :2, :]` 保留三个轴。
- `axis=1` 指 shape 中下标为 1 的 T 轴，不是“数值等于 1 的行”。
- 任何 `reshape` 前后元素总数必须一致：`2×4×3=24`。
""",
    """
- 交换 T/F 后代码仍可能运行，但语义已经错了；所以不能只看是否报错。
- 把 `axis` 与 shape 顺序背成固定含义；轴意义由当前张量 contract 决定。
- 把 `features[0, 1]` 误以为选择第 1～2 帧；它只选择单个位置。
""",
    "把 `axis=1` 改成 `axis=2`，运行前先写下输出 shape，并说出被平均掉的是哪个轴。",
)

cells += lesson(
    5,
    "广播、维度变换和可执行 shape 断言",
    "理解 PyTorch 如何自动扩展长度为 1 的轴，以及何时必须换轴。",
    """
- PyTorch 线性层常吃 `[B,T,F]`，`Conv1d` 常吃 `[B,C,T]`。
- `transpose(1, 2)` 交换第 1、2 轴；数据意义跟着位置移动。
- 广播从末尾向前比较，每一轴必须相等或其中一个是 1。
""",
    """
import torch

torch.set_num_threads(1)
B, T, F = 2, 5, 3
features = torch.arange(B * T * F, dtype=torch.float32).reshape(B, T, F)

# 每个特征维度一个缩放系数；shape [F] 会广播到 [B,T,F]。
feature_scale = torch.tensor([1.0, 0.1, 10.0])
scaled = features * feature_scale
print("广播后：", scaled.shape)

# [B,T,F] -> [B,F,T]，把特征轴放到 Conv1d 的 channel 位置。
for_conv = scaled.transpose(1, 2)
print("送入 Conv1d：", for_conv.shape)

# 再换回来，检查数值没有被改变。
restored = for_conv.transpose(1, 2)
print("换回 [B,T,F]：", restored.shape)

assert restored.shape == (B, T, F)
assert torch.equal(restored, scaled)
""",
    """
- `transpose` 改变轴顺序，但不会自动复制或丢掉数值。
- `torch.equal` 同时检查 shape 和所有元素完全相同。
- 变量名 `for_conv` 是提示，不会强迫 PyTorch 理解轴含义；断言才是契约。
""",
    """
- 把 `reshape(B,F,T)` 当成换轴：reshape 只是按存储顺序重新解释，通常会打乱语义。
- 广播意外发生，导致代码能跑但把参数加在错误轴上。
- 只写 `assert x.ndim == 3`，却不检查每个轴的具体意义。
""",
    "把 `feature_scale` 改成长度为 4 的 Tensor，观察广播错误；再解释为什么长度 3 可以工作。",
)

cells += lesson(
    6,
    "dtype、device 与 NumPy/Tensor 边界",
    "看懂浮点特征、整数标签和模型设备为什么必须匹配。",
    """
- 神经网络特征通常是 `float32`；类别 ID 通常是 `int64/long`。
- CPU Tensor 与 GPU Tensor 不能直接运算。
- `.to(...)` 返回转换后的 Tensor；要把返回值保存起来。
""",
    """
import numpy as np
import torch

numpy_waveform = np.array([0.0, 0.2, -0.2], dtype=np.float32)

# from_numpy 在 CPU 上通常与原数组共享内存；此处复制一份避免意外联动。
waveform = torch.from_numpy(numpy_waveform).clone()
token_ids = torch.tensor([1, 2], dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
waveform = waveform.to(device=device, dtype=torch.float32)
token_ids = token_ids.to(device=device, dtype=torch.long)

print("waveform:", waveform.dtype, waveform.device, waveform.shape)
print("token_ids:", token_ids.dtype, token_ids.device, token_ids.shape)

assert waveform.is_floating_point()
assert token_ids.dtype == torch.int64
assert waveform.device == token_ids.device
""",
    """
- 同一台机器上可能输出 `cpu` 或 `cuda:0`，两者都正确。
- `token_ids` 不是连续声音数值，而是离散类别编号，因此使用整数。
- `clone()` 在这里让 Tensor 拥有独立数据，教学时更容易推理副作用。
""",
    """
- `expected scalar type Long`：标签或索引 dtype 不对。
- `Expected all tensors to be on the same device`：模型和数据不在同一设备。
- 调用 `x.to(device)` 却没写回 `x = ...`，原变量仍在旧设备。
""",
    "打印 `torch.cuda.is_available()`；如果是 False，不代表课程失败，只代表本机当前用 CPU。",
)

cells += lesson(
    7,
    "nn.Module、__init__ 与 forward",
    "读懂模型类如何保存层，以及调用 model(x) 时发生了什么。",
    """
- `__init__` 在创建模型时运行一次，用于定义带参数的层。
- `forward` 在每次 `model(x)` 时运行，描述数据流。
- 不要直接调用 `model.forward(x)`；`model(x)` 会保留 PyTorch 的钩子机制。
""",
    """
import torch
from torch import nn


class TinyAcousticEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()  # 初始化父类 nn.Module 的内部状态。
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.ctc_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 输入 contract：[B,T,F]。
        assert features.ndim == 3
        hidden = self.input_layer(features)  # [B,T,F] -> [B,T,H]
        hidden = self.activation(hidden)     # shape 不变，只改变数值
        logits = self.ctc_head(hidden)       # [B,T,H] -> [B,T,C]
        return logits


B, T, F, H, C = 2, 6, 4, 8, 5
model = TinyAcousticEncoder(input_dim=F, hidden_dim=H, num_classes=C)
features = torch.randn(B, T, F)
logits = model(features)

print(model)
print("输入：", features.shape, "输出 logits：", logits.shape)
print("可训练参数量：", sum(p.numel() for p in model.parameters()))

assert logits.shape == (B, T, C)
""",
    """
- `self.xxx` 会把子层登记到模型中，所以 optimizer 能找到它们的参数。
- `Linear` 只变换最后一轴；前面的 B、T 轴被当成批量位置保留。
- `logits` 是未归一化分数，不是概率，也还不是最终文本。
""",
    """
- 忘记 `super().__init__()`，子模块登记可能失败。
- 在 `forward` 内临时创建 `nn.Linear`，每次调用都会得到新参数，无法正常训练。
- 把输出 C 误认为字符数；CTC 通常还需要一个 blank 类别。
""",
    "把 `C` 从 5 改成 7，观察只有输出最后一轴改变；再说明为什么 B 和 T 不变。",
)

cells += lesson(
    8,
    "损失、梯度、optimizer 与完整训练一步",
    "逐行区分前向计算、反向传播和参数更新。",
    """
训练一步固定包含：清旧梯度 → 前向 → 算损失 → 反向 → 更新。下面用最小回归演示，ASR 只是在模型和损失上更复杂。
""",
    """
import torch
from torch import nn

torch.manual_seed(0)  # 固定随机种子，让本次演示可复现。

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
y = 2.0 * x + 1.0

weight_before = model.weight.detach().clone()

optimizer.zero_grad()       # 1. 清除上一步累积的梯度。
prediction = model(x)       # 2. 前向计算预测值。
loss = loss_fn(prediction, y)  # 3. 用预测和目标计算一个标量损失。
loss.backward()             # 4. 自动求每个参数对 loss 的梯度。

print("loss：", float(loss))
print("weight 梯度：", model.weight.grad)

optimizer.step()            # 5. 按学习率和梯度修改参数。
weight_after = model.weight.detach().clone()

print("更新前 weight：", weight_before)
print("更新后 weight：", weight_after)
assert not torch.equal(weight_before, weight_after)
""",
    """
- `backward()` 计算梯度，但不会更新参数；`step()` 才更新。
- `.grad` 在 backward 之前通常是 `None`，之后才有数值。
- `detach().clone()` 得到不参与梯度、也不与原参数共享更新的快照。
""",
    """
- 忘记 `zero_grad()`：PyTorch 默认累加梯度，第二步会混入第一步。
- 把 `model.eval()` 当成“关闭梯度”；它只切换 Dropout/BatchNorm 行为。
- 在训练前把整个 forward 放进 `torch.no_grad()`，梯度图不会建立。
""",
    "先把 `optimizer.step()` 注释掉，验证参数不再变化；之后恢复，保证后续执行副本一致。",
)

cells += lesson(
    9,
    "变长语音的 padding、lengths 与 mask",
    "理解补零只是为了组成矩形，真实长度必须一直保留。",
    """
- 两条语音长度不同时，batch Tensor 仍必须是规则矩形。
- `pad_sequence` 补零；`lengths` 告诉模型/损失哪些位置是真的。
- mask 为 True 的位置通常表示有效，但不同库可能相反，要读 API contract。
""",
    """
import torch
from torch.nn.utils.rnn import pad_sequence

# 两条语音分别有 3 帧和 5 帧；每帧 2 维特征。
utterance_a = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
utterance_b = torch.tensor([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0], [8.0, 8.0]])
items = [utterance_a, utterance_b]

lengths = torch.tensor([item.shape[0] for item in items], dtype=torch.long)
padded = pad_sequence(items, batch_first=True, padding_value=0.0)

T_max = padded.shape[1]
time_index = torch.arange(T_max).unsqueeze(0)  # [T] -> [1,T]
valid_mask = time_index < lengths.unsqueeze(1)  # [1,T] 与 [B,1] 广播成 [B,T]

print("padded [B,T,F]：\\n", padded)
print("lengths [B]：", lengths)
print("valid_mask [B,T]：\\n", valid_mask)

assert padded.shape == (2, 5, 2)
assert valid_mask.sum(dim=1).tolist() == lengths.tolist()
""",
    """
- 第一条语音后两帧是 padding，不属于真实声音。
- `lengths=[3,5]` 的单位是“特征帧”，不是原始采样点，也不是 token 数。
- 比较式通过广播一次生成整个 `[B,T]` mask。
""",
    """
- 对 padding 一起求均值，使短语音被更多零稀释。
- 卷积下采样后忘记同步更新 lengths。
- 把 `lengths` 放在 CPU、logits 放在 GPU，而所用算子要求设备一致或相反；以 API 文档为准。
""",
    "把第一条语音增加一帧，先预测 `padded`、`lengths` 和 `valid_mask` 哪些部分会变化。",
)

cells += lesson(
    10,
    "声音如何变成模型输入：waveform → frame → spectrum → Log-Mel",
    "沿着最小声学前端跟踪单位和 shape。",
    """
- waveform 的横轴单位是采样点；分帧后时间轴单位变成帧。
- FFT 把一帧从时域变成频率 bin；这里只演示 log-power，正式 Mel 滤波器在主线第 5～6 课。
- 每一步都打印 shape，比死记公式更可靠。
""",
    """
import torch

sample_rate = 16_000
duration = 0.10
num_samples = int(sample_rate * duration)
time = torch.arange(num_samples) / sample_rate

# 440 Hz 正弦波，shape [S]。
waveform = 0.5 * torch.sin(2 * torch.pi * 440.0 * time)

frame_length = 400  # 25 ms × 16 kHz
hop_length = 160    # 10 ms × 16 kHz

# unfold 沿采样轴滑窗：[S] -> [T, frame_length]。
frames = waveform.unfold(dimension=0, size=frame_length, step=hop_length)
window = torch.hann_window(frame_length)
windowed = frames * window

# rfft 的频率 bin 数为 frame_length // 2 + 1。
spectrum = torch.fft.rfft(windowed, dim=-1)
power = spectrum.abs().square()
log_power = torch.log(power.clamp_min(1e-10))

print("waveform [S]：", waveform.shape)
print("frames [T,frame_length]：", frames.shape)
print("spectrum [T,freq_bins]：", spectrum.shape)
print("log_power [T,freq_bins]：", log_power.shape)

assert frames.shape[1] == frame_length
assert spectrum.shape[1] == frame_length // 2 + 1
assert torch.isfinite(log_power).all()
""",
    """
- 0.1 秒语音不是恰好 10 帧；是否补尾、是否居中会改变边界帧数。
- `rfft` 只保留实信号不重复的半边频谱，所以得到 201 个 bin。
- `clamp_min` 防止对 0 取 log 得到负无穷。
""",
    """
- 把毫秒直接当采样点，例如把 25 ms 写成 25 个点。
- 混淆窗长与帧移：窗长决定每帧看多宽，帧移决定多久输出一帧。
- 忽略不同库的 `center/pad` 默认值，导致帧数差 1～数帧。
""",
    "把频率从 440 改成 880；shape 不会变化，但最大能量所在的频率 bin 应该向高频移动。",
)

cells += lesson(
    11,
    "CTC 的 logits、log_probs、targets 与 lengths",
    "把 CTCLoss 的每个输入和 shape 对齐，避免最常见的静默错误。",
    """
- 模型通常输出 `[B,T,C]` logits；PyTorch CTCLoss 要 `[T,B,C]` log-probabilities。
- `input_lengths` 是每条语音有效输出帧数；`target_lengths` 是每条标签长度。
- `targets` 可拼成一维；blank ID 不能作为普通目标 token。
""",
    """
import torch
from torch import nn

torch.manual_seed(1)
B, T, C = 2, 6, 4  # 类别 0 是 blank，1~3 是普通 token。

logits_btc = torch.randn(B, T, C, requires_grad=True)

# log_softmax 把每帧类别分数变成对数概率；transpose 变成 CTCLoss 所需 [T,B,C]。
log_probs_tbc = logits_btc.log_softmax(dim=-1).transpose(0, 1)

# 第 1 条目标 [1,2]，第 2 条目标 [2]，拼成一维 [1,2,2]。
targets = torch.tensor([1, 2, 2], dtype=torch.long)
input_lengths = torch.tensor([6, 5], dtype=torch.long)
target_lengths = torch.tensor([2, 1], dtype=torch.long)

ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
loss = ctc_loss_fn(log_probs_tbc, targets, input_lengths, target_lengths)
loss.backward()

print("logits [B,T,C]：", logits_btc.shape)
print("log_probs [T,B,C]：", log_probs_tbc.shape)
print("targets：", targets, "target_lengths：", target_lengths)
print("CTC loss：", float(loss))
print("logits 梯度有限：", bool(torch.isfinite(logits_btc.grad).all()))

assert log_probs_tbc.shape == (T, B, C)
assert target_lengths.sum().item() == targets.numel()
assert input_lengths.max().item() <= T
""",
    """
- `log_softmax(dim=-1)` 必须沿类别 C 轴归一化。
- `transpose(0,1)` 只交换 T/B，不改变 C。
- 三个断言分别检查 shape、目标拼接长度和有效输入长度。
""",
    """
- 直接把 softmax 概率传给 CTCLoss；它需要 log-probabilities。
- 下采样后仍传原始特征长度，`input_lengths` 超过 logits 的 T。
- 目标含连续重复 token 时，最短可对齐长度不只是 target length，还要插入 blank。
""",
    "把 `input_lengths[1]` 临时改成 7，观察错误信息；然后恢复为 5。",
)

cells += lesson(
    12,
    "CTC Greedy 解码：argmax 不是最终文本",
    "理解逐帧路径如何经过“合并重复、删除 blank”变成 token。",
    """
- `argmax` 每帧选最大类别，得到的是 frame path。
- CTC collapse 的顺序是先合并连续重复，再删除 blank。
- 被 blank 隔开的相同 token 不会被合并，例如 `1, blank, 1 -> 1,1`。
""",
    """
import torch


def ctc_collapse(path: list[int], blank_id: int = 0) -> list[int]:
    tokens = []
    previous = None
    for current in path:
        # 只有“与上一帧不同”且“不是 blank”的符号才输出。
        if current != previous and current != blank_id:
            tokens.append(current)
        previous = current
    return tokens


# 0=blank。逐帧路径有 10 帧，最终只有 4 个 token。
frame_path = [0, 1, 1, 0, 2, 2, 0, 1, 0, 1]
tokens = ctc_collapse(frame_path, blank_id=0)

vocabulary = {1: "你", 2: "好"}
text = "".join(vocabulary[token] for token in tokens)

print("逐帧路径：", frame_path)
print("折叠 token：", tokens)
print("最终文本：", text)

assert tokens == [1, 2, 1, 1]
assert text == "你好你你"
""",
    """
- 路径中的相邻 `1,1` 合并一次。
- 最后的两个 1 中间有 blank，因此代表两个相同 token。
- Greedy 简单但不一定得到总概率最大的文本；Prefix Beam 在主线第 13 课。
""",
    """
- 先删 blank 再合并重复，会把 `1,blank,1` 错误变成一个 1。
- 把 frame path 的长度当成文字长度。
- 忘记词表和模型的 token ID 必须完全一致。
""",
    "把 `frame_path` 改成 `[1,1,0,1,1]`，先预测最后有几个 token。",
)

cells += lesson(
    13,
    "流式 chunk、cache 与 reset",
    "理解流式不是把整段输入随便切开，而是跨块保存必要历史。",
    """
- 下面用“跨 chunk 累计和”模拟有状态流式计算。
- `state` 是上一块结束时的累计值，必须传到下一块。
- 新会话必须 reset，否则会把上一位用户的历史带进来。
""",
    """
import torch


class StreamingCumulativeSum:
    def __init__(self):
        self.state = torch.tensor(0.0)

    def reset(self) -> None:
        self.state = torch.tensor(0.0)

    def process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        # 当前块内部累计，再加上前一块留下的 state。
        output = torch.cumsum(chunk, dim=0) + self.state
        # detach + clone：缓存数值，不保留旧的训练计算图或共享视图。
        self.state = output[-1].detach().clone()
        return output


stream = StreamingCumulativeSum()
chunks = [torch.tensor([1.0, 2.0]), torch.tensor([3.0]), torch.tensor([4.0, 5.0])]
stream_outputs = [stream.process_chunk(chunk) for chunk in chunks]
stream_result = torch.cat(stream_outputs)

offline_input = torch.cat(chunks)
offline_result = torch.cumsum(offline_input, dim=0)

print("流式：", stream_result)
print("离线：", offline_result)
print("最大误差：", float((stream_result - offline_result).abs().max()))

assert torch.equal(stream_result, offline_result)

stream.reset()
assert stream.state.item() == 0.0
""",
    """
- 每块输出长度可以不同，拼接后应与离线参考完全一致。
- 若每块都从 0 开始累计，第二块开始就会错，但每个函数仍“能运行”。
- 真正 ASR 的 cache 可能是卷积历史帧、注意力 K/V、解码器状态和未完成音频帧。
""",
    """
- chunk 结束就清空状态，造成边界不连续。
- 不同 WebSocket 会话共享同一 cache，造成用户间串话。
- 忘记 EOF/flush：最后不足一帧的数据永远不被处理。
""",
    "删除 `+ self.state` 后比较流式和离线结果，定位从第几个 chunk 开始不同；之后恢复。",
)

cells += lesson(
    14,
    "系统化排错：值、shape、dtype、device、finite、gradient",
    "遇到报错或训练异常时有固定检查顺序，而不是重装环境。",
    """
先看 Traceback 最后一行，再向上找第一处自己的代码。若没有报错但结果异常，就在模块边界做审计。
""",
    """
import torch


def audit_tensor(name: str, tensor: torch.Tensor) -> None:
    # 打印 Tensor 的最小诊断信息；不修改输入。
    print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
    if tensor.is_floating_point():
        finite = torch.isfinite(tensor)
        print(
            "  finite=", bool(finite.all()),
            "min=", float(tensor[finite].min()) if finite.any() else "无有限值",
            "max=", float(tensor[finite].max()) if finite.any() else "无有限值",
        )


healthy = torch.tensor([[0.0, 1.0], [-1.0, 2.0]], dtype=torch.float32)
suspicious = torch.tensor([0.0, float("nan"), float("inf")])

audit_tensor("healthy", healthy)
audit_tensor("suspicious", suspicious)

assert torch.isfinite(healthy).all()
assert not torch.isfinite(suspicious).all()
""",
    """
建议固定按以下顺序排错：

1. 文件/环境是否正确；
2. 输入值和单位；
3. shape 与轴含义；
4. dtype/device；
5. NaN/Inf；
6. lengths/mask；
7. gradient 是否存在且有限；
8. 与一个更小、更慢但可信的参考实现对照。
""",
    """
- 只检查 loss，不检查进入模型的数据，定位太晚。
- 因为代码没报错就认为语义正确。
- 一次同时修改很多地方，无法知道哪一项真正解决问题。
""",
    "把 `healthy` 的一个值改成 `float('nan')`，观察审计输出和断言分别怎样暴露问题。",
)

cells += [
    md(
        """
## 15. 怎样把伴读方法带回 46 节主课

以后每看到一个代码单元，在上方新建 Markdown cell，先填写这张卡；题目可以之后再答，但 shape 卡建议当场写：

```text
输入变量：
输入 shape / dtype / 单位：
本单元新增的函数或类：
每一行对数据做了什么：
输出 shape / dtype / 单位：
我预计看到的输出：
常见失败与检查方法：
```

主线中的对应关系：

| 本伴读节次 | 回到主课后重点观察 |
|---:|---|
| 1～3 | 所有 Python、导入、对象和函数调用 |
| 4～6 | 第 1～9 课：波形、频谱、Log-Mel、Tensor 和卷积 shape |
| 7～9 | 第 8～14 课：编码器、训练循环、变长 batch |
| 10 | 第 1～6 课：完整声学前端 |
| 11～12 | 第 10～14 课：CTC loss 和解码 |
| 13 | 第 15～18、23～24、29、36 课：流式状态 |
| 14 | 所有实验、部署与真实盲测 |

当代码还看不懂时，不需要硬撑到下一课。把不懂的那一行连同它前后的 `shape/dtype` 发给我，我会继续按这套格式拆开。
"""
    ),
    md(
        """
## 完成状态

到这里，你已经看过 ASR 代码里反复出现的核心结构：

- Python 数据流、函数和对象；
- NumPy/PyTorch 的 shape、axis、dtype、device；
- `nn.Module`、loss、gradient、optimizer；
- 变长 batch、声学前端、CTC、流式 cache；
- 一套可重复的排错顺序。

现在无需答题。下一步直接打开 `基础_01_Python最小语法与函数.ipynb`；每当遇到熟悉结构，就回来对照对应小节。
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "course": {
            "role": "beginner-code-companion",
            "audience": "absolute-beginner",
            "code_cells": 14,
        },
    },
)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    source_path = NOTEBOOK_DIR / f"{STEM}.ipynb"
    executed_output_path = executed_path(source_path)

    source = copy.deepcopy(notebook)
    nbf.write(source, source_path)

    ensure_executed_directories()
    executed = copy.deepcopy(notebook)
    client = NotebookClient(
        executed,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    sanitize_notebook_outputs(executed)
    nbf.write(executed, executed_output_path)

    print(f"wrote {source_path}")
    print(f"wrote {executed_output_path}")


if __name__ == "__main__":
    main()
