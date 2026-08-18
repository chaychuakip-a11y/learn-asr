from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

from notebook_layout import (
    ensure_executed_directories,
    executed_path,
    sanitize_notebook_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


@dataclass(frozen=True)
class Lesson:
    number: int
    slug: str
    title: str
    prerequisites: str
    concepts: tuple[str, str, str]
    core_cells: list
    exercises: str
    next_step: str


def intro(lesson: Lesson) -> list:
    concepts = "、".join(lesson.concepts)
    return [
        md(
            f"""
# PyTorch 零基础 {lesson.number}/6：{lesson.title}

这是第 1～41 课 ASR 主线之前的桥梁课。先预测，再运行；看懂输出后必须改一个值验证自己的解释。

| 项目 | 内容 |
|---|---|
| 前置要求 | {lesson.prerequisites} |
| 建议投入 | 60～90 分钟，可分两次完成 |
| 核心概念 | {concepts} |
| 完成标准 | 能解释代码、独立完成练习、从空白重写本课核心函数 |
"""
        ),
        md(
            f"""
## 课前诊断（先不要运行代码）

1. 用自己的话解释：{lesson.concepts[0]}。
2. 猜测 {lesson.concepts[1]} 最容易出现哪一种错误。
3. 写下你对 {lesson.concepts[2]} 的暂时理解；不会可以明确写“不知道”。

这三题不计分，只用于留下学习前证据。
"""
        ),
    ]


def exercise_block(lesson: Lesson) -> list:
    return [
        md(
            f"""
## 本课练习（保留作答区）

{lesson.exercises}

评分：每题 0～2 分。达到 16/20 可以继续；12～15 分次日重做错题；低于 12 分回看代码并从空白复现。
"""
        ),
        md(
            f"""
## 离场票与间隔复习

- [ ] 我能闭卷解释：{'、'.join(lesson.concepts)}。
- [ ] 我能预测核心代码的 shape、dtype 或数值方向。
- [ ] 我能从空白重写至少一个函数，并通过正常、边界、错误输入测试。
- [ ] 我能说出一个“代码能运行但语义错误”的例子。

复习安排：明天闭卷回忆 5 分钟；7 天后重做第 4、7、10 题；30 天后重新构造最小实验。

下一步：{lesson.next_step}
"""
        ),
    ]


LESSONS = [
    Lesson(
        number=1,
        slug="Python最小语法与函数",
        title="Python 最小语法与函数",
        prerequisites="会打开 Notebook，并知道 Shift+Enter",
        concepts=("变量与数据类型", "列表、索引与循环", "函数、异常与断言"),
        core_cells=[
            md(
                """
## 1. 变量不是公式，是带名字的数据

`=` 表示把右边的值绑定到左边的名字。运行前先预测 `num_samples` 和每个变量的类型。
"""
            ),
            code(
                """
sample_rate = 16_000
duration_seconds = 2.5
num_samples = int(sample_rate * duration_seconds)
utterance = "你好"
is_training = True

print(num_samples)
print(type(sample_rate), type(duration_seconds), type(utterance), type(is_training))

assert num_samples == 40_000
"""
            ),
            md(
                """
## 2. 列表、索引和切片

索引从 0 开始；切片 `[start:stop]` 包含 start、不包含 stop。负索引从末尾倒数。
"""
            ),
            code(
                """
tokens = ["你", "好", "语", "音"]
print("第一个：", tokens[0])
print("最后一个：", tokens[-1])
print("中间两个：", tokens[1:3])
print("长度：", len(tokens))

assert tokens[1:3] == ["好", "语"]
"""
            ),
            md("## 3. 条件和循环\n\n循环让同一规则作用于多个元素；条件让规则只在满足要求时执行。"),
            code(
                """
lengths = [16_000, 8_000, 32_000]
long_utterances = []
for length in lengths:
    seconds = length / sample_rate
    print(f"{length} samples -> {seconds:.1f} s")
    if seconds >= 1.0:
        long_utterances.append(length)

assert long_utterances == [16_000, 32_000]
"""
            ),
            md(
                """
## 4. 函数：把规则命名并重复使用

函数的输入叫参数，`return` 给出输出。先明确输入单位和错误边界，比背语法重要。
"""
            ),
            code(
                """
def samples_to_seconds(num_samples: int, sample_rate: int) -> float:
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return num_samples / sample_rate


assert samples_to_seconds(32_000, 16_000) == 2.0
assert samples_to_seconds(0, 16_000) == 0.0

try:
    samples_to_seconds(100, 0)
except ValueError as error:
    print("捕获到预期错误：", error)
else:
    raise AssertionError("错误输入应当抛出 ValueError")
"""
            ),
            md(
                """
## 5. 如何读报错

先看最后一行的异常类型和消息，再向上找第一处属于自己代码的行。不要一看到红字就重装环境。
"""
            ),
            code(
                """
def require_mono(channel_count: int) -> None:
    assert isinstance(channel_count, int), "channel_count must be int"
    assert channel_count == 1, f"expected mono, got {channel_count} channels"


require_mono(1)
print("正常输入通过断言")
"""
            ),
        ],
        exercises="""
1. `sample_rate = 16000` 中变量名和值分别是什么？
2. 预测 `[10, 20, 30, 40][1:3]`。
3. 写循环打印 0、1、2 的平方。
4. 把 48,000 个采样点在 16 kHz 下换算成秒。
5. 解释为什么切片的 stop 不包含在结果中容易产生 off-by-one。
6. 编写 `seconds_to_samples(seconds, sample_rate)`，检查负时长和非法采样率。
7. 为函数写正常、边界、错误输入三类测试。
8. 故意访问列表越界，记录异常类型和最后一行消息。
9. 用一句话区分 `print` 与 `return`。
10. 说明这套语法怎样用于“遍历一批语音并检查长度”。
""",
        next_step="基础 2：Tensor 创建、dtype、shape 与索引。",
    ),
    Lesson(
        number=2,
        slug="Tensor创建索引与Shape",
        title="Tensor 创建、索引与 shape",
        prerequisites="完成基础 1；能看懂变量、列表、函数调用",
        concepts=("Tensor 与 dtype", "shape、ndim 与 numel", "索引、切片与维度"),
        core_cells=[
            md("## 1. Tensor 是同一种 dtype 的多维数字容器"),
            code(
                """
import torch

waveform = torch.tensor([0.0, 0.25, -0.5, 0.25], dtype=torch.float32)
token_ids = torch.tensor([3, 8, 2], dtype=torch.long)

print(waveform)
print("waveform:", waveform.shape, waveform.dtype, waveform.ndim, waveform.numel())
print("token_ids:", token_ids.shape, token_ids.dtype)

assert waveform.shape == (4,)
assert waveform.numel() == 4
assert token_ids.dtype == torch.int64
"""
            ),
            md(
                """
## 2. shape 的每个轴都必须有语义

ASR 常用 `[B,T,F]`：`B` 是 batch，`T` 是时间帧，`F` 是每帧特征。shape 不是装饰信息，它决定操作含义。
"""
            ),
            code(
                """
B, T, F = 2, 5, 3
features = torch.arange(B * T * F, dtype=torch.float32).reshape(B, T, F)

print("features shape:", features.shape)
print("第 1 条语音:", features[0].shape)
print("第 1 条语音第 2 帧:", features[0, 1].shape)
print("所有语音前两帧:", features[:, :2, :].shape)

assert features.shape == (2, 5, 3)
assert features[0].shape == (5, 3)
assert features[0, 1].shape == (3,)
assert features[:, :2, :].shape == (2, 2, 3)
"""
            ),
            md("## 3. 增加或删除长度为 1 的轴"),
            code(
                """
single_waveform = torch.zeros(16_000)
batch_of_one = single_waveform.unsqueeze(0)
restored = batch_of_one.squeeze(0)

print(single_waveform.shape, "->", batch_of_one.shape, "->", restored.shape)
assert batch_of_one.shape == (1, 16_000)
assert torch.equal(restored, single_waveform)
"""
            ),
            md(
                """
## 4. dtype 不匹配是常见接口错误

神经网络输入通常是浮点数，类别编号通常是 `torch.long`。不要靠猜测转换，先打印契约。
"""
            ),
            code(
                """
float_features = token_ids.float()
back_to_ids = float_features.long()

print(float_features, float_features.dtype)
print(back_to_ids, back_to_ids.dtype)
assert torch.equal(back_to_ids, token_ids)
"""
            ),
            md("## 5. 设备是 Tensor 契约的一部分"),
            code(
                """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
moved = features.to(device)
print("本机教学设备：", moved.device)
assert moved.device.type in {"cpu", "cuda"}
"""
            ),
        ],
        exercises="""
1. 写出 `torch.zeros(4, 100, 80)` 的 B、T、F 和数字总数。
2. 预测 `x=torch.zeros(2,3,4)` 时 `x[0].shape`。
3. 预测 `x[:, 1:, :2].shape`。
4. 解释为什么类别 ID 常用 `torch.long`。
5. 把 `[T]` 波形变成 `[1,T]`，再还原并验证数值一致。
6. 创建正常、空 Tensor，检查 shape 和 numel。
7. 故意给索引超过范围，记录报错。
8. 写 `describe_tensor(x)`，打印 shape、dtype、device、min/max。
9. 解释 `squeeze()` 不指定轴可能造成的隐患。
10. 画出 `[B,T,F] -> features[0] -> features[0,0]` 的轴变化。
""",
        next_step="基础 3：运算、广播、矩阵乘法与维度变换。",
    ),
    Lesson(
        number=3,
        slug="Tensor运算广播与维度变换",
        title="Tensor 运算、广播与维度变换",
        prerequisites="完成基础 2；能解释任意示例 Tensor 的每个轴",
        concepts=("逐元素运算与归约", "广播规则", "reshape、transpose 与矩阵乘法"),
        core_cells=[
            md("## 1. 逐元素运算与归约"),
            code(
                """
import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print("x * 2 =\\n", x * 2)
print("全部均值：", x.mean())
print("每行均值：", x.mean(dim=1))
print("每列最大值：", x.max(dim=0).values)

assert x.mean(dim=1).shape == (2,)
assert torch.allclose(x.mean(dim=1), torch.tensor([2.0, 5.0]))
"""
            ),
            md(
                """
## 2. 广播：从尾部维度对齐

两个维度相等或其中一个为 1 时可以广播。能运行不代表语义正确，所以先写 shape 再写算式。
"""
            ),
            code(
                """
features = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
feature_bias = torch.tensor([10.0, 20.0, 30.0])
shifted = features + feature_bias

print(features.shape, "+", feature_bias.shape, "->", shifted.shape)
assert shifted.shape == (2, 4, 3)
assert torch.equal(shifted[0, 0], features[0, 0] + feature_bias)
"""
            ),
            md("## 3. reshape 改分组方式，transpose/permute 改轴顺序"),
            code(
                """
x = torch.arange(24).reshape(2, 3, 4)
flat_time = x.reshape(2, 12)
swapped = x.transpose(1, 2)
conv_layout = x.permute(0, 2, 1)

print("原始 [B,T,F]：", x.shape)
print("reshape：", flat_time.shape)
print("transpose：", swapped.shape)
print("Conv1d 常用 [B,F,T]：", conv_layout.shape)

assert swapped.shape == (2, 4, 3)
assert torch.equal(swapped, conv_layout)
"""
            ),
            md("## 4. 矩阵乘法把最后一个特征轴映射到新维度"),
            code(
                """
B, T, F, H = 2, 5, 3, 4
features = torch.randn(B, T, F)
weight = torch.randn(F, H)
encoded = features @ weight

print("[B,T,F] @ [F,H] ->", encoded.shape)
assert encoded.shape == (B, T, H)
"""
            ),
            md("## 5. Mask 也依赖广播"),
            code(
                """
values = torch.tensor([[[1.0], [2.0], [99.0]], [[3.0], [99.0], [99.0]]])
mask = torch.tensor([[True, True, False], [True, False, False]])
masked = values.masked_fill(~mask.unsqueeze(-1), 0.0)

print(masked.squeeze(-1))
assert torch.equal(masked.squeeze(-1), torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]]))
"""
            ),
        ],
        exercises="""
1. `x.shape=[2,3,4]` 时，`x.mean(dim=1).shape` 是什么？
2. `[2,5,80] + [80]` 为什么能广播？
3. 判断 `[2,5,80] + [5]` 是否能广播，并运行验证。
4. 把 `[B,T,F]` 转成 Conv1d 所需 `[B,F,T]`。
5. 解释 reshape 与 permute 的区别。
6. 写一个 `[B,T,F] @ [F,H]` 示例并断言输出 shape。
7. 故意制造矩阵乘法维度不匹配，读懂报错中的 shape。
8. 实现按最后一维做零均值标准化，注意数值稳定性。
9. 比较 `mean()`、`mean(dim=1)` 和 `mean(dim=(1,2))`。
10. 解释为什么广播错误有时不会报错却会悄悄改变语义。
""",
        next_step="基础 4：autograd、loss、梯度与优化器。",
    ),
    Lesson(
        number=4,
        slug="Autograd损失与优化器",
        title="Autograd、损失与优化器",
        prerequisites="完成基础 3；理解 Tensor 运算和矩阵乘法",
        concepts=("计算图与 requires_grad", "loss 与 backward", "zero_grad 与 optimizer.step"),
        core_cells=[
            md("## 1. PyTorch 自动记录参数如何影响结果"),
            code(
                """
import torch

w = torch.tensor(0.0, requires_grad=True)
x = torch.tensor(3.0)
prediction = w * x
loss = (prediction - 6.0) ** 2
loss.backward()

print("prediction:", prediction.item())
print("loss:", loss.item())
print("d(loss)/d(w):", w.grad.item())
assert w.grad.item() == -36.0
"""
            ),
            md(
                """
## 2. 梯度不是更新；optimizer.step 才修改参数

`backward()` 计算方向，`step()` 执行更新。梯度默认累加，因此每轮需要清零。
"""
            ),
            code(
                """
w = torch.nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.SGD([w], lr=0.05)

for step in range(6):
    prediction = w * 3.0
    loss = (prediction - 6.0) ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"step={step}, w={w.item():.4f}, loss={loss.item():.4f}")

assert abs(w.item() - 2.0) < 0.01
"""
            ),
            md("## 3. 完整的最小线性回归"),
            code(
                """
torch.manual_seed(0)
x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
y = 2 * x + 1
model = torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

history = []
for step in range(80):
    prediction = model(x)
    loss = loss_fn(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    history.append(loss.item())

print("first/final loss:", history[0], history[-1])
print("learned weight/bias:", model.weight.item(), model.bias.item())
assert history[-1] < history[0] * 1e-4
assert abs(model.weight.item() - 2.0) < 0.02
assert abs(model.bias.item() - 1.0) < 0.02
"""
            ),
            md("## 4. 推理时不需要梯度"),
            code(
                """
with torch.no_grad():
    test_prediction = model(torch.tensor([[4.0]]))

print("x=4 prediction:", test_prediction.item())
assert not test_prediction.requires_grad
assert abs(test_prediction.item() - 9.0) < 0.05
"""
            ),
            md("## 5. 梯度审计：先看是否存在，再看是否有限"),
            code(
                """
for name, parameter in model.named_parameters():
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    print(name, "grad norm =", parameter.grad.norm().item())
"""
            ),
        ],
        exercises="""
1. 区分 prediction、target 和 loss。
2. `backward()` 与 `optimizer.step()` 各做什么？
3. 为什么每轮要 `zero_grad()`？
4. 把学习率改成 0、0.01、1.0，预测并记录 loss 曲线。
5. 手算 `w=0,x=3,target=6` 时平方误差对 w 的梯度。
6. 从空白重写最小线性回归训练循环。
7. 故意删除 `zero_grad()`，比较参数和梯度变化。
8. 加入断言检查 loss、梯度和参数都为有限数。
9. 解释 `torch.no_grad()` 为什么用于验证或推理。
10. 把直线训练的 input/model/loss 映射到 ASR 的特征/编码器/CTC loss。
""",
        next_step="基础 5：nn.Module、训练/验证模式与保存加载。",
    ),
    Lesson(
        number=5,
        slug="nnModule与训练验证循环",
        title="nn.Module 与训练、验证循环",
        prerequisites="完成基础 4；能解释训练循环的五步",
        concepts=("nn.Module 与 forward", "train/eval 模式", "state_dict 与可复现性"),
        core_cells=[
            md("## 1. 用 nn.Module 组织有参数的计算"),
            code(
                """
import io
import torch
from torch import nn


class TinyEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"expected [B,T,F], got {tuple(features.shape)}")
        return self.network(features)


torch.manual_seed(7)
encoder = TinyEncoder(input_dim=8, hidden_dim=12)
x = torch.randn(2, 5, 8)
y = encoder(x)
print("input/output:", x.shape, y.shape)
assert y.shape == (2, 5, 12)
"""
            ),
            md("## 2. 参数必须注册在 Module 中"),
            code(
                """
parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
print("parameter count:", parameter_count)
for name, parameter in encoder.named_parameters():
    print(name, tuple(parameter.shape))
assert parameter_count > 0
"""
            ),
            md("## 3. train/eval 会改变 Dropout、BatchNorm 等层的行为"),
            code(
                """
dropout_model = nn.Sequential(nn.Linear(8, 8), nn.Dropout(p=0.5))
example = torch.ones(64, 8)

dropout_model.train()
train_a = dropout_model(example)
train_b = dropout_model(example)

dropout_model.eval()
with torch.no_grad():
    eval_a = dropout_model(example)
    eval_b = dropout_model(example)

print("train outputs equal:", torch.equal(train_a, train_b))
print("eval outputs equal:", torch.equal(eval_a, eval_b))
assert not torch.equal(train_a, train_b)
assert torch.equal(eval_a, eval_b)
"""
            ),
            md("## 4. 一个可复用的训练步骤"),
            code(
                """
classifier = nn.Linear(4, 3)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()


def train_step(inputs: torch.Tensor, targets: torch.Tensor) -> float:
    classifier.train()
    logits = classifier(inputs)
    loss = loss_fn(logits, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


inputs = torch.randn(6, 4)
targets = torch.tensor([0, 1, 2, 0, 1, 2])
loss_value = train_step(inputs, targets)
print("train loss:", loss_value)
assert loss_value > 0
"""
            ),
            md("## 5. state_dict 保存的是可复现的参数状态"),
            code(
                """
buffer = io.BytesIO()
torch.save(encoder.state_dict(), buffer)
buffer.seek(0)

clone = TinyEncoder(input_dim=8, hidden_dim=12)
clone.load_state_dict(torch.load(buffer, weights_only=True))
clone.eval()
encoder.eval()
with torch.no_grad():
    original_output = encoder(x)
    clone_output = clone(x)

assert torch.equal(original_output, clone_output)
print("state_dict round-trip: exact match")
"""
            ),
        ],
        exercises="""
1. `__init__` 与 `forward` 分别负责什么？
2. 为什么层要赋给 `self.xxx`？
3. 预测 `[B,T,8]` 经过 `Linear(8,12)` 的 shape。
4. 比较 train/eval 下 Dropout 输出。
5. 写函数统计总参数量和可训练参数量。
6. 为 TinyEncoder 加分类头并断言 logits shape。
7. 故意传入 `[B,F]`，验证接口错误消息。
8. 实现不更新参数的 `validation_step`。
9. 保存、加载 state_dict 并比较固定输入输出。
10. 列出实验复现至少需要保存的随机种子、代码、参数和数据版本。
""",
        next_step="基础 6：Dataset、DataLoader、变长语音 padding 与 mask。",
    ),
    Lesson(
        number=6,
        slug="Dataset_DataLoader与变长语音Batch",
        title="Dataset、DataLoader 与变长语音 Batch",
        prerequisites="完成基础 5；会定义 nn.Module 和训练步骤",
        concepts=("Dataset 与 DataLoader", "collate、padding 与 lengths", "mask 与有效区域"),
        core_cells=[
            md("## 1. Dataset 定义一个样本，DataLoader 组织迭代与 batch"),
            code(
                """
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class ToySpeechDataset(Dataset):
    def __init__(self):
        self.waveforms = [
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([0.4, 0.5, 0.6, 0.7, 0.8]),
            torch.tensor([0.9, 1.0]),
        ]
        self.labels = [0, 1, 0]

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, index):
        return self.waveforms[index], self.labels[index]


dataset = ToySpeechDataset()
waveform, label = dataset[1]
print(len(dataset), waveform.shape, label)
assert len(dataset) == 3
assert waveform.shape == (5,)
"""
            ),
            md("## 2. 变长语音不能直接 stack，需要 collate"),
            code(
                """
def speech_collate(batch):
    waveforms, labels = zip(*batch)
    lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long)
    padded = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    return {"waveforms": padded, "lengths": lengths, "labels": labels}


loader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=speech_collate)
batch = next(iter(loader))
for key, value in batch.items():
    print(key, value.shape, value.dtype)

assert batch["waveforms"].shape == (3, 5)
assert torch.equal(batch["lengths"], torch.tensor([3, 5, 2]))
assert batch["labels"].shape == (3,)
"""
            ),
            md("## 3. lengths 生成 mask，区分真实数据与 padding"),
            code(
                """
def lengths_to_mask(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError(f"expected [B] lengths, got {tuple(lengths.shape)}")
    if (lengths < 0).any():
        raise ValueError("lengths must be non-negative")
    if max_length is None:
        max_length = int(lengths.max()) if lengths.numel() else 0
    steps = torch.arange(max_length, device=lengths.device)
    return steps.unsqueeze(0) < lengths.unsqueeze(1)


mask = lengths_to_mask(batch["lengths"], batch["waveforms"].shape[1])
print(mask)
assert mask.shape == (3, 5)
assert torch.equal(mask.sum(dim=1), batch["lengths"])
"""
            ),
            md("## 4. Masked mean 不让 padding 污染统计量"),
            code(
                """
padded = batch["waveforms"]
valid_sum = (padded * mask).sum(dim=1)
masked_mean = valid_sum / batch["lengths"].clamp_min(1)

manual = torch.tensor([
    dataset.waveforms[0].mean(),
    dataset.waveforms[1].mean(),
    dataset.waveforms[2].mean(),
])
print("masked mean:", masked_mean)
assert torch.allclose(masked_mean, manual)
"""
            ),
            md("## 5. 每个 batch 都要审计接口契约"),
            code(
                """
def audit_speech_batch(batch):
    waveforms = batch["waveforms"]
    lengths = batch["lengths"]
    labels = batch["labels"]
    assert waveforms.ndim == 2
    assert lengths.shape == labels.shape == (waveforms.shape[0],)
    assert waveforms.dtype == torch.float32
    assert lengths.dtype == labels.dtype == torch.long
    assert (lengths <= waveforms.shape[1]).all()
    assert torch.isfinite(waveforms).all()


audit_speech_batch(batch)
print("batch contract passed")
"""
            ),
        ],
        exercises="""
1. 用一句话区分 Dataset 与 DataLoader。
2. 为什么变长波形不能直接 `torch.stack`？
3. 写出示例 batch 中 waveforms、lengths、labels 的 shape。
4. 根据 lengths=[3,5,2] 手画 bool mask。
5. 解释为什么只补零但不传 length/mask 会出错。
6. 为 collate 加入空波形策略并说明选择。
7. 为 lengths_to_mask 写正常、全零、空、负数测试。
8. 故意把 labels 变成 float，确认审计器能发现。
9. 比较普通 mean 和 masked mean 的数值差异。
10. 画出 Dataset -> collate -> padded batch -> encoder -> loss 的数据流并标 shape/dtype。
""",
        next_step="进入 ASR 主线第 1 课《声音与采样》，之后第 7～9 课会把这些能力用于声学编码器。",
    ),
]


def build_notebook(lesson: Lesson):
    cells = intro(lesson) + lesson.core_cells + exercise_block(lesson)
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
            "foundation_course": {
                "lesson": lesson.number,
                "total_lessons": len(LESSONS),
                "title": lesson.title,
            },
        },
    )
    _, notebook = nbf.validator.normalize(notebook, strip_invalid_metadata=True)
    return notebook


def main():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    ensure_executed_directories()
    for lesson in LESSONS:
        base_name = f"基础_{lesson.number:02d}_{lesson.slug}"
        source_path = NOTEBOOK_DIR / f"{base_name}.ipynb"
        executed_output = executed_path(source_path)

        source_notebook = build_notebook(lesson)
        nbf.write(source_notebook, source_path)

        executed_notebook = copy.deepcopy(source_notebook)
        client = NotebookClient(
            executed_notebook,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()
        sanitize_notebook_outputs(executed_notebook)
        _, executed_notebook = nbf.validator.normalize(
            executed_notebook,
            strip_invalid_metadata=True,
        )
        nbf.write(executed_notebook, executed_output)
        code_count = sum(cell.cell_type == "code" for cell in source_notebook.cells)
        print(f"built {base_name}: {len(source_notebook.cells)} cells, {code_count} code cells")


if __name__ == "__main__":
    main()
