"""Build and execute the post-CTC frontier ASR lessons.

The generated notebooks are intentionally small enough to run on CPU. They
teach architecture and invariants; they do not claim production accuracy.
"""

from __future__ import annotations

import argparse
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
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def lesson_42() -> nbf.NotebookNode:
    cells = [
        md(
            r"""
# 第 42 课：从卷积与注意力到 Conformer

这一课回答四个问题：为什么纯卷积和纯 Transformer 都不够理想？Conformer 怎样同时建模局部发音和长距离上下文？`[B,T,D]` 的 shape 怎样流动？流式 Conformer 还需要改什么？

建议先完成第 7～14 课。CPU 约需 1～3 分钟。
"""
        ),
        md(
            r"""
## 学习导航与完成标准

完成后你应能：

1. 解释卷积、注意力、前馈网络各自负责什么；
2. 写出 Conformer block 的残差顺序；
3. 正确构造 padding mask，并说明 mask 为什么不改变时间长度；
4. 证明全上下文注意力会读取未来；
5. 说出改成流式模型必须处理的 attention window、因果卷积与 cache。

证据门槛：闭卷画出 block，从空白补完 `ConformerBlock.forward`，并通过本课三个断言。
"""
        ),
        md(
            r"""
## 课前诊断（不要运行代码）

1. 一个音素主要是局部模式，还是需要整句话才能辨认？
2. “我想吃苹___”中的缺失字为什么可能需要较远的上下文？
3. 输入 `[B,T,D]=[2,100,80]` 经过不下采样的编码块，输出 shape 应是什么？

先写下答案。第 1、2 题并不矛盾：现代编码器正是要同时处理两种尺度。
"""
        ),
        code(
            """
import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)
np.random.seed(7)
random.seed(7)
torch.set_num_threads(2)

print("torch:", torch.__version__)
print("device: cpu（本课故意保持小模型，CPU 即可）")
"""
        ),
        md(
            r"""
## 1. 为什么是 Conformer？

语音同时有两种结构：

- **局部结构**：几十毫秒内的共振峰、爆破、摩擦等发音线索，卷积擅长；
- **全局结构**：较远词语、句法和说话上下文，注意力擅长。

Conformer 把两者放在同一个残差块中。常见的 Macaron 形式为：

$$
x_1=x+\tfrac12\mathrm{FFN}(x)
$$
$$
x_2=x_1+\mathrm{MHSA}(x_1)
$$
$$
x_3=x_2+\mathrm{Conv}(x_2)
$$
$$
y=\mathrm{LayerNorm}(x_3+\tfrac12\mathrm{FFN}(x_3))
$$

两个 FFN 各乘 $1/2$，注意力负责全局，卷积负责局部。残差连接让每个模块只需学习对当前表示的修正。
"""
        ),
        code(
            """
def lengths_to_padding_mask(lengths: torch.Tensor, max_len=None):
    \"\"\"返回 [B,T]；True 表示 padding，供 MultiheadAttention 使用。\"\"\"
    max_len = int(max_len or lengths.max())
    t = torch.arange(max_len, device=lengths.device)
    return t.unsqueeze(0) >= lengths.unsqueeze(1)

lengths = torch.tensor([7, 4])
padding_mask = lengths_to_padding_mask(lengths)
print(padding_mask.int())

expected = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1],
], dtype=torch.bool)
assert torch.equal(padding_mask, expected)
print("断言通过：True 只出现在第二条样本的 padding 区域。")
"""
        ),
        md(
            r"""
### Mask 最容易犯的错

`lengths=[7,4]` 表示两条序列的真实长度。PyTorch `MultiheadAttention` 的 `key_padding_mask=True` 表示“不要读取这里”，不是“这里有效”。不同 API 的布尔语义可能相反，必须查清契约。

Mask 不会删除时间步，因此输入输出仍是 `[B,T,D]`。为了防止 padding 位置经过残差后出现非零值，我们在 block 末尾再显式清零。
"""
        ),
        code(
            """
class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.0):
        super().__init__()
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    \"\"\"教学版 Conformer 卷积模块；保持 [B,T,D] 不变。\"\"\"
    def __init__(self, d_model, kernel_size=7, dropout=0.0):
        super().__init__()
        assert kernel_size % 2 == 1, "离线 same padding 需要奇数 kernel"
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm(x).transpose(1, 2)       # [B,D,T]
        z = F.glu(self.pointwise_in(z), dim=1) # 2D -> D
        z = self.depthwise(z)                  # 每个通道独立做时间卷积
        z = F.silu(self.batch_norm(z))
        z = self.pointwise_out(z).transpose(1, 2)
        return self.dropout(z)
"""
        ),
        md(
            r"""
## 2. 组装一个 Conformer block

注意 `batch_first=True` 后，注意力输入是 `[B,T,D]`。如果漏掉它，代码可能不会立刻报错，却会把 batch 当成时间维，这是危险的静默错误。
"""
        ),
        code(
            """
class ConformerBlock(nn.Module):
    def __init__(self, d_model=32, num_heads=4, kernel_size=7, dropout=0.0):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.conv = ConvModule(d_model, kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None, attn_mask=None):
        x = x + 0.5 * self.ffn1(x)
        z = self.attn_norm(x)
        z, _ = self.attn(
            z, z, z,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + z
        x = x + self.conv(x)
        x = self.final_norm(x + 0.5 * self.ffn2(x))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x


B, T, D = 2, 40, 32
x = torch.randn(B, T, D)
lengths = torch.tensor([40, 27])
mask = lengths_to_padding_mask(lengths, T)
block = ConformerBlock(d_model=D)
block.eval()

with torch.no_grad():
    y = block(x, padding_mask=mask)

print("input:", tuple(x.shape), "output:", tuple(y.shape))
print("padding 最大绝对值:", y[1, 27:].abs().max().item())
assert y.shape == x.shape
assert torch.count_nonzero(y[1, 27:]) == 0
print("断言通过：shape 保持不变，padding 输出为 0。")
"""
        ),
        md(
            r"""
## 3. 全上下文为什么不能直接流式？

离线 self-attention 中，第 5 帧可以读取第 50 帧。流式系统在第 5 帧到来时尚未收到第 50 帧，因此不能这样算。

下面只隔离注意力模块做实验：保留前半段不变，只大幅修改未来。如果过去的输出随之改变，就证明模型读取了未来。
"""
        ),
        code(
            """
attn = nn.MultiheadAttention(16, 4, batch_first=True, dropout=0.0).eval()
base = torch.randn(1, 12, 16)
future_changed = base.clone()
future_changed[:, 6:] += 20.0

causal_mask = torch.triu(torch.ones(12, 12, dtype=torch.bool), diagonal=1)

with torch.no_grad():
    full_a, _ = attn(base, base, base, need_weights=False)
    full_b, _ = attn(future_changed, future_changed, future_changed, need_weights=False)
    causal_a, _ = attn(base, base, base, attn_mask=causal_mask, need_weights=False)
    causal_b, _ = attn(
        future_changed, future_changed, future_changed,
        attn_mask=causal_mask, need_weights=False,
    )

full_past_change = (full_a[:, :6] - full_b[:, :6]).abs().max().item()
causal_past_change = (causal_a[:, :6] - causal_b[:, :6]).abs().max().item()
print(f"全上下文：未来改变导致过去最大变化 {full_past_change:.6f}")
print(f"因果注意力：未来改变导致过去最大变化 {causal_past_change:.6f}")
assert full_past_change > 1e-3
assert causal_past_change < 1e-5
print("断言通过：因果 mask 阻止注意力读取未来。")
"""
        ),
        md(
            r"""
### 重要边界

给注意力加 causal mask **还不等于整个 Conformer 已经流式化**。上面的 `ConvModule` 使用左右对称 padding，仍会读取未来帧。真正的流式实现还需要：

1. 卷积只做左 padding，并跨 chunk 保存最近 `kernel_size-1` 帧；
2. 注意力限制左/右上下文，并缓存过去的 K/V；
3. 位置编码在 chunk 边界保持连续；
4. padding mask、cache 长度和下采样后的时间坐标一致；
5. 用“整段推理 vs 任意切块推理”一致性测试验收。
"""
        ),
        md(
            r"""
## 4. 接上 CTC 头

Conformer 是编码器，不直接规定训练目标。最简单的组合是在每个时间步接线性层，输出 `vocab_size + 1` 类 logits，再使用 CTC loss。这里用合成数据验证 shape 与梯度通路，不冒充真实识别准确率。
"""
        ),
        code(
            """
class TinyConformerCTC(nn.Module):
    def __init__(self, feat_dim=24, d_model=32, vocab_size=8):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.encoder = nn.ModuleList([
            ConformerBlock(d_model, num_heads=4, kernel_size=7)
            for _ in range(2)
        ])
        self.ctc_head = nn.Linear(d_model, vocab_size + 1)  # 0 是 blank

    def forward(self, features, lengths):
        mask = lengths_to_padding_mask(lengths, features.size(1))
        x = self.input_proj(features)
        for layer in self.encoder:
            x = layer(x, padding_mask=mask)
        return self.ctc_head(x)


model = TinyConformerCTC()
features = torch.randn(2, 30, 24)
input_lengths = torch.tensor([30, 24], dtype=torch.long)
targets = torch.tensor([1, 2, 3, 4, 2, 5, 6], dtype=torch.long)
target_lengths = torch.tensor([4, 3], dtype=torch.long)

logits = model(features, input_lengths)              # [B,T,C]
log_probs = logits.log_softmax(-1).transpose(0, 1)   # CTCLoss 要 [T,B,C]
loss = nn.CTCLoss(blank=0, zero_infinity=True)(
    log_probs, targets, input_lengths, target_lengths
)
loss.backward()

grad_norm = model.input_proj.weight.grad.norm().item()
print("logits:", tuple(logits.shape))
print(f"CTC loss={loss.item():.4f}, input_proj grad norm={grad_norm:.4f}")
assert logits.shape == (2, 30, 9)
assert math.isfinite(loss.item()) and grad_norm > 0
print("断言通过：Conformer → CTC 的前向和反向链路完整。")
"""
        ),
        md(
            r"""
## 5. 分层练习

### A. 回忆与解释（每题 1 分）

1. 卷积和注意力分别擅长哪种时间尺度？
2. 为什么 Conformer 中有两个乘 $1/2$ 的 FFN？
3. `key_padding_mask=True` 在本课代码中表示有效还是无效？
4. Conformer 和 CTC 是同一层面的概念吗？

### B. 预测与计算（每题 2 分）

5. `[B,T,D]=[3,120,64]` 通过不下采样 block 后 shape 是什么？
6. kernel size 为 15 的对称卷积，单层每个位置最多读取左右各多少帧？
7. 如果前端每帧步长 10 ms、总下采样 4 倍，编码器 100 个时间步覆盖约多少秒？
8. 为什么 `target_length > input_length` 会使 CTC 无法对齐？

### C. 编程与排错（每题 3 分）

9. 删除 block 末尾的 `masked_fill`，观察 padding 输出并解释结果。
10. 故意去掉 `batch_first=True`，记录 shape 或语义错误。
11. 把 `kernel_size` 改为 15，统计参数量是否线性增加。
12. 从空白 cell 重写 `lengths_to_padding_mask` 和 block 的残差顺序。

满分 24；达到 19 分且能从空白画出结构，再进入 RNN-T/TDT。
"""
        ),
        md(
            r"""
## 离场小测（闭卷发给老师）

1. 用不超过 80 字解释 Conformer 为什么适合语音。
2. 写出输入 `[B,T,F]` 到 CTC logits 的 shape 变化。
3. 为什么“causal attention”不等于“流式 Conformer”？
4. 画出 FFN → MHSA → Conv → FFN 的残差结构。

请同时写出你最没有把握的一题。老师会依据错误类型决定补讲还是进入第 43 课。
"""
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
            "course": {
                "stage": "前沿声学模型",
                "lesson": 42,
                "version": 1,
                "evidence_level": 3,
            },
        }
    )
    return nb


def lesson_43() -> nbf.NotebookNode:
    cells = [
        md(
            r"""
# 第 43 课：RNN-T 与 TDT——二维对齐和跳帧

这一课从你已掌握的 CTC 出发，回答三个问题：RNN-T 为什么比 CTC 多一个维度？它怎样在不看未来的情况下利用已输出文字？TDT 为什么可以减少大量 blank 解码步骤？

前置：第 10～14 课与第 42 课。CPU 约需 1～2 分钟。
"""
        ),
        md(
            r"""
## 完成标准

完成后你应能：

1. 画出 RNN-T 的 encoder、predictor、joiner；
2. 解释 logits `[B,T,U+1,V]` 的每一维；
3. 在二维 lattice 上区分 blank 边和 label 边；
4. 从空白写出 log-space forward recurrence；
5. 解释 TDT 的 token head 与 duration head 怎样减少解码步数。

本课实现教学版单样本 RNN-T loss，用暴力枚举验证数值与梯度。生产训练应使用经过优化和测试的 RNNT loss kernel。
"""
        ),
        md(
            r"""
## 课前诊断（先回答）

1. CTC 的每一帧预测是否依赖此前已经输出的 token？
2. 若音频有 `T=100` 个编码帧、转录有 `U=12` 个 token，RNN-T lattice 有哪两个坐标？
3. 流式模型为什么不能依赖完整句子的未来帧？
"""
        ),
        code(
            """
import itertools
import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(13)
np.random.seed(13)
random.seed(13)
torch.set_num_threads(2)
print("torch:", torch.__version__)
"""
        ),
        md(
            r"""
## 1. CTC 与 RNN-T 的根本差别

CTC 在时间步 $t$ 只根据声学编码器表示预测：

$$P(k\mid h_t)$$

RNN-T 另外使用 predictor 表示此前已经输出的 token 历史：

$$P(k\mid h_t, g_u)$$

```text
音频特征 → Encoder → h[t] ─┐
                             ├→ Joiner → token/blank
历史 token → Predictor → g[u]┘
```

- `t`：已经处理到哪个声学时间步；
- `u`：已经输出了多少个目标 token；
- blank：时间前进一步 `(t,u) → (t+1,u)`；
- 下一个正确 label：输出 token，但时间不前进 `(t,u) → (t,u+1)`。

所以 joint logits 是 `[B,T,U+1,V]`，而不是 CTC 的 `[B,T,V]`。
"""
        ),
        code(
            """
class TinyRNNT(nn.Module):
    def __init__(self, feat_dim=12, vocab_size=6, hidden=16, blank=0):
        super().__init__()
        self.blank = blank
        self.vocab_size = vocab_size
        self.encoder = nn.GRU(feat_dim, hidden, batch_first=True)
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.predictor = nn.GRU(hidden, hidden, batch_first=True)
        self.enc_proj = nn.Linear(hidden, hidden)
        self.pred_proj = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, vocab_size)

    def encode(self, features):
        return self.encoder(features)[0]

    def predict(self, targets):
        # 在目标前加 blank 作为教学版 BOS，得到 U+1 个 predictor 状态。
        bos = torch.full(
            (targets.size(0), 1), self.blank,
            dtype=targets.dtype, device=targets.device,
        )
        tokens_in = torch.cat([bos, targets], dim=1)
        embedded = self.embedding(tokens_in)
        return self.predictor(embedded)[0]

    def join(self, enc, pred):
        # [B,T,1,H] + [B,1,U+1,H] → [B,T,U+1,V]
        joint = torch.tanh(
            self.enc_proj(enc).unsqueeze(2)
            + self.pred_proj(pred).unsqueeze(1)
        )
        return self.output(joint)

    def forward(self, features, targets):
        return self.join(self.encode(features), self.predict(targets))


model = TinyRNNT()
features = torch.randn(2, 7, 12)
targets = torch.tensor([[1, 2, 3], [3, 2, 1]])
logits = model(features, targets)
print("features:", tuple(features.shape))
print("targets:", tuple(targets.shape))
print("joint logits:", tuple(logits.shape))
assert logits.shape == (2, 7, 4, 6)
print("断言通过：U 个目标需要 U+1 个 predictor 状态。")
"""
        ),
        md(
            r"""
## 2. 二维 forward 动态规划

令 $\alpha(t,u)$ 表示到达 lattice 状态 `(t,u)` 的所有路径概率之和（在 log-space 中）。两种转移为：

$$
\alpha(t+1,u)\;\mathrel{\oplus}=\;\alpha(t,u)+\log p(\text{blank}\mid t,u)
$$

$$
\alpha(t,u+1)\;\mathrel{\oplus}=\;\alpha(t,u)+\log p(y_u\mid t,u)
$$

$\oplus$ 是 `logaddexp`，对应普通概率空间的加法。最终 loss 为 $-\alpha(T,U)$。
"""
        ),
        code(
            """
def rnnt_loss_single(log_probs, targets, blank=0):
    \"\"\"教学版 RNN-T forward loss。

    log_probs: [T,U+1,V]
    targets: [U]
    \"\"\"
    T, U1, _ = log_probs.shape
    U = targets.numel()
    assert U1 == U + 1

    neg_inf = log_probs.new_tensor(float("-inf"))
    alpha = [[neg_inf for _ in range(U + 1)] for _ in range(T + 1)]
    alpha[0][0] = log_probs.new_zeros(())

    for t in range(T + 1):
        for u in range(U + 1):
            current = alpha[t][u]
            if t < T:  # blank 消耗一个声学帧
                candidate = current + log_probs[t, u, blank]
                alpha[t + 1][u] = torch.logaddexp(alpha[t + 1][u], candidate)
            if t < T and u < U:  # label 消耗一个目标 token
                label = int(targets[u])
                candidate = current + log_probs[t, u, label]
                alpha[t][u + 1] = torch.logaddexp(alpha[t][u + 1], candidate)

    return -alpha[T][U], alpha


tiny_logits = torch.randn(2, 3, 4, requires_grad=True)  # T=2,U=2,V=4
tiny_targets = torch.tensor([1, 2])
tiny_log_probs = tiny_logits.log_softmax(-1)
loss, alpha = rnnt_loss_single(tiny_log_probs, tiny_targets)
loss.backward()

print(f"loss={loss.item():.6f}")
print("alpha(T,U)=", alpha[2][2].item())
print("gradient norm=", tiny_logits.grad.norm().item())
assert math.isfinite(loss.item()) and tiny_logits.grad.norm() > 0
print("断言通过：forward loss 有限且能反向传播。")
"""
        ),
        md(
            r"""
### 为什么必须把所有合法路径相加？

同一个转录可能对应许多对齐。例如目标 `[A,B]`、`T=2` 时，可以先输出 A，也可以先 blank 再输出 A。训练数据没有逐帧对齐标注，RNN-T loss 必须边缘化所有合法路径。

下面用递归暴力枚举这个极小例子，并与动态规划比较。真实输入的路径数会爆炸，因此生产训练绝不能暴力枚举。
"""
        ),
        code(
            """
def enumerate_path_logps(log_probs, targets, t=0, u=0, score=None):
    T, _, _ = log_probs.shape
    U = targets.numel()
    score = log_probs.new_zeros(()) if score is None else score
    if t == T and u == U:
        return [score]
    paths = []
    if t < T:
        paths += enumerate_path_logps(
            log_probs, targets, t + 1, u,
            score + log_probs[t, u, 0],
        )
    if t < T and u < U:
        paths += enumerate_path_logps(
            log_probs, targets, t, u + 1,
            score + log_probs[t, u, int(targets[u])],
        )
    return paths


with torch.no_grad():
    path_scores = enumerate_path_logps(tiny_log_probs.detach(), tiny_targets)
    brute_log_prob = torch.logsumexp(torch.stack(path_scores), dim=0)
    dp_log_prob = -rnnt_loss_single(tiny_log_probs.detach(), tiny_targets)[0]

print("合法路径数:", len(path_scores))
print(f"暴力枚举 log P={brute_log_prob.item():.7f}")
print(f"动态规划 log P={dp_log_prob.item():.7f}")
assert torch.allclose(brute_log_prob, dp_log_prob, atol=1e-6)
print("断言通过：动态规划与所有合法路径之和完全一致。")
"""
        ),
        md(
            r"""
## 3. Predictor 为什么让 RNN-T 比 CTC 更像“声学 + 语言”联合模型？

同一声学帧 `h[t]` 会和不同历史状态 `g[u]` 组合。例如已经输出“北”时，joiner 对“京”的评分可以不同于已经输出“背”时的评分。

但 predictor 不是一个可以随意替代的大语言模型：它通常规模较小，且只在 ASR 配对数据上学习。领域热词、专名和长上下文仍需专门设计与评估。
"""
        ),
        code(
            """
model.eval()
enc = model.encode(torch.randn(1, 5, 12))
history_a = torch.tensor([[1, 2]])
history_b = torch.tensor([[4, 5]])
pred_a = model.predict(history_a)
pred_b = model.predict(history_b)

with torch.no_grad():
    dist_a = model.join(enc[:, 2:3], pred_a[:, -1:]).softmax(-1).squeeze()
    dist_b = model.join(enc[:, 2:3], pred_b[:, -1:]).softmax(-1).squeeze()

print("相同声学帧，不同 token 历史的最大概率差:", (dist_a-dist_b).abs().max().item())
assert not torch.allclose(dist_a, dist_b)
print("断言通过：predictor 历史会改变 joiner 的输出分布。")
"""
        ),
        md(
            r"""
## 4. TDT：同时预测 token 与 duration

标准 Transducer 解码中，blank 通常只让时间前进一帧。长音频会产生大量 blank 步骤。Token-and-Duration Transducer（TDT）增加 duration 分布：

$$P(v,d\mid t,u)=P_T(v\mid t,u)P_D(d\mid t,u)$$

- token head 决定输出 blank 或文字 token；
- duration head 决定时间向前跳多少帧；
- 预测较大 duration 时可以一次跳过多帧，减少 joiner 调用。

下面不是完整 TDT loss，而是一个透明的解码复杂度实验。
"""
        ),
        code(
            """
def transducer_blank_steps(num_frames):
    # 最简化的标准 transducer：至少逐帧前进。
    return list(range(num_frames))


def tdt_duration_steps(num_frames, durations):
    visited = []
    t = 0
    i = 0
    while t < num_frames:
        visited.append(t)
        duration = max(1, int(durations[i % len(durations)]))
        t += duration
        i += 1
    return visited


T = 30
standard = transducer_blank_steps(T)
tdt = tdt_duration_steps(T, durations=[4, 3, 5])
print("标准逐帧访问次数:", len(standard), standard)
print("TDT 跳帧访问次数:", len(tdt), tdt)
print(f"本例 joiner 步数减少 {(1-len(tdt)/len(standard))*100:.1f}%")
assert tdt[0] == 0 and len(tdt) < len(standard)
print("断言通过：duration 预测可以跳过中间声学帧。")
"""
        ),
        md(
            r"""
### TDT 的边界

跳帧不是免费午餐：duration 预测错了可能跨过重要声学证据。训练需要正确归一化所有 token-duration 对齐，解码也要限制合法 duration。生产实现应使用 NeMo 等经过验证的 loss 和解码器，而不是本课的复杂度演示。
"""
        ),
        md(
            r"""
## 5. CTC、RNN-T、TDT 选型

| 目标 | 更自然的起点 | 原因 |
|---|---|---|
| 最简单训练与批量离线推理 | CTC | `[B,T,V]`，高度并行 |
| 实时、利用输出历史 | RNN-T | 原生流式，predictor 建模历史 |
| 高吞吐 Transducer | TDT | duration head 减少 blank/逐帧步骤 |

最终选择必须在自己的数据上比较 CER/WER、RTF、首字延迟、尾延迟、增量稳定性和内存，不能只引用论文中的单一数字。
"""
        ),
        md(
            r"""
## 分层练习（24 分）

### A. 回忆（每题 1 分）

1. RNN-T 的三个网络分别叫什么？
2. blank 边改变 `t` 还是 `u`？label 边呢？
3. 为什么 predictor 输入需要 BOS？
4. TDT 比 RNN-T 多预测什么？

### B. 推理（每题 2 分）

5. `T=80,U=10,V=5000` 时 joint logits 的 shape 是什么？
6. 为什么不能直接保存大 batch 的完整 `[B,T,U+1,V]`？
7. 目标有相邻重复 token 时，RNN-T 是否像 CTC 一样必须插 blank？解释。
8. duration 总预测偏大时，可能出现什么识别错误？

### C. 编程与排错（每题 3 分）

9. 把暴力枚举改为同时返回 B/L 路径字符串。
10. 用 `T=3,U=2` 再验证动态规划，观察路径数。
11. 故意把 label 边写成 `(t+1,u+1)`，说明它变成了什么限制。
12. 从空白重写 `rnnt_loss_single`，通过枚举一致性断言。

达到 19/24 且能解释 lattice，才进入自监督预训练。
"""
        ),
        md(
            r"""
## 离场小测（闭卷发给老师）

1. 用一句话说出 CTC 与 RNN-T 条件概率的差别。
2. 画一个 `T=3,U=2` 的 lattice，标出 blank 和 label 边。
3. 为什么 RNN-T 能流式，而普通双向 AED 通常不能？
4. TDT 的速度来自哪里？它会引入什么风险？

请附上本课所有断言是否通过，以及你最没有把握的一题。
"""
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
            "course": {
                "stage": "前沿声学模型",
                "lesson": 43,
                "version": 1,
                "evidence_level": 3,
            },
        }
    )
    return nb


def lesson_44() -> nbf.NotebookNode:
    cells = [
        md(
            r"""
# 第 44 课：自监督语音预训练——不给转录也能学声学表示

人工转录昂贵，但原始音频极多。自监督学习（SSL）先用音频本身构造训练目标，学到可迁移的声学表示，再用较少转录数据微调 ASR。

本课用“遮住若干声学帧、根据上下文恢复离散单元”的 CPU 小实验贯通 wav2vec 2.0 / HuBERT 类方法的核心逻辑。
"""
        ),
        md(
            r"""
## 完成标准

1. 区分自监督预训练和有标签 ASR 微调；
2. 解释 feature encoder、mask、context encoder、target/codebook；
3. 说明为什么 loss 只能在被 mask 的位置计算；
4. 亲手训练一个 masked acoustic model，使遮挡位置准确率显著高于随机猜测；
5. 把预训练 encoder 接到 CTC head，验证梯度链路。

重要边界：本课使用合成“声学单元”，用于验证机制，不代表真实语音效果。
"""
        ),
        md(
            r"""
## 课前诊断

1. 没有文字转录时，神经网络还能从音频中获得什么监督信号？
2. 如果目标帧没有被遮住，模型可能采用什么投机办法？
3. 为什么预训练好以后仍需要少量带文字的数据？
"""
        ),
        code(
            """
import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(21)
np.random.seed(21)
random.seed(21)
torch.set_num_threads(2)
device = torch.device("cpu")
print("torch:", torch.__version__, "device:", device)
"""
        ),
        md(
            r"""
## 1. 从监督 ASR 到自监督预训练

监督 ASR 需要 `(音频, 转录)`：

```text
音频 → Encoder → CTC/RNN-T/LLM → “今天天气很好”
```

自监督预训练只需要音频：

```text
音频 → 局部声学特征 → 随机遮挡 → Context Encoder
                              ↓
                    预测被遮挡位置的离散声学目标
```

- wav2vec 2.0：量化潜在表示作为目标，以对比学习区分正确目标与负样本；
- HuBERT：先聚类得到伪标签，再做 masked classification，并可迭代聚类；
- WavLM：在此基础上强调噪声、说话人和重叠语音等鲁棒性。

它们的共同点不是某个 API，而是：**把输入的一部分隐藏，迫使模型用上下文学习语音结构。**
"""
        ),
        md(
            r"""
## 2. 构造透明的合成“语音”

我们用 8 个离散声学单元模拟音素。每个单元有一个特征原型，同一个单元连续保持 3 帧，再叠加噪声。这保留了语音最重要的教学性质之一：相邻帧具有强相关性。
"""
        ),
        code(
            """
NUM_CODES = 8
FEAT_DIM = 16
SEQ_LEN = 48
SEGMENT = 3

generator = torch.Generator().manual_seed(123)
prototypes = F.normalize(
    torch.randn(NUM_CODES, FEAT_DIM, generator=generator), dim=-1
)


def make_synthetic_audio(batch_size, generator=None):
    generator = generator or torch.default_generator
    num_segments = math.ceil(SEQ_LEN / SEGMENT)
    segment_codes = torch.randint(
        0, NUM_CODES, (batch_size, num_segments), generator=generator
    )
    codes = segment_codes.repeat_interleave(SEGMENT, dim=1)[:, :SEQ_LEN]
    noise = 0.12 * torch.randn(
        batch_size, SEQ_LEN, FEAT_DIM, generator=generator
    )
    features = prototypes[codes] + noise
    return features, codes


features, codes = make_synthetic_audio(2, torch.Generator().manual_seed(1))
print("features:", tuple(features.shape), "targets:", tuple(codes.shape))
print("第一条离散单元:", codes[0, :18].tolist())
assert features.shape == (2, 48, 16)
assert torch.equal(codes[0, 0::3], codes[0, 1::3])
print("断言通过：离散单元连续 3 帧，特征带有噪声。")
"""
        ),
        md(
            r"""
## 3. Masked Acoustic Model

输入投影后，把选中的位置替换成同一个可学习 `mask_embedding`。Transformer 必须根据未遮挡的邻居恢复目标 code。

如果把原始目标帧继续交给模型，它只需识别该帧原型，不必学习上下文，这就是信息泄漏。
"""
        ),
        code(
            """
def make_mask(batch_size, seq_len, probability=0.35, generator=None):
    generator = generator or torch.default_generator
    mask = torch.rand(batch_size, seq_len, generator=generator) < probability
    # 保证每条样本至少有一个监督位置。
    for b in range(batch_size):
        if not mask[b].any():
            mask[b, 0] = True
    return mask


class MaskedAcousticModel(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, d_model=32, num_codes=NUM_CODES):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.mask_embedding = nn.Parameter(torch.zeros(d_model))
        self.position = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.01)
        # 语音有很强的局部连续性。先加入局部卷积偏置，再由
        # Transformer 汇总更长上下文；这与 Conformer 的动机一致。
        self.local_context = nn.Conv1d(
            d_model, d_model, kernel_size=5, padding=2, groups=d_model
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=96,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(d_model, num_codes)

    def encode(self, features, mask=None):
        x = self.input_proj(features)
        if mask is not None:
            replacement = self.mask_embedding.view(1, 1, -1)
            x = torch.where(mask.unsqueeze(-1), replacement, x)
        x = x + self.position[:, :x.size(1)]
        x = x + self.local_context(x.transpose(1, 2)).transpose(1, 2)
        return self.context_encoder(x)

    def forward(self, features, mask=None):
        return self.classifier(self.encode(features, mask))


ssl_model = MaskedAcousticModel().to(device)
mask = make_mask(2, SEQ_LEN, generator=torch.Generator().manual_seed(2))
ssl_logits = ssl_model(features, mask)
print("mask 比例:", mask.float().mean().item())
print("logits:", tuple(ssl_logits.shape))
assert ssl_logits.shape == (2, SEQ_LEN, NUM_CODES)
print("断言通过：每个时间步预测一个离散声学目标。")
"""
        ),
        md(
            r"""
## 4. 只在 mask 位置训练

损失为：

$$
\mathcal L=-\frac{1}{|M|}\sum_{t\in M}\log P(c_t\mid \tilde{x})
$$

$M$ 是被遮挡位置，$\tilde{x}$ 是遮挡后的输入。未遮挡帧提供上下文，但不计入本课的预测 loss。
"""
        ),
        code(
            """
def masked_loss_and_accuracy(logits, targets, mask):
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    loss = F.cross_entropy(selected_logits, selected_targets)
    accuracy = (selected_logits.argmax(-1) == selected_targets).float().mean()
    return loss, accuracy


optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=3e-3)
train_generator = torch.Generator().manual_seed(300)
history = []

ssl_model.train()
for step in range(121):
    batch_x, batch_codes = make_synthetic_audio(32, train_generator)
    batch_mask = make_mask(32, SEQ_LEN, 0.35, train_generator)
    logits = ssl_model(batch_x, batch_mask)
    loss, acc = masked_loss_and_accuracy(logits, batch_codes, batch_mask)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), 5.0)
    optimizer.step()

    if step % 20 == 0:
        history.append((step, loss.item(), acc.item()))

for step, loss_value, acc_value in history:
    print(f"step={step:3d} loss={loss_value:.4f} masked_acc={acc_value:.3f}")

eval_generator = torch.Generator().manual_seed(999)
eval_x, eval_codes = make_synthetic_audio(128, eval_generator)
eval_mask = make_mask(128, SEQ_LEN, 0.35, eval_generator)
ssl_model.eval()
with torch.no_grad():
    eval_loss, eval_acc = masked_loss_and_accuracy(
        ssl_model(eval_x, eval_mask), eval_codes, eval_mask
    )

chance = 1 / NUM_CODES
print(f"评估 masked accuracy={eval_acc.item():.3f}; 随机猜测={chance:.3f}")
assert eval_acc > chance + 0.25
print("断言通过：模型确实从上下文恢复了大量被遮挡单元。")
"""
        ),
        md(
            r"""
### 你应该观察什么？

训练准确率不会必然单调，因为每一步的音频和 mask 都在变化。真正有意义的是固定随机种子的独立评估集，并与 `1/NUM_CODES` 的随机基线比较。

本例相邻 3 帧常属于同一个单元，所以模型可利用邻居。真实语音更复杂：目标可能来自量化器、K-means 聚类或教师模型，还需要处理说话人、噪声、音高和通道等变化。
"""
        ),
        md(
            r"""
## 5. 把预训练 encoder 接到 CTC

预训练阶段预测的是声学 code；ASR 微调阶段把 classifier 换成文字词表，并用 CTC/RNN-T/AED 等目标训练。Encoder 可以全部微调，也可以先冻结再逐步解冻。
"""
        ),
        code(
            """
class SSLCTCModel(nn.Module):
    def __init__(self, pretrained, vocab_size=10):
        super().__init__()
        self.pretrained = pretrained
        self.ctc_head = nn.Linear(32, vocab_size + 1)  # 0=blank

    def forward(self, x):
        hidden = self.pretrained.encode(x, mask=None)
        return self.ctc_head(hidden)


ctc_model = SSLCTCModel(ssl_model)
asr_x, _ = make_synthetic_audio(2, torch.Generator().manual_seed(44))
asr_targets = torch.tensor([1, 2, 3, 4, 2, 5, 6], dtype=torch.long)
input_lengths = torch.tensor([48, 48], dtype=torch.long)
target_lengths = torch.tensor([4, 3], dtype=torch.long)

ctc_logits = ctc_model(asr_x)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)(
    ctc_logits.log_softmax(-1).transpose(0, 1),
    asr_targets,
    input_lengths,
    target_lengths,
)
ctc_model.zero_grad()
ctc_loss.backward()
encoder_grad = ctc_model.pretrained.input_proj.weight.grad.norm().item()
head_grad = ctc_model.ctc_head.weight.grad.norm().item()

print("CTC logits:", tuple(ctc_logits.shape))
print(f"loss={ctc_loss.item():.4f}, encoder_grad={encoder_grad:.4f}, head_grad={head_grad:.4f}")
assert ctc_logits.shape == (2, 48, 11)
assert encoder_grad > 0 and head_grad > 0
print("断言通过：预训练 encoder 与新的 CTC head 可以联合微调。")
"""
        ),
        md(
            r"""
## 6. 从教学实验到真实系统

真实预训练通常需要：

1. 数千到数百万小时经许可的多域音频；
2. 稳定的分布式训练、混合精度与 checkpoint；
3. 防止静音、重复、音乐、隐私数据和伪标签错误污染；
4. 下游 speaker-disjoint ASR 测试；
5. 比较“随机初始化”和“相同微调预算下的 SSL 初始化”，才可归因于预训练。

不要用“预训练 loss 下降”替代 CER/WER，也不要只在训练说话人上测试。
"""
        ),
        md(
            r"""
## 分层练习（24 分）

### A. 回忆（每题 1 分）

1. 自监督的“监督”来自哪里？
2. wav2vec 2.0 与 HuBERT 的目标构造有何不同？
3. 为什么要 mask 连续 span，而不只是单个点？
4. 预训练 classifier 为什么通常不能直接当文字输出层？

### B. 推理（每题 2 分）

5. `NUM_CODES=100` 时随机准确率是多少？
6. mask 比例为 0 会发生什么？比例为 1 又会怎样？
7. 为什么随机切分帧会造成 train/test 泄漏？
8. 何时应冻结 encoder，何时应全量微调？

### C. 编程（每题 3 分）

9. 把单点 mask 改为长度 3 的连续 span。
10. 比较带/不带位置编码的 masked accuracy。
11. 把 `SEGMENT` 从 3 改成 1，预测结果并验证。
12. 从空白实现 masked loss，保证只选择 `mask=True` 的位置。

达到 19/24，且能解释信息泄漏，才进入迷你音频语言模型。
"""
        ),
        md(
            r"""
## 离场小测（闭卷发给老师）

1. 为什么无文字音频仍能训练声学 encoder？
2. 本课为何只在 mask 位置算 loss？
3. 自监督预训练如何迁移到 CTC？
4. 写出至少两个会让 SSL 评估虚高的数据泄漏方式。

附上你的最终 masked accuracy 和最不确定的一题。
"""
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
            "course": {
                "stage": "前沿声学模型",
                "lesson": 44,
                "version": 1,
                "evidence_level": 3,
            },
        }
    )
    return nb


def lesson_45() -> nbf.NotebookNode:
    cells = [
        md(
            r"""
# 第 45 课：迷你音频语言模型——Audio Encoder 如何接入 LLM

现代 Large Audio-Language Model（LALM）不再把语音只看作逐帧分类问题，而是把连续音频表示接入语言模型，让语言模型自回归生成转录。

本课在 CPU 上训练一个完整的迷你系统：`Audio Encoder → Projector → Decoder-only LM → 文本 token`，并实现 teacher forcing 与 greedy generation。
"""
        ),
        md(
            r"""
## 完成标准

1. 解释 Audio Encoder、Projector、LLM 各自职责；
2. 写出音频帧率下采样前后的 shape；
3. 解释为什么音频 prefix 不计算 next-token loss；
4. 训练迷你 LALM 并用未见样本自回归生成；
5. 说明生成式 ASR 的上下文优势与幻觉风险。

本课是结构复现，不是 Qwen3-ASR 等基础模型的规模复现。
"""
        ),
        md(
            r"""
## 课前诊断

1. 如果声学 encoder 每秒输出 100 帧，而 LLM 每秒只接受约 10 个音频 embedding，需要哪个模块缩短序列？
2. LLM 生成第 5 个文字 token 时可以依赖什么？
3. 为什么语言能力越强，ASR 不一定越忠于原始声音？
"""
        ),
        code(
            """
import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(31)
np.random.seed(31)
random.seed(31)
torch.set_num_threads(2)
device = torch.device("cpu")
print("torch:", torch.__version__, "device:", device)
"""
        ),
        md(
            r"""
## 1. 架构与 shape

```text
音频 [B,T,F]
  → Audio Encoder（局部特征 + 下采样）
  → speech embeddings [B,T',D_a]
  → Projector
  → audio prefix [B,T',D_lm]
  → 与 [BOS, y1, y2, ...] 的文字 embedding 拼接
  → Decoder-only Transformer
  → next-token logits
```

Projector 不只是“改 shape”。实际系统中，它还要完成两种表示空间的对齐，并控制音频序列长度，否则 LLM 的注意力成本会过高。

Qwen3-ASR 的公开报告采用 AuT encoder、8 倍下采样、projector 和 Qwen3 LLM；本课用更小的卷积 encoder 和 Transformer 复现相同接口思想。
"""
        ),
        md(
            r"""
## 2. 合成语音—文字对

设 8 个内容 token，每个 token 对应一个带噪声声学原型并持续 3 帧。文字词表另外包含 `PAD/BOS/EOS`。

与第 44 课不同，本课训练目标是文字 next-token，而不是离散声学 code。
"""
        ),
        code(
            """
PAD, BOS, EOS = 0, 1, 2
NUM_CONTENT = 8
VOCAB_SIZE = 3 + NUM_CONTENT
FEAT_DIM = 16
TOKENS_PER_UTT = 6
FRAMES_PER_TOKEN = 3
AUDIO_LEN = TOKENS_PER_UTT * FRAMES_PER_TOKEN

data_generator = torch.Generator().manual_seed(2026)
audio_prototypes = F.normalize(
    torch.randn(NUM_CONTENT, FEAT_DIM, generator=data_generator), dim=-1
)


def make_audio_text_batch(batch_size, generator=None, noise_std=0.10):
    generator = generator or torch.default_generator
    content = torch.randint(
        0, NUM_CONTENT, (batch_size, TOKENS_PER_UTT), generator=generator
    )
    text = content + 3
    frame_codes = content.repeat_interleave(FRAMES_PER_TOKEN, dim=1)
    noise = noise_std * torch.randn(
        batch_size, AUDIO_LEN, FEAT_DIM, generator=generator
    )
    audio = audio_prototypes[frame_codes] + noise
    return audio, text


audio, text = make_audio_text_batch(2, torch.Generator().manual_seed(5))
print("audio:", tuple(audio.shape), "text:", tuple(text.shape))
print("sample text ids:", text[0].tolist())
assert audio.shape == (2, 18, 16) and text.shape == (2, 6)
print("断言通过：每个文字 token 对应 3 个带噪声声学帧。")
"""
        ),
        md(
            r"""
## 3. 搭建迷你 LALM

卷积 `kernel_size=stride=3` 把 18 帧压到 6 个 speech embedding。真实系统通常使用多层卷积、Conformer/Transformer 和更复杂的下采样。

语言模型输入序列为：

```text
[audio_1 ... audio_6] [BOS, y1, y2, ... y6]
```

文字位置的监督目标为：

```text
[y1, y2, ... y6, EOS]
```

因此只在最后 7 个文字位置计算交叉熵；音频 prefix 提供条件，不要求它预测文字。
"""
        ),
        code(
            """
class MiniAudioEncoder(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, audio_dim=32):
        super().__init__()
        self.subsample = nn.Conv1d(
            feat_dim, audio_dim,
            kernel_size=FRAMES_PER_TOKEN,
            stride=FRAMES_PER_TOKEN,
        )
        self.norm = nn.LayerNorm(audio_dim)

    def forward(self, audio):
        x = self.subsample(audio.transpose(1, 2)).transpose(1, 2)
        return self.norm(F.silu(x))


class MiniAudioLanguageModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, audio_dim=32, d_model=48):
        super().__init__()
        self.audio_encoder = MiniAudioEncoder(audio_dim=audio_dim)
        self.projector = nn.Sequential(
            nn.Linear(audio_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Parameter(torch.randn(1, 32, d_model) * 0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.lm = nn.TransformerEncoder(layer, num_layers=2)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def audio_prefix(self, audio):
        return self.projector(self.audio_encoder(audio))

    def forward(self, audio, text_input):
        prefix = self.audio_prefix(audio)
        text_emb = self.token_embedding(text_input)
        x = torch.cat([prefix, text_emb], dim=1)
        x = x + self.position[:, :x.size(1)]
        causal = torch.triu(
            torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        hidden = self.lm(x, mask=causal, is_causal=True)
        return self.lm_head(hidden), prefix.size(1)


lal_model = MiniAudioLanguageModel().to(device)
text_input = torch.cat([
    torch.full((2, 1), BOS, dtype=torch.long), text
], dim=1)
all_logits, prefix_len = lal_model(audio, text_input)
text_logits = all_logits[:, prefix_len:]
print("audio prefix length:", prefix_len)
print("all logits:", tuple(all_logits.shape))
print("supervised text logits:", tuple(text_logits.shape))
assert prefix_len == 6
assert text_logits.shape == (2, 7, VOCAB_SIZE)
print("断言通过：18 个声学帧 → 6 个 prefix；7 个文字位置接受监督。")
"""
        ),
        md(
            r"""
## 4. Teacher forcing 训练

训练时将正确历史 `[BOS,y1,...]` 输入模型，这称为 teacher forcing。模型学习在音频和正确文字前缀条件下预测下一个 token。

推理时没有正确未来文字，只能把自己的输出逐步送回模型，因此错误可能累积。
"""
        ),
        code(
            """
def lal_loss(model, audio, target_text):
    bos = torch.full(
        (target_text.size(0), 1), BOS,
        dtype=torch.long, device=target_text.device,
    )
    text_input = torch.cat([bos, target_text], dim=1)
    text_target = torch.cat([
        target_text,
        torch.full_like(bos, EOS),
    ], dim=1)
    logits, prefix_len = model(audio, text_input)
    supervised_logits = logits[:, prefix_len:]
    loss = F.cross_entropy(
        supervised_logits.reshape(-1, VOCAB_SIZE),
        text_target.reshape(-1),
    )
    accuracy = (
        supervised_logits.argmax(-1) == text_target
    ).float().mean()
    return loss, accuracy


optimizer = torch.optim.AdamW(lal_model.parameters(), lr=3e-3)
train_gen = torch.Generator().manual_seed(77)
history = []
lal_model.train()
for step in range(181):
    batch_audio, batch_text = make_audio_text_batch(48, train_gen)
    loss, accuracy = lal_loss(lal_model, batch_audio, batch_text)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lal_model.parameters(), 5.0)
    optimizer.step()
    if step % 30 == 0:
        history.append((step, loss.item(), accuracy.item()))

for step, loss_value, acc_value in history:
    print(f"step={step:3d} loss={loss_value:.4f} teacher_forced_acc={acc_value:.3f}")

assert history[-1][1] < history[0][1] * 0.35
print("断言通过：next-token loss 显著下降。")
"""
        ),
        md(
            r"""
## 5. 真正的自回归生成

下面不提供正确转录。模型从 `[BOS]` 开始，每次取最后一个位置的最大概率 token，追加后重新运行，直到生成 `EOS` 或达到上限。
"""
        ),
        code(
            """
@torch.no_grad()
def greedy_generate(model, audio, max_new_tokens=10):
    model.eval()
    history = torch.full(
        (audio.size(0), 1), BOS, dtype=torch.long, device=audio.device
    )
    finished = torch.zeros(audio.size(0), dtype=torch.bool, device=audio.device)
    outputs = [[] for _ in range(audio.size(0))]

    for _ in range(max_new_tokens):
        logits, _ = model(audio, history)
        next_token = logits[:, -1].argmax(-1)
        history = torch.cat([history, next_token.unsqueeze(1)], dim=1)
        for b, token in enumerate(next_token.tolist()):
            if not finished[b]:
                if token == EOS:
                    finished[b] = True
                else:
                    outputs[b].append(token)
        if finished.all():
            break
    return outputs


test_audio, test_text = make_audio_text_batch(
    64, torch.Generator().manual_seed(12345)
)
generated = greedy_generate(lal_model, test_audio)
token_correct = 0
token_total = 0
exact = 0
for prediction, reference in zip(generated, test_text.tolist()):
    token_correct += sum(a == b for a, b in zip(prediction, reference))
    token_total += max(len(reference), len(prediction))
    exact += int(prediction == reference)

print("前 5 条未见样本：")
for i in range(5):
    print("reference=", test_text[i].tolist(), "generated=", generated[i])
print(f"token accuracy={token_correct/token_total:.3f}")
print(f"exact sequence accuracy={exact/len(generated):.3f}")
assert token_correct / token_total > 0.90
assert exact / len(generated) > 0.65
print("断言通过：模型在未见音频上完成了自回归转录。")
"""
        ),
        md(
            r"""
## 6. 为什么 LALM 强，也为什么会幻觉？

语言模型可以利用：

- 已生成文字、对话历史、领域词表和世界知识；
- 多语言共用表示；
- 指令或上下文中的专名提示。

但 next-token 目标优化的是“条件下最可能的文字”，不保证每个字都有充分声学证据。当输入是静音、强噪声或分布外声音时，强语言先验可能生成流畅但错误的内容。

生产系统至少要做：

1. 静音/非语音负样本训练；
2. 空转录能力与 VAD；
3. 数字、姓名、否定词的证据保护；
4. 噪声、截断、音乐和提示注入测试；
5. 保存原始转录、置信度和审计信息。
"""
        ),
        code(
            """
# 反事实测试：把音频替换为全零，观察模型是否仍生成内容。
silent_audio = torch.zeros_like(test_audio[:8])
silent_outputs = greedy_generate(lal_model, silent_audio)
print("全零输入的生成结果：")
for output in silent_outputs:
    print(output)

nonempty = sum(bool(x) for x in silent_outputs)
print(f"非空输出 {nonempty}/{len(silent_outputs)}")
print("这不是准确率测试，而是生成式 ASR 必须具备的静音安全测试。")
"""
        ),
        md(
            r"""
## 7. 从本课到 Qwen3-ASR

| 本课 | 大规模 LALM |
|---|---|
| 18 帧合成特征 | 真实 Fbank/波形与长音频 |
| 单层卷积 encoder | 数亿参数音频 encoder |
| 小 projector | 多层对齐与下采样模块 |
| 2 层 Transformer | 预训练 LLM |
| 180 步随机数据 | 海量预训练、SFT、RL |

搭建现实系统的合理方式不是从零复现数千万小时训练，而是：理解并验证本课接口 → 选择开放 checkpoint → 准备合法领域数据 → 参数高效或全量微调 → 做独立真实评测。
"""
        ),
        md(
            r"""
## 分层练习（24 分）

### A. 回忆（每题 1 分）

1. Audio Encoder、Projector、LLM 各做什么？
2. 为什么 projector 常伴随下采样？
3. teacher forcing 和推理的输入历史有何不同？
4. 本课哪些位置计算交叉熵？

### B. 推理（每题 2 分）

5. 输入 1,000 帧、下采样 8 倍，prefix 约多长？
6. prefix 长度翻倍时，普通全注意力计算量约变为多少倍？
7. 为什么静音也生成文字属于严重失败？
8. LALM 上下文 biasing 与传统 WFST 热词各有什么风险？

### C. 编程（每题 3 分）

9. 将 noise_std 提高到 0.5，重新评估生成准确率。
10. 删除 projector 的非线性层，比较收敛速度。
11. 增加 `NO_SPEECH` 样本，使静音输出 EOS。
12. 从空白实现 greedy generation，禁止读取参考文字。

达到 19/24 且通过“生成不读取 reference”的代码审查，再进入真实 Qwen3-ASR。
"""
        ),
        md(
            r"""
## 离场小测（闭卷发给老师）

1. 用自己的话解释“把音频作为 LLM prefix”。
2. 写出 `[B,T,F] → [B,T',D_lm] → token` 的 shape 链。
3. teacher forcing 准确率高为什么不能证明真实生成准确率高？
4. 列出三项生成式 ASR 的幻觉测试。

附上你的 token/exact accuracy、静音输出和最没有把握的一题。
"""
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
            "course": {
                "stage": "前沿声学模型",
                "lesson": 45,
                "version": 1,
                "evidence_level": 3,
            },
        }
    )
    return nb


def lesson_46() -> nbf.NotebookNode:
    cells = [
        md(
            r"""
# 第 46 课：Qwen3-ASR——真实前沿模型的推理、微调与验收

这一课把前 45 课连接到真实开放模型。你将完成硬件预检、官方数据格式、speaker-disjoint 切分、CER/WER、可选真实推理，以及一套不会把训练集过拟合冒充效果的微调流程。

当前项目环境是 CPU，因此所有数据与评测实验都可运行；模型下载和 GPU 推理默认关闭，避免意外下载数 GB 权重。
"""
        ),
        md(
            r"""
## 完成标准

1. 能画出 Qwen3-ASR 的 AuT encoder → projector → Qwen3；
2. 能选择 0.6B/1.7B、offline/streaming 后端；
3. 生成并验证官方 JSONL 格式，且 train/eval 说话人不重叠；
4. 从空白实现 CER，并用手算样例验收；
5. 写出 baseline → SFT → 独立评测 → 部署门禁的完整方案；
6. 有 GPU 时能启用可选单元完成真实转录。
"""
        ),
        md(
            r"""
## 1. 2026 架构快照

根据 Qwen3-ASR Technical Report：

```text
16 kHz 音频
→ 128 维 Fbank
→ AuT encoder
→ 8 倍下采样，约 12.5 Hz speech embedding
→ Projector
→ Qwen3-0.6B / Qwen3-1.7B
→ 自回归 ASR 文本
```

- 0.6B 版本：180M AuT encoder，hidden size 896，强调准确率—效率平衡；
- 1.7B 版本：300M AuT encoder，hidden size 1024，强调更高质量；
- 动态注意力窗 1～8 秒，用一个模型兼容 offline/streaming；
- 官方共支持 52 种语言和方言（30 种语言、22 种中文方言）；
- 训练经历 AuT 预训练、Qwen3-Omni 预训练、ASR SFT 与 GSPO RL。

资料：

- 技术报告：https://arxiv.org/abs/2601.21337
- 官方仓库：https://github.com/QwenLM/Qwen3-ASR
- 官方微调说明：https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning

厂商报告中的 SOTA、速度和内部集结果是选型线索，不是你自己场景的验收结论。
"""
        ),
        md(
            r"""
## 2. 硬件与后端预检

Transformers 后端适合先验证离线推理；官方仓库当前的流式推理使用 vLLM 后端。vLLM/FlashAttention 通常要求合适的 NVIDIA CUDA 环境，Windows 用户更适合 WSL2/Linux 或云 GPU。
"""
        ),
        code(
            """
import importlib.util
import json
import math
import platform
import re
import string
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path.cwd()
print("python:", sys.version.split()[0])
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("qwen_asr installed:", importlib.util.find_spec("qwen_asr") is not None)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("当前走 CPU 教学路径；真实 Qwen3-ASR 推理单元保持关闭。")
"""
        ),
        code(
            """
def prefix_budget(audio_seconds, output_hz=12.5):
    return math.ceil(audio_seconds * output_hz)


for seconds in [1, 10, 60, 1200]:
    print(f"{seconds:4d} 秒 → 约 {prefix_budget(seconds):5d} 个 AuT 输出时间步")

assert prefix_budget(10) == 125
print("断言通过：下采样降低序列长度，但 20 分钟仍是很长的注意力输入。")
"""
        ),
        md(
            r"""
## 3. 建立不会泄漏的数据集

官方 SFT JSONL 每行至少包含：

```json
{"audio":"/path/utt.wav","text":"language Chinese<asr_text>你好世界"}
```

有语言标签时使用 `language Chinese/English...`；没有时使用 `language None`。但工业数据还应在自己的元数据表保存 `speaker_id、device、domain、duration、license`，用于分组切分和分桶评测。

下面使用仓库内 FSDD 多说话人样例。切分单位必须是 speaker，而不是随机切音频，否则同一个人的音色会同时出现在训练和评估中。
"""
        ),
        code(
            """
DIGIT_TEXT = {0: "zero", 1: "one", 2: "two"}
audio_files = sorted((ROOT / "data" / "fsdd_multispeaker").glob("*.wav"))
records = []
for path in audio_files:
    match = re.fullmatch(r"(\\d+)_([^_]+)_\\d+\\.wav", path.name)
    if not match:
        continue
    digit = int(match.group(1))
    speaker = match.group(2)
    info = sf.info(path)
    records.append({
        "audio": str(path.resolve()),
        "text": f"language English<asr_text>{DIGIT_TEXT[digit]}",
        "plain_text": DIGIT_TEXT[digit],
        "speaker_id": speaker,
        "sample_rate": info.samplerate,
        "duration": info.frames / info.samplerate,
    })

speakers = sorted({r["speaker_id"] for r in records})
eval_speaker = speakers[-1]
train_records = [r for r in records if r["speaker_id"] != eval_speaker]
eval_records = [r for r in records if r["speaker_id"] == eval_speaker]
train_speakers = {r["speaker_id"] for r in train_records}
eval_speakers = {r["speaker_id"] for r in eval_records}

print("speakers:", speakers)
print("train/eval records:", len(train_records), len(eval_records))
print("train speakers:", train_speakers, "eval speakers:", eval_speakers)
print("官方 JSONL 行示例:")
print(json.dumps({k: train_records[0][k] for k in ["audio", "text"]}, ensure_ascii=False))

assert records and train_speakers.isdisjoint(eval_speakers)
assert all(Path(r["audio"]).is_file() for r in records)
assert all(r["sample_rate"] > 0 and r["duration"] > 0 for r in records)
print("断言通过：路径、音频属性有效，train/eval 说话人完全隔离。")
"""
        ),
        code(
            """
# 改成 True 才会写文件；已运行版本不会修改你的数据目录。
WRITE_MANIFESTS = False
if WRITE_MANIFESTS:
    manifest_dir = ROOT / "data" / "qwen3_asr_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, items in [("train", train_records), ("eval", eval_records)]:
        output = manifest_dir / f"{split}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for item in items:
                official = {"audio": item["audio"], "text": item["text"]}
                handle.write(json.dumps(official, ensure_ascii=False) + "\\n")
        print("wrote", output)
else:
    print("WRITE_MANIFESTS=False：只验证，不写文件。")
"""
        ),
        md(
            r"""
## 4. CER/WER：先把尺子做对

中文通常报告 CER，空格分词语言通常报告 WER：

$$\mathrm{ErrorRate}=\frac{S+D+I}{N}$$

`S/D/I` 分别是替换、删除、插入，`N` 是参考序列长度。文本规范化必须在实验开始前固定并版本化；随结果修改规则会污染比较。
"""
        ),
        code(
            """
def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            substitute = previous[j - 1] + (ref_item != hyp_item)
            delete = previous[j] + 1
            insert = current[j - 1] + 1
            current.append(min(substitute, delete, insert))
        previous = current
    return previous[-1]


def normalize_zh(text):
    # 教学规则：保留中文、字母和数字；生产规则必须覆盖数字/金额/日期等。
    return "".join(ch.lower() for ch in text if ch.isalnum())


def normalize_en(text):
    table = str.maketrans("", "", string.punctuation)
    return text.lower().translate(table).split()


def cer(reference, hypothesis):
    ref = list(normalize_zh(reference))
    hyp = list(normalize_zh(hypothesis))
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(reference, hypothesis):
    ref = normalize_en(reference)
    hyp = normalize_en(hypothesis)
    return edit_distance(ref, hyp) / max(1, len(ref))


print("CER:", cer("今天天气很好", "今天天气好"))
print("WER:", wer("we learn speech models", "we learn models"))
assert cer("今天天气很好", "今天天气好") == 1 / 6
assert wer("we learn speech models", "we learn models") == 1 / 4
assert cer("完全相同", "完全相同") == 0
print("断言通过：手算删除错误与实现一致。")
"""
        ),
        md(
            r"""
## 5. 可选：真实模型推理

建议为真实模型单独建立 Python 3.12 + CUDA 环境，不要破坏本课程 CPU 环境：

```powershell
uv venv .venv-qwen --python 3.12
.\.venv-qwen\Scripts\Activate.ps1
uv pip install -U qwen-asr
```

流式/vLLM 路线按照官方仓库安装 `qwen-asr[vllm]`，通常在 Linux/WSL2 或云 GPU 上进行。先用 0.6B 验证资源与数据契约，再决定是否换 1.7B。

下面的开关默认为 `False`。开启前确认已有 CUDA、足够显存/内存、模型下载许可与磁盘空间。
"""
        ),
        code(
            """
RUN_REAL_MODEL = False
MODEL_ID = "Qwen/Qwen3-ASR-0.6B"

if RUN_REAL_MODEL:
    if not torch.cuda.is_available():
        raise RuntimeError("真实模型实验需要可用 CUDA；请切换到 GPU 环境。")
    if importlib.util.find_spec("qwen_asr") is None:
        raise RuntimeError("请先在独立环境安装：uv pip install -U qwen-asr")

    from qwen_asr import Qwen3ASRModel

    real_model = Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=4,
        max_new_tokens=128,
    )
    sample_path = eval_records[0]["audio"]
    started = time.perf_counter()
    result = real_model.transcribe(audio=sample_path, language="English")[0]
    elapsed = time.perf_counter() - started
    duration = eval_records[0]["duration"]
    print("reference:", eval_records[0]["plain_text"])
    print("language:", result.language)
    print("prediction:", result.text)
    print(f"elapsed={elapsed:.3f}s RTF={elapsed/duration:.3f}")
else:
    print("RUN_REAL_MODEL=False：未下载权重，CPU 课程执行保持轻量。")
"""
        ),
        md(
            r"""
## 6. 官方 SFT 命令与正确实验顺序

官方微调脚本支持 JSONL 音频—文本对、单 GPU 和 `torchrun` 多 GPU。示例核心参数为：

```powershell
python qwen3_asr_sft.py `
  --model_path Qwen/Qwen3-ASR-0.6B `
  --train_file .\train.jsonl `
  --eval_file .\eval.jsonl `
  --output_dir .\qwen3-asr-sft-out `
  --batch_size 4 `
  --grad_acc 8 `
  --lr 2e-5 `
  --epochs 1 `
  --save_steps 200
```

课程将 batch 调小只是为了降低显存门槛，不代表最佳超参数。正确顺序是：

1. **冻结数据与规范**：许可、去重、speaker/domain split、文本规范版本；
2. **跑零样本 baseline**：保存逐条输出、CER/WER、RTF 和失败桶；
3. **小批量过拟合**：先证明数据格式、loss、checkpoint 和加载链路正确；
4. **正式 SFT**：只根据 validation 选 checkpoint，绝不看 test 调参；
5. **独立 test**：总体与方言/噪声/设备/长度/专名/数字分桶；
6. **回归门禁**：新领域提升不能用通用能力、静音幻觉或延迟恶化换取；
7. **部署验收**：并发、显存、首 token、尾延迟、崩溃恢复和版本回滚。
"""
        ),
        md(
            r"""
## 7. 一张最低验收表

| 维度 | 必测项 |
|---|---|
| 质量 | CER/WER、专名召回、数字准确率、漏段率 |
| 鲁棒性 | 静音、音乐、低 SNR、重叠说话、口音/方言、儿童/老人 |
| 忠实度 | 否定词、数字、姓名、提示注入、无语音幻觉 |
| 流式 | 首字延迟、partial 修改次数、endpoint 尾延迟 |
| 性能 | RTF、并发吞吐、p50/p95/p99、峰值显存 |
| 工程 | 模型/词表/规范版本、回滚、日志脱敏、授权与许可 |

“平均 CER 降低”不足以发布。例如金额识别恶化、静音产生文本或 p99 延迟翻倍，都可能阻止上线。
"""
        ),
        md(
            r"""
## 分层练习（24 分）

### A. 回忆（每题 1 分）

1. Qwen3-ASR 三个主要结构部件是什么？
2. 0.6B 和 1.7B 应怎样做第一次选型？
3. 官方 JSONL 的两个必需字段是什么？
4. CER 分母是什么？

### B. 推理（每题 2 分）

5. 60 秒音频在 12.5 Hz 下约有多少 speech embedding？
6. 为什么随机按 utterance 切分可能造成 speaker 泄漏？
7. 微调后 validation CER 降、test CER 升，可能是什么原因？
8. 为什么必须单独测试静音和否定词？

### C. 实战（每题 3 分）

9. 为自己的 10 条音频生成 JSONL，并通过路径/采样率检查。
10. 从空白实现 edit distance，用三组手算例子验证。
11. 有 GPU 时运行 0.6B baseline，保存逐条预测与 RTF。
12. 写出包含数据、质量、延迟、幻觉和回滚的上线门禁。

达到 19/24 只是理论通关；真正完成还需在你自己的、从未参与训练的数据上跑出可复现报告。
"""
        ),
        md(
            r"""
## 结业综合任务

选择一个真实场景（会议、客服、短视频、车载或方言），提交：

1. 需求：语言、是否流式、延迟与设备预算；
2. 数据卡：来源、许可、说话人切分、时长、噪声和文本规范；
3. baseline：至少一个 CTC/RNN-T 系统与一个 LALM；
4. 指标：总体与分桶 CER/WER、RTF、延迟、幻觉测试；
5. 搭建图：前端、模型、解码、后处理、服务和状态；
6. 失败分析：至少 20 条错误分类；
7. 改进实验：一次只改变一个变量；
8. 复现说明：代码、配置、随机种子、模型和数据版本。

能独立完成并答辩这八项，才算从“会运行模型”进入“会搭建和评估 ASR 系统”。
"""
        ),
        md(
            r"""
## 离场小测（闭卷发给老师）

1. 画出 Qwen3-ASR 数据流并标注下采样。
2. 为什么一定要先跑 baseline 再微调？
3. 给出 CER 公式，并手算一个例子。
4. 设计三条能发现生成式 ASR 幻觉的音频。
5. 你的目标场景是什么？质量、延迟、硬件三项约束分别是什么？

附上本课断言结果；有 GPU 时再附真实 baseline 表。
"""
        ),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
            "course": {
                "stage": "前沿声学模型",
                "lesson": 46,
                "version": 1,
                "evidence_level": 4,
            },
        }
    )
    return nb


LESSONS = {
    42: ("42_Conformer_卷积注意力与流式边界", lesson_42),
    43: ("43_RNNT与TDT_二维对齐和跳帧", lesson_43),
    44: ("44_自监督语音预训练_遮挡与声学表示", lesson_44),
    45: ("45_迷你音频语言模型_AudioEncoder接入LLM", lesson_45),
    46: ("46_Qwen3ASR_推理微调与验收", lesson_46),
}


def write_and_execute(number: int) -> tuple[Path, Path]:
    stem, builder = LESSONS[number]
    source_path = NOTEBOOK_DIR / f"{stem}.ipynb"
    executed_output = executed_path(source_path)
    ensure_executed_directories()
    notebook = builder()
    nbf.write(notebook, source_path)

    executed = nbf.from_dict(notebook)
    client = NotebookClient(
        executed,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    sanitize_notebook_outputs(executed)
    nbf.write(executed, executed_output)
    return source_path, executed_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lesson",
        type=int,
        choices=sorted(LESSONS),
        action="append",
        help="Build one or more lessons; default builds all implemented lessons.",
    )
    args = parser.parse_args()
    selected = args.lesson or sorted(LESSONS)
    for number in selected:
        source, executed = write_and_execute(number)
        print(f"BUILT {source.relative_to(ROOT)}")
        print(f"EXECUTED {executed.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
