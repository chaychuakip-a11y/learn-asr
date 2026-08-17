from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "专题_流式ASR实验室_Chunk缓存PGS与实时率.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
    md(
        """
# 流式 ASR 实验室：Chunk、缓存、PGS、RTF 与尾延迟

这是第 15～18、24、29 课的贯通专题。我们用真实开源音频和可验证的小组件回答五个问题：

1. 任意 chunk 边界下，在线分帧是否与离线完全一致？
2. 因果编码器缓存是否既不漏算也不重复计算？
3. PGS 遇到替换、重复包、乱序和多会话时是否正确？
4. RTF、首结果、最终延迟、P50/P95/P99 分别测什么？
5. 为什么平均 RTF 小于 1，系统仍可能积压或让用户感觉很慢？

专题中的“模型”是可验证的教学组件。重点是状态与时序契约，不冒充成熟识别模型。
"""
    ),
    md(
        """
## 0. 端到端时间线

```text
麦克风样本
  → 网络/设备 chunk
  → 在线分帧缓存
  → 特征缓存
  → 编码器 cache / right context
  → CTC 解码状态
  → PGS 暂定与稳定文本
  → endpoint / final
```

每一层都有自己的状态。把它们全部塞进一个全局变量，单会话可能正常，并发时会互相污染。
"""
    ),
    code(
        """
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from IPython.display import Audio, clear_output, display

def find_root():
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("请从 learn_asr 或 notebooks 目录启动 Jupyter")

ROOT = find_root()
AUDIO_PATH = ROOT / "data" / "spoken_digits_0_to_9_16k.wav"
wave, sample_rate = sf.read(AUDIO_PATH, dtype="float32")
if wave.ndim > 1:
    wave = wave.mean(axis=1)

print(f"真实音频：{AUDIO_PATH.name}｜采样率 {sample_rate} Hz｜时长 {len(wave)/sample_rate:.2f} 秒")
display(Audio(wave, rate=sample_rate))
"""
    ),
    md(
        """
## 1. 网络 chunk 不是声学 frame

- **chunk**：设备或网络一次交付的一批样本，边界会随网络和回调变化；
- **frame**：固定分析窗口，例如 25 ms；
- **hop**：相邻帧起点间隔，例如 10 ms。

16 kHz 下 frame=25 ms、hop=10 ms，对应 400 点和 160 点。一个 317 点的网络 chunk 不应被强行当成一帧。
"""
    ),
    code(
        """
FRAME_LENGTH = round(sample_rate * 0.025)
HOP_LENGTH = round(sample_rate * 0.010)

def offline_frames(signal, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    signal = np.asarray(signal)
    if len(signal) < frame_length:
        return np.empty((0, frame_length), dtype=signal.dtype)
    starts = range(0, len(signal) - frame_length + 1, hop_length)
    return np.stack([signal[start:start + frame_length] for start in starts])

class StreamingFramer:
    def __init__(self, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
        if frame_length <= 0 or hop_length <= 0:
            raise ValueError("frame_length 和 hop_length 必须为正数")
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.buffer = np.empty(0, dtype=np.float32)
        self.total_input_samples = 0
        self.total_frames = 0

    def accept(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self.total_input_samples += len(chunk)
        self.buffer = np.concatenate([self.buffer, chunk])
        frames = []
        while len(self.buffer) >= self.frame_length:
            frames.append(self.buffer[:self.frame_length].copy())
            self.buffer = self.buffer[self.hop_length:]
            self.total_frames += 1
        if not frames:
            return np.empty((0, self.frame_length), dtype=np.float32)
        return np.stack(frames)

    def audit(self):
        return {
            "input_samples": self.total_input_samples,
            "frames": self.total_frames,
            "buffer_samples": len(self.buffer),
            "buffer_starts_at_sample": self.total_frames * self.hop_length,
        }

offline = offline_frames(wave)
print("frame:", FRAME_LENGTH, "samples｜hop:", HOP_LENGTH, "samples｜offline frames:", len(offline))
"""
    ),
    code(
        """
def random_partition(length, rng, minimum=1, maximum=1200):
    sizes = []
    remaining = length
    while remaining:
        size = min(remaining, int(rng.integers(minimum, maximum + 1)))
        sizes.append(size)
        remaining -= size
    return sizes

def stream_with_sizes(signal, sizes):
    framer = StreamingFramer()
    outputs = []
    cursor = 0
    for size in sizes:
        produced = framer.accept(signal[cursor:cursor + size])
        if len(produced):
            outputs.append(produced)
        cursor += size
    frames = np.concatenate(outputs, axis=0) if outputs else np.empty((0, FRAME_LENGTH), np.float32)
    return frames, framer

rng = np.random.default_rng(20260817)
worst_error = 0.0
for trial in range(100):
    sizes = random_partition(len(wave), rng)
    online, framer = stream_with_sizes(wave, sizes)
    assert online.shape == offline.shape
    worst_error = max(worst_error, float(np.max(np.abs(online - offline))))
    expected_leftover = wave[len(online) * HOP_LENGTH:]
    assert np.array_equal(framer.buffer, expected_leftover)

print("100 种随机切块全部通过")
print("离线/在线最大样本差：", worst_error)
print("最终状态：", framer.audit())
"""
    ),
    md(
        """
### 怎样理解 buffer

随机分割是一种性质测试：只要样本顺序相同，不论如何切块，完整帧序列必须与离线标准逐样本一致。

最终 buffer 不是泄漏；它从“下一个尚未输出帧的起点”开始保存剩余样本。尾部是否补零必须由明确的 finalize 策略决定，不能在每个 chunk 末尾补零。
"""
    ),
    code(
        """
def naive_chunk_frames(signal, sizes):
    outputs = []
    cursor = 0
    for size in sizes:
        local = offline_frames(signal[cursor:cursor + size])
        if len(local):
            outputs.append(local)
        cursor += size
    return np.concatenate(outputs) if outputs else np.empty((0, FRAME_LENGTH), np.float32)

pattern = [317, 503, 211, 997]
sizes = []
remaining = len(wave)
i = 0
while remaining:
    size = min(remaining, pattern[i % len(pattern)])
    sizes.append(size)
    remaining -= size
    i += 1

correct, _ = stream_with_sizes(wave, sizes)
naive = naive_chunk_frames(wave, sizes)
print("正确帧数：", len(correct))
print("每个 chunk 独立分帧：", len(naive))
print("丢失帧数：", len(correct) - len(naive))
"""
    ),
    md(
        """
## 2. 因果编码器缓存：不漏算，也不重复返回

用长度为 5 的因果 FIR 表示最小因果卷积：当前输出只依赖当前与过去。跨 chunk 时保存最近 K-1 个输入；既不能清零，也不能把缓存区域的输出再次返回。
"""
    ),
    code(
        """
FIR_KERNEL = np.array([0.45, 0.25, 0.15, 0.10, 0.05], dtype=np.float64)

def offline_causal_fir(signal, kernel=FIR_KERNEL):
    return np.convolve(np.asarray(signal, dtype=np.float64), kernel, mode="full")[:len(signal)]

class StreamingCausalFIR:
    def __init__(self, kernel=FIR_KERNEL):
        self.kernel = np.asarray(kernel, dtype=np.float64)
        self.cache = np.zeros(len(self.kernel) - 1, dtype=np.float64)

    def accept(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float64).reshape(-1)
        combined = np.concatenate([self.cache, chunk])
        full = np.convolve(combined, self.kernel, mode="full")
        start = len(self.cache)
        output = full[start:start + len(chunk)]
        if len(self.cache):
            self.cache = combined[-len(self.cache):].copy()
        return output

short_signal = wave[:sample_rate].astype(np.float64)
offline_fir = offline_causal_fir(short_signal)
rng = np.random.default_rng(9)
maximum_error = 0.0
for trial in range(100):
    sizes = random_partition(len(short_signal), rng, maximum=500)
    model = StreamingCausalFIR()
    cursor = 0
    parts = []
    for size in sizes:
        parts.append(model.accept(short_signal[cursor:cursor + size]))
        cursor += size
    online_fir = np.concatenate(parts)
    maximum_error = max(maximum_error, float(np.max(np.abs(online_fir - offline_fir))))

print("100 种随机切块的因果卷积全部通过")
print("离线/流式最大误差：", maximum_error)
print("cache shape：", model.cache.shape)
"""
    ),
    md(
        """
### 推广到真实编码器

- 因果卷积保存过去输入或激活；
- Chunk Attention 保存有限左上下文 K/V；
- 循环网络保存 hidden/cell state；
- subsampling 还可能保存不足一个步幅的尾部。

必须测试 chunk=1、随机 chunk、空 chunk、最后短 chunk、reset、新会话、两会话交错，以及长音频内存是否有界。
"""
    ),
    md(
        """
## 3. PGS：暂定文本不是只能追加的字符串

本专题沿用教学字段：`apd` 追加片段，`rpl` 替换 `rg=[start,end]` 范围，`sn` 表示顺序，`event_id` 用于幂等去重，`final` 结束会话。具体厂商字段以其正式协议为准。
"""
    ),
    code(
        """
@dataclass
class PGSBuffer:
    parts: list[str] = field(default_factory=list)
    seen_event_ids: set[str] = field(default_factory=set)
    last_sequence: int = -1
    final: bool = False

    def accept(self, event):
        event_id = str(event["event_id"])
        sequence = int(event["sn"])
        if event_id in self.seen_event_ids:
            return "duplicate_ignored"
        if self.final:
            raise RuntimeError("final 后不能再修改文本")
        if sequence != self.last_sequence + 1:
            raise ValueError(f"事件乱序或缺包：期望 sn={self.last_sequence + 1}，收到 {sequence}")
        mode = event["pgs"]
        text = str(event["text"])
        if mode == "apd":
            self.parts.append(text)
        elif mode == "rpl":
            start, end = map(int, event["rg"])
            if start < 0 or end < start or end >= len(self.parts):
                raise IndexError(f"非法替换范围 {event['rg']}，当前片段数 {len(self.parts)}")
            self.parts[start:end + 1] = [text]
        else:
            raise ValueError(f"未知 pgs 模式：{mode}")
        self.seen_event_ids.add(event_id)
        self.last_sequence = sequence
        self.final = bool(event.get("final", False))
        return "accepted"

    @property
    def text(self):
        return "".join(self.parts)

events = [
    {"event_id": "e0", "sn": 0, "pgs": "apd", "text": "我想订"},
    {"event_id": "e1", "sn": 1, "pgs": "apd", "text": "明天"},
    {"event_id": "e2", "sn": 2, "pgs": "rpl", "rg": [1, 1], "text": "后天"},
    {"event_id": "e3", "sn": 3, "pgs": "apd", "text": "的机票", "final": True},
]

buffer = PGSBuffer()
for event in events:
    print(event["sn"], event["pgs"], buffer.accept(event), "→", end=" ")
    print(buffer.text)
assert buffer.text == "我想订后天的机票"
"""
    ),
    code(
        """
step_widget = widgets.IntSlider(min=0, max=len(events), value=0, description="已接收事件")
pgs_output = widgets.Output()

def replay_pgs(change=None):
    count = step_widget.value
    replay = PGSBuffer()
    with pgs_output:
        clear_output(wait=True)
        print("初始文本：''")
        for event in events[:count]:
            replay.accept(event)
            print(f"sn={event['sn']} {event['pgs']} {event.get('rg', '')} → {replay.text!r}")

step_widget.observe(replay_pgs, names="value")
display(step_widget, pgs_output)
replay_pgs()

duplicate_demo = PGSBuffer()
assert duplicate_demo.accept(events[0]) == "accepted"
assert duplicate_demo.accept(events[0]) == "duplicate_ignored"
assert duplicate_demo.text == "我想订"

out_of_order_demo = PGSBuffer()
try:
    out_of_order_demo.accept(events[1])
except ValueError as exc:
    print("乱序检查：", exc)
"""
    ),
    md("## 4. 多会话隔离\n\n连接 A 和 B 的事件会交错执行。正确结构是 `session_id → SessionState`，每个状态分别持有 framer、encoder cache、decoder、PGS、endpoint 和指标。"),
    code(
        """
class SessionManager:
    def __init__(self):
        self.sessions = {}

    def start(self, session_id):
        if session_id in self.sessions:
            raise ValueError("重复 session_id")
        self.sessions[session_id] = PGSBuffer()

    def accept(self, session_id, event):
        if session_id not in self.sessions:
            raise KeyError("event 早于 start")
        return self.sessions[session_id].accept(event)

    def text(self, session_id):
        return self.sessions[session_id].text

manager = SessionManager()
manager.start("A")
manager.start("B")
events_a = [
    {"event_id": "a0", "sn": 0, "pgs": "apd", "text": "北京"},
    {"event_id": "a1", "sn": 1, "pgs": "apd", "text": "天气", "final": True},
]
events_b = [
    {"event_id": "b0", "sn": 0, "pgs": "apd", "text": "上海"},
    {"event_id": "b1", "sn": 1, "pgs": "apd", "text": "机票", "final": True},
]
for session_id, event in [("A", events_a[0]), ("B", events_b[0]), ("A", events_a[1]), ("B", events_b[1])]:
    manager.accept(session_id, event)
assert manager.text("A") == "北京天气"
assert manager.text("B") == "上海机票"
print("A：", manager.text("A"), "｜B：", manager.text("B"), "｜未串话")
"""
    ),
    md(
        """
## 5. RTF、延迟和吞吐必须分开

RTF = 纯处理耗时 / 音频时长。RTF=0.25 表示处理 1 秒音频平均需要约 0.25 秒计算时间，不表示首字一定在 250 ms 出现。

- **首结果时间**：说话开始到第一次可见结果；
- **更新延迟**：某段音频结束到包含它的 partial 返回；
- **最终延迟**：说话结束到 final 返回；
- **尾延迟**：P95/P99，暴露少数非常慢的请求；
- **吞吐**：单位时间能处理多少路或多少秒音频。
"""
    ),
    code(
        """
def simulate_stream(utterance_ms=5000, chunk_ms=320, right_context_ms=0,
                    compute_rtf=0.3, jitter_ms=5, seed=0):
    rng = np.random.default_rng(seed)
    chunk_ends = np.arange(chunk_ms, utterance_ms + chunk_ms, chunk_ms, dtype=float)
    chunk_ends[-1] = utterance_ms
    chunk_starts = np.concatenate([[0.0], chunk_ends[:-1]])
    chunk_durations = chunk_ends - chunk_starts
    available = np.minimum(chunk_ends + right_context_ms, utterance_ms + right_context_ms)
    compute = chunk_durations * compute_rtf + np.maximum(0.0, rng.normal(0, jitter_ms, len(chunk_ends)))
    starts = np.empty_like(chunk_ends)
    finishes = np.empty_like(chunk_ends)
    previous_finish = 0.0
    for i in range(len(chunk_ends)):
        starts[i] = max(available[i], previous_finish)
        finishes[i] = starts[i] + compute[i]
        previous_finish = finishes[i]
    update_delays = finishes - chunk_ends
    return {
        "chunk_ends": chunk_ends,
        "available": available,
        "starts": starts,
        "finishes": finishes,
        "compute": compute,
        "update_delays": update_delays,
        "rtf": compute.sum() / utterance_ms,
        "first_result_ms": finishes[0],
        "final_delay_ms": finishes[-1] - utterance_ms,
        "max_queue_wait_ms": float(np.max(starts - available)),
    }

chunk_widget = widgets.SelectionSlider(options=[40, 80, 160, 320, 640], value=320, description="chunk ms")
right_widget = widgets.SelectionSlider(options=[0, 40, 80, 160, 320], value=0, description="右上下文 ms")
rtf_widget = widgets.FloatSlider(min=0.1, max=1.5, step=0.1, value=0.3, description="计算 RTF", continuous_update=False)
jitter_widget = widgets.IntSlider(min=0, max=100, step=5, value=5, description="抖动 ms", continuous_update=False)
latency_output = widgets.Output()

def show_latency(*_):
    result = simulate_stream(chunk_ms=chunk_widget.value, right_context_ms=right_widget.value,
                             compute_rtf=rtf_widget.value, jitter_ms=jitter_widget.value, seed=7)
    delays = result["update_delays"]
    with latency_output:
        clear_output(wait=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        axes[0].plot(result["chunk_ends"], result["chunk_ends"], "--", label="音频结束")
        axes[0].plot(result["chunk_ends"], result["finishes"], "o-", label="结果返回")
        axes[0].set(xlabel="音频时间 (ms)", ylabel="墙钟时间 (ms)", title="输入与结果时间")
        axes[0].legend()
        axes[1].plot(result["chunk_ends"], delays, "o-")
        axes[1].set(xlabel="chunk 结束位置 (ms)", ylabel="更新延迟 (ms)", title="partial 等待")
        plt.tight_layout()
        plt.show()
        p50, p95, p99 = np.percentile(delays, [50, 95, 99])
        print(f"RTF={result['rtf']:.3f}｜首结果={result['first_result_ms']:.1f} ms｜最终延迟={result['final_delay_ms']:.1f} ms")
        print(f"P50/P95/P99={p50:.1f}/{p95:.1f}/{p99:.1f} ms｜最大排队={result['max_queue_wait_ms']:.1f} ms")
        if result["max_queue_wait_ms"] > 0:
            print("出现排队：可处理 chunk 的到达速度超过完成速度。")

for control in [chunk_widget, right_widget, rtf_widget, jitter_widget]:
    control.observe(show_latency, names="value")
display(widgets.VBox([chunk_widget, right_widget, rtf_widget, jitter_widget]), latency_output)
show_latency()
"""
    ),
    md(
        """
### 拖动前先预测

1. chunk 从 80 ms 增到 640 ms，RTF 是否必然变化？首结果呢？
2. 增加 160 ms 右上下文，哪类延迟至少增加约 160 ms？
3. 计算 RTF 从 0.3 增到 1.2，队列如何变化？
4. 平均 jitter 很小，为什么 P99 仍可能恶化？

这个模拟说明：RTF 是计算量比率，延迟是时间线结果，二者不能互相替代。

当前模拟保守地假设句末也等待完整右上下文。真实系统收到明确 `end` 后可能用 flush/padding 提前结束；是否等待、怎样补齐、最终结果何时稳定，都必须写进 endpoint 契约并单独测量。
"""
    ),
    code(
        """
# 语料级 RTF 应用总计算/总音频；简单平均逐句 RTF 回答的是另一问题。
durations_s = np.array([0.4, 0.8, 1.5, 4.0, 12.0])
compute_s = 0.08 + 0.22 * durations_s
per_utterance_rtf = compute_s / durations_s
for duration, compute, rtf in zip(durations_s, compute_s, per_utterance_rtf):
    print(f"duration={duration:4.1f}s compute={compute:.3f}s RTF={rtf:.3f}")
print("简单平均逐句 RTF：", per_utterance_rtf.mean())
print("语料总计算/总音频：", compute_s.sum() / durations_s.sum())
"""
    ),
    md(
        """
## 6. Backpressure 与资源回收

若每 100 ms 到达一个 chunk，但处理平均要 130 ms，单路就会逐渐积压。多路并发共享计算资源时还要考虑容量。

生产系统至少需要有界队列、并发上限、负载拒绝或降级、每会话内存预算、分阶段耗时，以及在断连、超时、异常和 final 后可靠回收状态。
"""
    ),
    md(
        """
## 7. 最终闭卷测试（40 分）

每题 0～2 分，达到 **32/40**，且代码题真实运行，才算通过。

### 概念与预测

1. 区分 sample、frame、hop、chunk，分别写单位。
2. 为什么不能在每个网络 chunk 上独立分帧？
3. frame=400、hop=160，输出 10 帧后，buffer 从原输入哪个点开始？
4. 因果卷积核长度为 7，最少保存多少历史输入？
5. 右上下文为何可能提高质量，却增加延迟？
6. PGS `rpl` 为什么不能实现成 append？
7. RTF=0.4 能否推出首字延迟=400 ms？
8. 为什么同时报告 P50、P95、P99？

### 编程与排错

9. 从空白写 StreamingFramer，对 1000 种随机分割与离线逐样本对照。
10. 增加 finalize 的 `drop` 与 `pad` 策略并记录有效长度。
11. 故意让每个 chunk 独立分帧，量化丢失帧数。
12. 从空白实现因果 FIR cache，测试 chunk=1 和随机 chunk。
13. 故意多返回 cache 区域输出，画出重复时间步。
14. 给 PGSBuffer 加有界乱序缓存和缺包超时。
15. 重放重复事件，证明文本不变。
16. 交错运行 10 个 session，证明不串状态并在 final 后回收。
17. 分开记录排队、特征、编码、解码和序列化耗时。
18. 构造 RTF<1 但 P99 很高的案例。

### 系统设计

19. 画 `start → audio* → end → final → close` 状态机并列出非法转移。
20. 用 5 分钟解释“为什么流式不是把离线函数放进 for 循环”，必须包含随机切块、会话隔离和延迟时间线。
"""
    ),
    md(
        """
<details><summary>展开关键答案与评分锚点</summary>

1. sample 是离散幅值；frame 是分析窗口；hop 是帧起点间隔；chunk 是输入批次。
2. chunk 边界处不足一帧的样本和跨边界重叠会丢失或错位。
3. 从 `10×160=1600` 开始。
4. 至少 6 个历史输入。
5. 当前输出必须等待未来帧。
6. `rpl` 会修改已有范围，append 无法撤销暂定文本。
7. 不能；首结果还受 chunk、右上下文、排队、解码和稳定策略影响。
8. 平均值隐藏少数极慢请求，P99 暴露尾部体验与容量风险。

代码题必须有断言、随机种子、边界输入和失败案例。只展示一次正常输入最多得 1 分。

</details>
"""
    ),
    md(
        """
## 8. 离场票

- [ ] 任意随机 chunk 下，在线帧与离线帧逐样本相同。
- [ ] 编码器流式输出与离线因果输出在容差内一致。
- [ ] PGS 支持追加、替换、幂等、顺序检查、final 和多会话隔离。
- [ ] 我能分别计算 RTF、首结果、最终延迟、P50/P95/P99 和队列等待。
- [ ] 我能解释 right context、chunk、beam、endpoint 对质量和延迟的不同影响。
- [ ] 我能在断连、超时、错误和 final 后回收所有会话状态。

达到这些证据后，再进入语言模型与 WFST。
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "course": {"kind": "streaming-intensive-lab", "version": 1, "related_lessons": [15, 16, 17, 18, 24, 29]},
    },
)

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUT)
print(f"wrote {OUT}")
