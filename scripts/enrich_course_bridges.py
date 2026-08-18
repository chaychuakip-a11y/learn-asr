from __future__ import annotations

from pathlib import Path
import re

import nbformat as nbf

from notebook_layout import executed_path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
TAG = "course-bridge-v3"


EARLY_RECALL = {
    1: (
        "频率与周期怎样互换；16 kHz 的 25 ms 有多少采样点；peak、RMS 与 dBFS 分别回答什么",
        "[音频零基础桥梁课](音频零基础课程索引.md)",
        "波形 `[T]`、采样率 Hz、幅值与时长",
        "带真实时间解释的离散波形，以及采样/混叠判断",
    ),
    2: (
        "秒/毫秒/采样点怎样换算；采样率为什么属于波形契约；数组切片的 stop 为什么不包含",
        "[音频基础 1、4、6](音频零基础课程索引.md)",
        "波形 `[T]`、采样率、frame_ms、hop_ms",
        "帧 `[num_frames, frame_size]`、帧起点和有效边界",
    ),
    3: (
        "正弦波的频率、周期、相位；波形叠加；采样率与可表示频率的关系",
        "[音频基础 2、4、5](音频零基础课程索引.md)",
        "一帧波形 `[N]` 与采样率",
        "复数 DFT/FFT 与带 Hz 坐标的频谱 bin",
    ),
    4: (
        "frame/hop 的索引；窗函数的目的；单帧 FFT 的频率轴",
        "[主线第 2～3 课](核心课程索引_第01到41课.md)",
        "长波形、frame/window/hop、n_fft",
        "时间帧 × 频率 bin 的功率谱及两条坐标轴",
    ),
    5: (
        "幅度、功率与 dB 的不同公式；STFT shape；频率 bin 的 Hz 含义",
        "[音频基础 3](音频基础_03_振幅RMS功率与dB.ipynb)与主线第 4 课",
        "非负频率功率谱 `[T,F]` 与采样率",
        "Mel 滤波器组能量和 Log-Mel `[T,M]`",
    ),
    6: (
        "真实 WAV 的输入审计；分帧/STFT/Mel 每一步 shape；dB 参考值和数值下限",
        "[音频基础 6](音频基础_06_真实WAV读取试听与输入审计.ipynb)与主线第 1～5 课",
        "明确契约的波形、采样率和前端配置",
        "可测试、可与 Librosa 数值对照的 Log-Mel 前端",
    ),
}


PHASE_BRIDGES = [
    (
        range(7, 10),
        "Log-Mel `[T,F]` 的轴与单位；变长序列为什么需要 length/mask；padding 不能参与统计",
        "特征、length、mask 与明确的 `[B,T,F]` 契约",
        "不受 padding 污染、时间长度可追踪的声学表示",
    ),
    (
        range(10, 15),
        "编码器输出时间长度；概率与 log 概率；有效帧 mask；重复 token 与 blank 的区别",
        "logits/log-probs、input_lengths、targets、target_lengths",
        "可验算的 CTC 概率、损失、解码结果或训练证据",
    ),
    (
        range(15, 19),
        "CTC collapse 与 prefix 状态；因果/非因果上下文；整段结果的基线",
        "按时间到达的 chunk、前端/模型/解码器状态",
        "可与离线对照的 partial/final、状态更新和延迟指标",
    ),
    (
        range(19, 25),
        "CTC beam 中的候选前缀；概率与负对数代价方向；开发集和测试集权限",
        "声学候选、语言模型/图状态、分数权重",
        "可解释、可冻结调参、可分块保持状态的上下文解码结果",
    ),
    (
        range(25, 31),
        "PyTorch 基线输出；shape/dtype/dynamic axes；流式 cache；P50/P99 与 RTF",
        "冻结模型与数值基线、目标硬件、输入输出/状态协议",
        "有一致性、性能、并发隔离和回滚证据的部署产物",
    ),
    (
        range(31, 37),
        "采样率、PCM、通道、RMS/dBFS、SNR；流式 chunk/cache 与时间戳",
        "真实麦克风波形、参考信号/多通道和会话状态",
        "保持时间映射、可旁路、可 reset 且用 CER/SNR 双重验证的前端输出",
    ),
    (
        range(37, 42),
        "raw ASR 与 normalized text 的证据边界；置信度；N-best；会话状态隔离",
        "带时间、说话人、候选与置信度的可追溯识别结果",
        "保留原证据、可校准、可拒绝或澄清的文本/语义结果",
    ),
    (
        range(42, 47),
        "CTC/流式状态和延迟；训练/测试数据权限；基线、消融与外部评测证据",
        "明确任务、数据、算力、延迟与风险约束",
        "能与 CTC/RNN-T/AED/LALM 基线公平比较的现代模型实验",
    ),
]


def lesson_number(path: Path) -> int:
    match = re.match(r"(\d{2})_", path.name)
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def phase_bridge(number: int) -> tuple[str, str, str, str]:
    if number in EARLY_RECALL:
        return EARLY_RECALL[number]
    for numbers, recall, input_contract, output_contract in PHASE_BRIDGES:
        if number in numbers:
            recovery = (
                f"[上一课]("
                + next(
                    p.name
                    for p in NOTEBOOK_DIR.glob(f"{number - 1:02d}_*.ipynb")
                    if not p.stem.endswith("_已运行")
                )
                + ")与[唯一学习路径](../LEARNING_PATH.md)"
            )
            return recall, recovery, input_contract, output_contract
    raise ValueError(f"unsupported lesson {number}")


def bridge_cell(number: int):
    recall, recovery, input_contract, output_contract = phase_bridge(number)
    text = f"""<!-- {TAG} -->
## 知识接力：先取回旧知识，再进入本课

### 3 分钟闭卷回忆

在新 Markdown cell 中回答，**不要先翻前文**：{recall}。

- 三项都能用“含义 + 单位/shape + 一个数字例子”回答：进入本课。
- 能回答两项：学习本课，但把缺口记入 `LEARNING_LOG.md`。
- 只能回答零到一项：先回到 {recovery}，做一次最小实验；不要靠继续看新术语掩盖断点。

### 本课接口契约

```text
输入：{input_contract}
  ↓ 本课要学会的变换、状态或判断
输出：{output_contract}
```

学完后必须能解释：输入的哪个单位/shape/状态若丢失，会让输出“仍能运行却语义错误”。
"""
    cell = nbf.v4.new_markdown_cell(text.strip() + "\n")
    cell.metadata["tags"] = [TAG]
    return cell


def insert_bridge(notebook, number: int) -> None:
    notebook.cells = [
        cell
        for cell in notebook.cells
        if TAG not in cell.metadata.get("tags", []) and TAG not in cell.source
    ]
    insert_at = 3 if len(notebook.cells) >= 3 else len(notebook.cells)
    notebook.cells.insert(insert_at, bridge_cell(number))
    notebook.metadata.setdefault("course", {})["bridge"] = TAG
    nbf.validator.normalize(notebook, strip_invalid_metadata=True)


def main() -> None:
    sources = sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.ipynb"))
    for source in sources:
        number = lesson_number(source)
        source_notebook = nbf.read(source, as_version=4)
        insert_bridge(source_notebook, number)
        nbf.write(source_notebook, source)

        reference = executed_path(source)
        if not reference.exists():
            raise FileNotFoundError(f"missing executed reference: {reference}")
        reference_notebook = nbf.read(reference, as_version=4)
        insert_bridge(reference_notebook, number)
        nbf.write(reference_notebook, reference)
        print(f"bridged lesson {number:02d}: {source.name}")


if __name__ == "__main__":
    main()
