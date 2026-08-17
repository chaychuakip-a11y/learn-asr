from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "学习中枢_诊断与掌握度仪表盘.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
    md(
        """
# 学习中枢：诊断、路线与掌握度仪表盘

这不是第 0 课，而是贯穿 41 课的“驾驶舱”。第一次学习、每完成一个阶段、以及间隔复习时都回来运行一次。

你会完成四件事：

1. 用 24 道题诊断八个阶段，而不是凭感觉判断“会了”；
2. 画出自己的掌握度图，自动找到最早的知识断点；
3. 用统一通关标准决定继续、复习还是返工；
4. 把分数、薄弱点和下一步保存为个人进度。

> 正确答案不是最终目标。真正掌握意味着：能解释、能预测、能实现、能排错、能迁移。
"""
    ),
    md(
        """
## 使用顺序

- **首次进入**：先闭卷做诊断，不要打开其他 Notebook。
- **学习每课**：预测输出 → 运行实验 → 改一个变量 → 写解释 → 从空白重写。
- **阶段结束**：回到这里复测，并完成对应阶段的综合任务。
- **复习日**：第 1、7、30 天只看问题，不先看代码和答案。

评分解释：诊断题只能说明“概念入口是否稳固”，不能替代编程与综合任务。
"""
    ),
    code(
        """
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, clear_output, display

plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

STAGES = {
    "声音与特征": "第 1～6 课",
    "张量与编码": "第 7～9 课",
    "CTC": "第 10～14 课",
    "流式 ASR": "第 15～18 课",
    "LM 与 WFST": "第 19～23 课",
    "量化与部署": "第 24～30 课",
    "音频前端": "第 31～36 课",
    "后处理语义": "第 37～41 课",
}

START_LESSON = {
    "声音与特征": 1,
    "张量与编码": 7,
    "CTC": 10,
    "流式 ASR": 15,
    "LM 与 WFST": 19,
    "量化与部署": 24,
    "音频前端": 31,
    "后处理语义": 37,
}

print("环境准备完成。请先做诊断，不要提前查看后面的解释。")
"""
    ),
    md("## 一、24 题闭卷诊断\n\n每个阶段 3 题。不会就选“暂时不知道”，这比猜测更有学习价值。"),
    code(
        """
QUESTIONS = [
    ("声音与特征", "16 kHz 采样率表示什么？", ["每秒 16 000 个样本", "最高频率一定是 16 kHz", "每帧 16 000 点", "暂时不知道"], 0, "采样率是每秒离散采样次数；理想可表示的最高频率小于 8 kHz。"),
    ("声音与特征", "数字音频幅值为 0.5，相对满刻度的 dBFS 最接近？", ["+6 dBFS", "-3 dBFS", "-6 dBFS", "暂时不知道"], 2, "幅值比用 20log10(0.5)，约为 -6.02 dBFS。"),
    ("声音与特征", "STFT 相比单次 FFT 主要增加了什么信息？", ["说话人身份", "频率随时间的变化", "采样位深", "暂时不知道"], 1, "STFT 对重叠短帧逐帧做 FFT，因此保留粗粒度时间位置。"),

    ("张量与编码", "一批音频补零后，模型为什么还需要 length 或 mask？", ["提升采样率", "区分真实帧与 padding", "增加类别数", "暂时不知道"], 1, "补零只是形状对齐，mask 才告诉模型和损失哪些位置有效。"),
    ("张量与编码", "Conv1d 输入通常使用哪种 shape？", ["[B, T, C]", "[B, C, T]", "[T, B]", "暂时不知道"], 1, "PyTorch Conv1d 的常见输入约定是 batch、channel、time。"),
    ("张量与编码", "时间下采样 4 倍后，400 帧大约变成多少帧？", ["100", "396", "1 600", "暂时不知道"], 0, "忽略边界细节时，时间长度约除以 4。"),

    ("CTC", "CTC 中 blank 的核心作用是什么？", ["表示静音文件", "允许帧数多于标签数并表达对齐", "替代所有空格", "暂时不知道"], 1, "blank 与重复折叠规则共同表达未知帧级对齐。"),
    ("CTC", "路径 A blank A 折叠后的标签是？", ["AA", "A", "blank", "暂时不知道"], 0, "相邻重复先合并，再删除 blank；被 blank 隔开的 A 会保留为两个 A。"),
    ("CTC", "CTC 前向算法为什么使用 LogSumExp？", ["只取最大路径", "稳定地累加多条路径概率", "把标签排序", "暂时不知道"], 1, "目标概率是所有合法路径概率之和，log-space 下用 LogSumExp 稳定计算。"),

    ("流式 ASR", "流式分帧器为什么要保留尾部缓存？", ["为了增大音量", "未凑成完整帧的样本要与下个 chunk 拼接", "为了保存文本", "暂时不知道"], 1, "chunk 边界通常不与帧边界对齐，余数不能丢。"),
    ("流式 ASR", "RTF=0.25 表示什么？", ["处理 1 秒音频约需 0.25 秒", "延迟固定 250 ms", "准确率 25%", "暂时不知道"], 0, "RTF=处理耗时/音频时长；它不是首字延迟或 P99 延迟。"),
    ("流式 ASR", "PGS 的 rpl 事件通常表达什么？", ["追加稳定文本", "替换某段暂定文本", "删除模型", "暂时不知道"], 1, "动态修正协议必须表达追加、替换及替换范围，客户端不能只拼字符串。"),

    ("LM 与 WFST", "语言模型在解码时主要提供什么？", ["声学帧", "词序列先验", "麦克风增益", "暂时不知道"], 1, "LM 评价词序列是否符合语言与领域习惯。"),
    ("LM 与 WFST", "WFST 路径权重常用负对数概率的原因？", ["路径代价可相加", "让图变成图片", "消除所有状态", "暂时不知道"], 0, "乘法概率经 -log 变成可累加代价，最短路对应较高概率。"),
    ("LM 与 WFST", "L 与 G 组合时必须匹配什么？", ["文件大小", "L 的输出符号与 G 的输入符号", "采样率", "暂时不知道"], 1, "组合依赖中间符号表和标签语义一致。"),

    ("量化与部署", "ONNX 导出后最重要的第一项验证是？", ["文件能打开", "与原框架在固定输入上数值一致", "文件名好看", "暂时不知道"], 1, "可加载不等于语义正确，必须验证 shape、状态和数值误差。"),
    ("量化与部署", "INT8 量化校准数据应当？", ["全是零", "代表真实输入分布", "只用最长音频", "暂时不知道"], 1, "激活范围来自校准数据；失真的分布会产生错误尺度。"),
    ("量化与部署", "流式服务中 cache 应属于谁？", ["全局共享", "具体会话", "日志线程", "暂时不知道"], 1, "状态必须按会话隔离，并在结束、超时或断连后回收。"),

    ("音频前端", "削波最直接的可见迹象是？", ["波形峰顶被压平", "频谱只有低频", "采样率变大", "暂时不知道"], 0, "超过可表示范围会被截断，产生平顶和额外谐波失真。"),
    ("音频前端", "VAD hangover 的主要目的？", ["避免词尾被过早截断", "提高位深", "删除回声参考", "暂时不知道"], 0, "语音概率短暂下降时继续保留一段，减少切掉弱词尾。"),
    ("音频前端", "AEC 为什么需要播放参考信号？", ["估计扬声器到麦克风的回声路径", "训练语言模型", "生成标点", "暂时不知道"], 0, "没有 far-end reference 很难区分近端语音与设备播放形成的回声。"),

    ("后处理语义", "ITN 的职责更接近？", ["把‘一百二十元’规范成‘120元’", "做 FFT", "估计声源方向", "暂时不知道"], 0, "ITN 把口语形式转为适合阅读或业务处理的书写形式。"),
    ("后处理语义", "为什么语义模块应保留 ASR 原始文本和 N-best？", ["方便追溯证据与处理不确定性", "节约全部内存", "让采样率升高", "暂时不知道"], 0, "语义输出应能追溯到识别证据，不能悄悄改掉姓名、数字和否定词。"),
    ("后处理语义", "低置信度关键槽位最稳妥的动作？", ["让 LLM 猜一个", "向用户确认", "直接删除", "暂时不知道"], 1, "金额、号码、姓名等高风险槽位应触发确认或回退。"),
]

quiz_widgets = []
stage_boxes = []
for stage in STAGES:
    children = []
    for index, (q_stage, prompt, options, answer, why) in enumerate(QUESTIONS):
        if q_stage != stage:
            continue
        radio = widgets.RadioButtons(
            options=options,
            value=None,
            description="",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "0px"},
        )
        quiz_widgets.append((index, radio))
        children.extend([widgets.HTML(f"<b>{index + 1}. {prompt}</b>"), radio])
    stage_boxes.append(widgets.VBox(children))

accordion = widgets.Accordion(children=stage_boxes)
for i, stage in enumerate(STAGES):
    accordion.set_title(i, f"{stage}｜{STAGES[stage]}")

display(accordion)
"""
    ),
    code(
        """
score_button = widgets.Button(description="提交并生成诊断图", button_style="primary")
quiz_output = widgets.Output()
quiz_result = {}

def plot_radar(stage_percent):
    labels = list(STAGES)
    values = [stage_percent[s] for s in labels]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={"polar": True})
    ax.plot(angles_closed, values_closed, marker="o", linewidth=2)
    ax.fill(angles_closed, values_closed, alpha=0.15)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_title("八阶段闭卷诊断（概念入口，不等于最终掌握）", pad=24)
    plt.show()

def score_quiz(_):
    answers = {i: radio.value for i, radio in quiz_widgets}
    with quiz_output:
        clear_output(wait=True)
        unanswered = [i + 1 for i, value in answers.items() if value is None]
        if unanswered:
            print("请先回答所有题目。未回答：", unanswered)
            return
        per_stage = defaultdict(lambda: [0, 0])
        wrong = []
        for i, (stage, prompt, options, answer, why) in enumerate(QUESTIONS):
            per_stage[stage][1] += 1
            if answers[i] == options[answer]:
                per_stage[stage][0] += 1
            else:
                wrong.append((i + 1, stage, prompt, options[answer], why))
        stage_percent = {s: round(100 * per_stage[s][0] / per_stage[s][1]) for s in STAGES}
        quiz_result.clear()
        quiz_result.update({"stage_percent": stage_percent, "wrong": wrong, "time": datetime.now().isoformat(timespec="seconds")})
        total = sum(v[0] for v in per_stage.values())
        print(f"总分：{total}/24。先找最早的知识断点，不要只看总分。")
        for stage in STAGES:
            correct, count = per_stage[stage]
            print(f"- {stage:<8}: {correct}/{count}（{stage_percent[stage]}%）")
        plot_radar(stage_percent)
        weak = [s for s in STAGES if stage_percent[s] < 67]
        if weak:
            first = weak[0]
            print(f"建议从第 {START_LESSON[first]} 课开始或回炉：{first}。原因：这是最早低于 2/3 的阶段。")
        else:
            print("概念入口全部达到 2/3。请用阶段综合任务继续验证实现与迁移能力。")
        if wrong:
            display(Markdown("### 错题反馈"))
            for number, stage, prompt, correct, why in wrong:
                display(Markdown(f"**第 {number} 题｜{stage}**：{prompt}<br>正确答案：`{correct}`<br>解释：{why}"))

score_button.on_click(score_quiz)
display(score_button, quiz_output)
"""
    ),
    md(
        """
## 二、用参数图建立“整条信号链”的直觉

拖动参数前先预测：

- 采样率增加，Nyquist 频率如何变化？
- 帧长增加，FFT 频率间隔如何变化？
- hop 变小，一秒钟会产生更多还是更少帧？

图中的“频率分辨率”是 `采样率 / FFT 点数`。它不等于模型一定能分辨两个任意接近的音高，但能帮助你理解频率网格。
"""
    ),
    code(
        """
sample_rate_widget = widgets.SelectionSlider(
    options=[8000, 16000, 24000, 48000], value=16000,
    description="采样率 Hz", continuous_update=False,
)
frame_ms_widget = widgets.IntSlider(
    min=10, max=50, step=5, value=25,
    description="帧长 ms", continuous_update=False,
)
hop_ms_widget = widgets.IntSlider(
    min=5, max=25, step=5, value=10,
    description="帧移 ms", continuous_update=False,
)

def signal_chain_view(sample_rate, frame_ms, hop_ms):
    duration = 0.08
    t = np.arange(round(sample_rate * duration)) / sample_rate
    signal = 0.65 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 1200 * t)
    frame_n = round(sample_rate * frame_ms / 1000)
    hop_n = round(sample_rate * hop_ms / 1000)
    n_fft = 1 << int(np.ceil(np.log2(frame_n)))
    frames_per_second = 1000 / hop_ms
    freq_step = sample_rate / n_fft
    nyquist = sample_rate / 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(t * 1000, signal, linewidth=1)
    for start in range(0, len(signal) - frame_n + 1, hop_n):
        axes[0].axvspan(start / sample_rate * 1000, (start + frame_n) / sample_rate * 1000, alpha=0.06)
    axes[0].set(title="波形与重叠帧", xlabel="时间 (ms)", ylabel="幅值")

    windowed = signal[:frame_n] * np.hanning(frame_n)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    axes[1].plot(freqs, 20 * np.log10(np.maximum(spectrum, 1e-8)))
    axes[1].set_xlim(0, min(4000, nyquist))
    axes[1].set(title="一帧的频谱", xlabel="频率 (Hz)", ylabel="相对幅度 (dB)")
    plt.tight_layout()
    plt.show()

    print(f"每帧 {frame_n} 点｜每次前进 {hop_n} 点｜约 {frames_per_second:.1f} 帧/秒")
    print(f"FFT 点数 {n_fft}｜频率网格间隔 {freq_step:.2f} Hz｜Nyquist 频率 {nyquist:.0f} Hz")

controls = {
    "sample_rate": sample_rate_widget,
    "frame_ms": frame_ms_widget,
    "hop_ms": hop_ms_widget,
}
display(widgets.VBox(list(controls.values())), widgets.interactive_output(signal_chain_view, controls))
"""
    ),
    md(
        """
## 三、掌握度不是一个分数

请按证据给每个阶段打 0～4 分：

- **0**：概念陌生；
- **1**：看答案能理解；
- **2**：闭卷能解释，能预测简单结果；
- **3**：能从空白实现核心算法，并排查常见错误；
- **4**：能在新数据、新约束或新系统中迁移，并说明边界。

通关要求不是所有阶段立刻达到 4，而是当前阶段至少达到 3，并且前置阶段没有低于 2。
"""
    ),
    code(
        """
mastery_sliders = {
    stage: widgets.IntSlider(min=0, max=4, value=0, step=1, description=stage, continuous_update=False)
    for stage in STAGES
}
mastery_output = widgets.Output()
mastery_button = widgets.Button(description="更新掌握度与下一步", button_style="info")

def update_mastery(_):
    values = {stage: slider.value for stage, slider in mastery_sliders.items()}
    with mastery_output:
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        names = list(values)
        scores = list(values.values())
        bars = ax.barh(names, scores)
        ax.set_xlim(0, 4)
        ax.set_xticks(range(5))
        ax.set_xlabel("证据等级（0～4）")
        ax.set_title("当前掌握度：先修补最早的断点")
        ax.invert_yaxis()
        for bar, score in zip(bars, scores):
            ax.text(score + 0.05, bar.get_y() + bar.get_height()/2, str(score), va="center")
        plt.show()

        first_gap = next((stage for stage, score in values.items() if score < 2), None)
        current = next((stage for stage, score in values.items() if score < 3), None)
        if first_gap:
            print(f"优先修补：{first_gap}，从第 {START_LESSON[first_gap]} 课开始。前置阶段低于 2 时不建议跳跃。")
        elif current:
            print(f"当前主攻：{current}，从第 {START_LESSON[current]} 课开始，并完成阶段综合任务。")
        else:
            print("所有阶段自评达到 3。请完成最终闭卷讲解、端到端项目和新数据评估后再判断是否真正掌握。")

mastery_button.on_click(update_mastery)
display(widgets.VBox(list(mastery_sliders.values())), mastery_button, mastery_output)
"""
    ),
    md(
        """
## 四、八个阶段的通关证据

| 阶段 | 不能只会 | 必须提交的证据 |
|---|---|---|
| 声音与特征 | 会调用 `librosa` | 能从公式解释 dB、画波形/频谱/声谱图，并预测参数变化 |
| 张量与编码 | 能跑通模型 | 能手算 shape/length/mask，写出带长度传播的编码器 |
| CTC | 会调用 `CTCLoss` | 能手推小例子、写前向算法与 prefix beam、解释 blank/repeat |
| 流式 ASR | 把音频切块 | 能证明离线/流式一致性，管理 cache/endpoint/PGS，测 RTF 与 P99 |
| LM 与 WFST | 知道缩写 | 能画 L/G 小图、组合并追踪一条路径、调 LM/热词权重 |
| 量化与部署 | 成功导出 ONNX | 能验证数值/状态一致性，给出质量—延迟—内存曲线和服务契约 |
| 音频前端 | 会降噪 | 能定位 clipping/DC/回声/VAD 边界，并用 CER/WER 判断前端是否有益 |
| 后处理语义 | 文本变得好看 | 能保留证据与 N-best，校准置信度，对高风险槽位触发确认 |

每阶段至少完成一次“从空白实现”和一次“故意制造错误再定位”。
"""
    ),
    md(
        """
## 五、阶段综合任务（不要看答案）

1. **声音与特征**：录制或选取两段不同声音，比较波形、RMS/dBFS、频谱、STFT、Log-Mel；解释每张图能回答和不能回答的问题。
2. **张量与编码**：把三段不等长 Log-Mel 组成 batch，手算下采样后的 length，证明 padding 不影响有效输出。
3. **CTC**：对一个 5 帧、2 标签例子枚举合法路径，再用动态规划验证总概率；实现 greedy 与 prefix beam 并构造反例。
4. **流式 ASR**：随机切分同一音频，证明不同 chunk 边界产生相同特征帧与最终文本；输出首字延迟、最终延迟、RTF、P50/P95/P99。
5. **LM 与 WFST**：构造包含同音候选和领域热词的小图，逐项改变声学分、LM 权重和热词权重，画决策边界。
6. **量化与部署**：比较 PyTorch、ONNX FP32、INT8 的误差、模型大小、吞吐和延迟；用两个并发 WebSocket 会话验证状态隔离。
7. **音频前端**：构造干净、噪声、削波、DC、回声五类输入，输出质量指标与 ASR 指标，说明哪些前端处理反而伤害识别。
8. **后处理语义**：让系统处理姓名、否定、日期、金额和号码；保留 spoken form、normalized form、confidence、evidence 和确认策略。
"""
    ),
    md(
        """
## 六、保存个人进度

下面按钮只在本地生成 `learning_progress.json`，该文件默认不提交 Git。保存前请先完成诊断并更新掌握度。
"""
    ),
    code(
        """
save_button = widgets.Button(description="保存本地学习进度", button_style="success")
save_output = widgets.Output()

def save_progress(_):
    with save_output:
        clear_output(wait=True)
        root = Path.cwd()
        if not (root / "pyproject.toml").exists() and (root.parent / "pyproject.toml").exists():
            root = root.parent
        payload = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "diagnostic": quiz_result,
            "mastery": {stage: slider.value for stage, slider in mastery_sliders.items()},
        }
        target = root / "learning_progress.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已保存：{target}")
        if not quiz_result:
            print("提醒：你还没有提交本次闭卷诊断。")

save_button.on_click(save_progress)
display(save_button, save_output)
"""
    ),
    md(
        """
## 七、今天就这样开始

如果你是第一次系统学习：

1. 完成上面的 24 题闭卷诊断；
2. 不管总分多少，从最早低于 2/3 的阶段开始；
3. 若“声音与特征”低于 2/3，打开第 1 课；
4. 学完一课后，用一句话回答：**输入是什么、输出是什么、单位是什么、shape 是什么、改变一个参数会怎样、最常见错误是什么？**
5. 把错题和实验结论写进 `LEARNING_LOG.md`，明天闭卷复述。

不要追求一天看很多课。每次能闭卷画出一张图、写出一个函数、解释一个失败案例，进步才是可验证的。
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "course": {"kind": "learning-hub", "version": 1},
    },
)

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUT)
print(f"wrote {OUT}")
