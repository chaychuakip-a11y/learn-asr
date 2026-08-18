from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

from notebook_layout import executed_path, sanitize_notebook_outputs


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
    question: str
    transfer: str
    sources: tuple[tuple[str, str], ...]
    cells: list
    challenge: str
    next_step: str


def intro(lesson: Lesson) -> list:
    sources = "\n".join(f"- [{name}]({url})" for name, url in lesson.sources)
    return [
        md(
            f"""
# 研究与工程思维 {lesson.number}/6：{lesson.title}

这不是一节“记术语”的课，而是一节**改变判断过程**的实验课。

| 项目 | 内容 |
|---|---|
| 核心问题 | {lesson.question} |
| 迁移价值 | {lesson.transfer} |
| 建议投入 | 90～150 分钟；先预测，再运行，再保留被推翻的判断 |
| 通关证据 | 能把结论写成“主张—证据—反证—边界—下一步” |

固定闭环：

```text
观察（发生了什么） → 假设（可能为什么） → 区分性预测 → 最小实验
        ↑                                      ↓
        └──── 更新置信度、记录反例、决定下一步 ────┘
```

**观察不是原因，总分不是解释，相关不是干预效果，运行成功不是结论成立。**
"""
        ),
        md(
            f"""
## 课前预测：先暴露自己的判断规则

1. 用一句话回答：{lesson.question}
2. 写出你最可能犯的判断错误，例如“只看平均值”或“看到相关就认定因果”。
3. 为本课写一个可被数据推翻的预测；不要写“应该会更好”这种没有阈值的话。
4. 写出什么结果会让你改变主意。

完成实验后回来修正。保留原答案，因为“怎样改主意”本身就是思维能力证据。
"""
        ),
        md(
            f"""
## 一手资料与课程取舍

{sources}

课程把这些资料转成小型、确定性、可运行的 ASR 实验。示例数据用于理解方法，不代表真实产品结论。
"""
        ),
    ]


def finish(lesson: Lesson) -> list:
    return [
        md(
            f"""
## 闭卷挑战

{lesson.challenge}

回答时强制使用下面的证据卡：

```text
主张：
证据：
最强替代解释：
什么结果会推翻主张：
适用边界：
下一步最小实验：
```
"""
        ),
        md(
            f"""
## 最小掌握门禁

- [ ] 我在运行前写了方向和数量级预测。
- [ ] 我能指出示例结论中至少一个替代解释。
- [ ] 我能从空白重写本课核心函数，并用边界输入测试。
- [ ] 我能说明“没有发现差异”和“证明没有差异”的区别。
- [ ] 我把一次被数据推翻的判断写入 `LEARNING_LOG.md`。
- [ ] 我能把本课方法迁移到一个非 ASR 问题。

下一步：{lesson.next_step}
"""
        ),
    ]


LESSONS = [
    Lesson(
        1,
        "从现象到可证伪假设",
        "从现象到可证伪假设",
        "识别率下降时，怎样避免凭第一印象修错地方？",
        "适用于线上故障、性能回退、学习困难和任何‘为什么’问题。",
        (
            ("NIST：实验设计的定义与目标", "https://itl.nist.gov/div898/handbook/pri/section1/pri11.htm"),
            ("NIST：随机化、重复与设计原则", "https://www.itl.nist.gov/div898/handbook/pmd/section3/pmd33.htm"),
        ),
        [
            md(
                """
## 1. 把五种句子分开

| 类型 | 例子 | 能否直接推出原因 |
|---|---|---|
| 观察 | 噪声切片 WER 从 12% 变成 25% | 不能 |
| 假设 | 新降噪器损坏了辅音 | 不能，等待检验 |
| 预测 | 若假设成立，关闭降噪器后噪声切片至少改善 5 个百分点 | 可以检验 |
| 结果 | 配对样本改善 1.1 个百分点，区间跨 0 | 证据不足 |
| 决策 | 暂不发布，先检查重采样与 VAD | 还包含成本和风险 |

“系统变差了，因为最近换了前端”把时间先后误当成原因。先列竞争假设，再找能让它们给出不同预测的测试。
"""
            ),
            code(
                """
from dataclasses import dataclass

@dataclass(frozen=True)
class Hypothesis:
    name: str
    prediction: str
    falsifier: str
    test_cost: float
    discrimination: float
    coverage: float

hypotheses = [
    Hypothesis("降噪损伤", "关闭降噪后 noisy WER 改善 >= 5pp", "改善 < 1pp", 2, 0.90, 0.65),
    Hypothesis("采样率误配", "8 kHz 输入的频带/时长合同异常", "全部输入合同一致", 1, 0.95, 0.40),
    Hypothesis("VAD 截断", "删除错误集中在句首/句尾", "边界与中间删除率相同", 1, 0.80, 0.55),
    Hypothesis("说话人偏移", "新说话人切片独立变差", "同说话人配对也退化", 3, 0.70, 0.80),
]

def test_value(h: Hypothesis) -> float:
    # 区分力和覆盖越高越好，成本越低越好；只是排序启发式，不是概率。
    if h.test_cost <= 0:
        raise ValueError("test_cost must be positive")
    return h.discrimination * h.coverage / h.test_cost

for h in sorted(hypotheses, key=test_value, reverse=True):
    print(f"{h.name:8s} value={test_value(h):.3f} | {h.prediction}")
"""
            ),
            md(
                """
## 2. 好假设必须冒险

“可能是数据问题”几乎不会失败，所以信息量很低。把它改写成：

> 若主要原因是 VAD 截断，那么新版相对旧版新增的删除错误中，至少 60% 位于首尾 300 ms；在关闭 VAD、其余条件固定的配对实验中差异应显著缩小。

它包含对象、方向、阈值、对照和反证。阈值应在看结果前写；看完数据再移动门槛叫事后合理化。
"""
            ),
            code(
                """
records = [
    {"slice": "clean", "position": "middle", "old": 1, "new": 1},
    {"slice": "clean", "position": "edge",   "old": 0, "new": 1},
    {"slice": "noisy", "position": "middle", "old": 2, "new": 3},
    {"slice": "noisy", "position": "edge",   "old": 1, "new": 5},
    {"slice": "noisy", "position": "edge",   "old": 2, "new": 6},
]

extra_by_position = {}
for row in records:
    extra = max(0, row["new"] - row["old"])
    extra_by_position[row["position"]] = extra_by_position.get(row["position"], 0) + extra

edge_fraction = extra_by_position["edge"] / sum(extra_by_position.values())
print("新增错误按位置:", extra_by_position)
print(f"首尾占比={edge_fraction:.1%}")
print("预测通过?", edge_fraction >= 0.60)
assert edge_fraction >= 0.60
"""
            ),
            md(
                """
## 3. 证据强度阶梯

```text
传闻 < 单个例子 < 同分布统计 < 固定样本配对对照
     < 随机化/重复实验 < 独立复现 < 目标场景持续监控
```

强证据也只支持特定范围。一次 8 kHz 中文短句实验不能推出所有语言、设备和长音频都成立。

### 练习：先不要运行答案

把“幅值低于 1000，所以声音小”改写为至少三个竞争假设，并为每个假设写一个区分性测试。必须考虑位深/dtype、RMS 与 peak、麦克风增益或标定基准。
"""
            ),
            code(
                """
def validate_hypothesis_card(card: dict) -> list[str]:
    required = ["observation", "hypothesis", "prediction", "threshold", "falsifier", "scope"]
    missing = [key for key in required if not str(card.get(key, "")).strip()]
    return missing

example = {
    "observation": "int16 文件的 peak=850",
    "hypothesis": "录音链路增益过低",
    "prediction": "同设备同距离的校准音 RMS 比基准低至少 12 dB",
    "threshold": "delta_rms_db <= -12",
    "falsifier": "换算到 dBFS 并校准后与基准差小于 3 dB",
    "scope": "当前设备、距离和 int16 PCM；不外推到 float 音频",
}
print("缺失字段:", validate_hypothesis_card(example))
assert validate_hypothesis_card(example) == []
"""
            ),
        ],
        "选一个你最近遇到的技术问题，写 3 个竞争假设。设计一个成本不超过 10 分钟、却能最大程度区分它们的实验；解释为什么它比直接改代码更有信息量。",
        "第 2 课把‘平均 WER’拆成错误类型与数据切片。",
    ),
    Lesson(
        2,
        "错误分类切片与辛普森悖论",
        "错误分类、切片与辛普森悖论",
        "总 WER 看起来不错时，系统可能在哪些人和场景上失败？",
        "适用于日志分析、质量看板、用户反馈和任何聚合指标。",
        (
            ("NIST SCTK：按说话人、句子和标签切片报告", "https://github.com/usnistgov/SCTK/blob/master/doc/options.htm"),
            ("Model Cards：按相关条件和群体报告性能", "https://research.google/pubs/model-cards-for-model-reporting/"),
            ("NIST AI RMF：部署场景与分解评估", "https://airc.nist.gov/airmf-resources/playbook/measure/"),
        ),
        [
            md(
                """
## 1. WER 是入口，不是终点

`WER = (S + D + I) / N`。相同的 10% WER 可能是：专有名词替换、整句幻觉插入、句尾被 VAD 删除，或少数说话人完全不可用。修复手段和风险完全不同。
"""
            ),
            code(
                """
def align_words(reference: str, hypothesis: str):
    ref, hyp = reference.split(), hypothesis.split()
    n, m = len(ref), len(hyp)
    dp = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0, [])  # cost, S, D, I, operations
    for i in range(1, n + 1):
        cost, s, d, ins, ops = dp[i-1][0]
        dp[i][0] = (cost+1, s, d+1, ins, ops+[("D", ref[i-1], "*")])
    for j in range(1, m + 1):
        cost, s, d, ins, ops = dp[0][j-1]
        dp[0][j] = (cost+1, s, d, ins+1, ops+[("I", "*", hyp[j-1])])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = []
            prev = dp[i-1][j-1]
            if ref[i-1] == hyp[j-1]:
                candidates.append((prev[0], prev[1], prev[2], prev[3], prev[4]+[("C", ref[i-1], hyp[j-1])]))
            else:
                candidates.append((prev[0]+1, prev[1]+1, prev[2], prev[3], prev[4]+[("S", ref[i-1], hyp[j-1])]))
            prev = dp[i-1][j]
            candidates.append((prev[0]+1, prev[1], prev[2]+1, prev[3], prev[4]+[("D", ref[i-1], "*")]))
            prev = dp[i][j-1]
            candidates.append((prev[0]+1, prev[1], prev[2], prev[3]+1, prev[4]+[("I", "*", hyp[j-1])]))
            dp[i][j] = min(candidates, key=lambda item: item[:4])
    return {"N": n, "S": dp[n][m][1], "D": dp[n][m][2], "I": dp[n][m][3], "ops": dp[n][m][4]}

example = align_words("播放 周杰伦 的 稻香", "播放 周杰论 稻香")
print(example)
print("WER=", (example["S"] + example["D"] + example["I"]) / example["N"])
assert (example["S"], example["D"], example["I"]) == (1, 1, 0)
"""
            ),
            md(
                """
## 2. 同一个总分下面有不同世界

下面的教学记录含设备、噪声、说话人和错误数。先预测哪个切片最差，再计算。小切片必须同时报告分母，不能只报百分比。
"""
            ),
            code(
                """
rows = [
    ("s1", "headset", "clean", 12, 0), ("s1", "farfield", "noisy", 10, 4),
    ("s2", "headset", "clean", 11, 1), ("s2", "farfield", "noisy", 9, 3),
    ("s3", "headset", "noisy", 13, 2), ("s3", "farfield", "clean", 10, 2),
    ("s4", "headset", "clean", 8, 0), ("s4", "farfield", "noisy", 12, 5),
]

def grouped_wer(rows, field_index):
    totals = {}
    for row in rows:
        key, words, errors = row[field_index], row[3], row[4]
        totals.setdefault(key, [0, 0])
        totals[key][0] += errors
        totals[key][1] += words
    return {key: {"errors": e, "words": n, "wer": e/n} for key, (e, n) in totals.items()}

print("overall:", sum(r[4] for r in rows) / sum(r[3] for r in rows))
print("device:", grouped_wer(rows, 1))
print("noise:", grouped_wer(rows, 2))
print("speaker:", grouped_wer(rows, 0))
"""
            ),
            md(
                """
## 3. 辛普森悖论：混合比例能翻转结论

下例中 B 在 clean 和 noisy 内都优于 A，但如果 A 主要在简单样本测试、B 主要在困难样本测试，跨不同测试集的总分会错误地宣称 A 更好。

这不是说“固定同一测试集的模型比较也会凭空翻转”。恰恰相反：它说明**比较必须使用相同样本，或按相同目标分布重新加权**。
"""
            ),
            code(
                """
systems = {
    "A": {"clean": (900, 0.05), "noisy": (100, 0.30)},
    "B": {"clean": (100, 0.04), "noisy": (900, 0.25)},
}

def mixed_wer(spec):
    return sum(n * rate for n, rate in spec.values()) / sum(n for n, _ in spec.values())

for name, spec in systems.items():
    print(name, "总WER", f"{mixed_wer(spec):.1%}", "分层", {k: v[1] for k, v in spec.items()})

target_weights = {"clean": 0.6, "noisy": 0.4}
for name, spec in systems.items():
    standardized = sum(target_weights[k] * spec[k][1] for k in target_weights)
    print(name, "按同一部署分布标准化", f"{standardized:.1%}")

assert mixed_wer(systems["A"]) < mixed_wer(systems["B"])
assert all(systems["B"][k][1] < systems["A"][k][1] for k in target_weights)
"""
            ),
            md(
                """
## 4. 错误频率不等于修复优先级

优先级还取决于影响：把“打开窗帘”识别错通常比把“转账一万元”识别错风险低。可以先用透明的启发式：

`priority = affected_users × error_rate × severity × fixability`

这不是客观真理；它迫使团队公开价值判断，并保留每个因子的依据。
"""
            ),
            code(
                """
issues = [
    {"name": "远场删除", "users": 5000, "rate": .18, "severity": 2, "fixability": .7},
    {"name": "金额误识别", "users": 400, "rate": .04, "severity": 10, "fixability": .8},
    {"name": "专名替换", "users": 2400, "rate": .11, "severity": 4, "fixability": .9},
]
for issue in issues:
    issue["priority"] = issue["users"] * issue["rate"] * issue["severity"] * issue["fixability"]
for issue in sorted(issues, key=lambda x: x["priority"], reverse=True):
    print(issue["name"], round(issue["priority"], 1))
"""
            ),
        ],
        "构造一个总体 WER 相同、但 S/D/I 结构和业务风险完全不同的两系统例子。再写一张最小切片表：场景、设备、说话人、长度、SNR、分母、误差率、风险。",
        "第 3 课学习怎样用对照、随机化和因子实验区分原因。",
    ),
    Lesson(
        3,
        "对照实验消融混杂与交互",
        "对照实验、消融、混杂与交互",
        "改了很多东西并且分数上涨，怎样知道究竟什么有效？",
        "适用于调参、重构、A/B 测试、音频处理链和性能优化。",
        (
            ("NIST：为什么单因素逐次优化会漏掉交互", "https://www.itl.nist.gov/div898/handbook/pri/section2/pri212.htm"),
            ("NIST：如何选择实验设计", "https://www.itl.nist.gov/div898/handbook/pri/section3/pri3.htm"),
            ("NIST：随机化与重复的作用", "https://www.itl.nist.gov/div898/handbook/pmd/section3/pmd33.htm"),
        ),
        [
            md(
                """
## 1. 公平对照的最低合同

只改变计划研究的因素，并固定：数据与划分、文本规范化、随机种子策略、训练预算、解码参数、硬件/线程、评分代码。若无法固定，就记录为阻断因素或协变量。

“新版模型 + 新前端 + 新 LM 在新测试集上更好”没有可归因性。它可以是有用的工程候选，但不是组件有效性的证据。
"""
            ),
            code(
                """
from itertools import product

# 两因素全因子：前端 F、语言模型 L。数值为教学 WER（越低越好）。
wer = {
    (0, 0): 20.0,
    (1, 0): 17.0,
    (0, 1): 18.0,
    (1, 1): 11.0,
}

for f, l in product([0, 1], repeat=2):
    print(f"frontend={f}, lm={l}, WER={wer[(f,l)]:.1f}%")

frontend_effect_without_lm = wer[(1, 0)] - wer[(0, 0)]
frontend_effect_with_lm = wer[(1, 1)] - wer[(0, 1)]
interaction = frontend_effect_with_lm - frontend_effect_without_lm
print("前端在无LM时的变化:", frontend_effect_without_lm, "pp")
print("前端在有LM时的变化:", frontend_effect_with_lm, "pp")
print("交互项:", interaction, "pp")
assert interaction == -4.0
"""
            ),
            md(
                """
## 2. 为什么“每次只改一个变量”不是完整规则

排错时，一次改变一个变量有利于归因；但探索多个因素时，纯 OFAT 会漏掉交互。上例的前端与 LM 一起使用有额外收益。正确升级是：

- 小范围定位：最小配对对照；
- 多因素筛选：全因子或设计良好的部分因子；
- 存在批次/说话人差异：分块；
- 存在时间漂移：随机化运行顺序并重复。
"""
            ),
            code(
                """
import numpy as np

rng = np.random.default_rng(7)
conditions = np.array([0] * 10 + [1] * 10)  # 0=baseline, 1=candidate
time_drift = np.linspace(0, 4, len(conditions))
true_effect = -1.5

def observed_difference(order):
    y = 20 + true_effect * conditions[order] + time_drift + rng.normal(0, .15, len(order))
    assigned = conditions[order]
    return y[assigned == 1].mean() - y[assigned == 0].mean()

blocked_order = np.arange(20)  # baseline 全在前，candidate 全在后：与漂移混杂
random_order = np.random.default_rng(3).permutation(20)
print("未随机化估计:", round(observed_difference(blocked_order), 2), "pp")
print("随机化估计:", round(observed_difference(random_order), 2), "pp")
print("真实干预效应:", true_effect, "pp")
"""
            ),
            md(
                """
## 3. 消融表必须回答机制问题

| 实验 | 前端 | LM | 训练预算 | 测试集 | 目的 |
|---|---:|---:|---:|---|---|
| E0 | 0 | 0 | 固定 | 固定 | 最弱合理基线 |
| E1 | 1 | 0 | 固定 | 固定 | 前端主效应 |
| E2 | 0 | 1 | 固定 | 固定 | LM 主效应 |
| E3 | 1 | 1 | 固定 | 固定 | 完整系统与交互 |

若完整系统只赢 0.2pp，却多 40% 延迟，结论不能只写“最优”。还需要统计区间和工程约束。
"""
            ),
            code(
                """
def audit_experiment(base: dict, candidate: dict, intended_factor: str):
    keys = sorted(set(base) | set(candidate))
    changed = [key for key in keys if base.get(key) != candidate.get(key)]
    confounds = [key for key in changed if key != intended_factor]
    return changed, confounds

base = {"frontend": "v1", "test": "fixed", "beam": 10, "threads": 1, "seed": 7}
candidate = {"frontend": "v2", "test": "new", "beam": 20, "threads": 1, "seed": 7}
changed, confounds = audit_experiment(base, candidate, "frontend")
print("发生变化:", changed)
print("混杂因素:", confounds)
assert confounds == ["beam", "test"]
"""
            ),
        ],
        "为“降噪、VAD、LM”设计一个 2×2×2 实验。写出响应变量、硬约束、随机化/分块方法、重复次数和交互项；再说明如果只能跑 4 次，你愿意牺牲什么结论。",
        "第 4 课学习点估计、Bootstrap 配对区间和置信度校准。",
    ),
    Lesson(
        4,
        "不确定性Bootstrap与配对比较",
        "不确定性、Bootstrap 与配对比较",
        "WER 从 10.0% 降到 9.7%，这是改进还是抽样波动？",
        "适用于模型比较、线上指标、个人学习测验和任何有限样本结论。",
        (
            ("SciPy：Bootstrap 置信区间", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html"),
            ("NIST SCTK：标准 ASR 评分与置信报告", "https://github.com/usnistgov/SCTK/blob/master/doc/sclite.htm"),
            ("Guo 等：现代神经网络置信度校准", "https://proceedings.mlr.press/v70/guo17a.html"),
        ),
        [
            md(
                """
## 1. 先明确统计单位

一句话内的词高度相关，同一说话人的多句话也相关。把 1000 个词假装成 1000 个独立样本会得到过窄区间。部署目标若是“新说话人”，应优先按说话人重采样；目标若是“这些固定说话人的新句子”，统计单位会不同。
"""
            ),
            code(
                """
import numpy as np

rng = np.random.default_rng(42)
rows = []
for speaker in range(12):
    difficulty = rng.uniform(0.04, 0.20)
    for utterance in range(8):
        words = int(rng.integers(8, 18))
        err_a = int(rng.binomial(words, difficulty))
        # B 平均略好，但不是每个说话人/句子都赢
        err_b = int(rng.binomial(words, max(0.01, difficulty - 0.018)))
        rows.append({"speaker": speaker, "words": words, "A": err_a, "B": err_b})

def micro_wer(data, system):
    return sum(row[system] for row in data) / sum(row["words"] for row in data)

print("A WER", f"{micro_wer(rows, 'A'):.2%}")
print("B WER", f"{micro_wer(rows, 'B'):.2%}")
print("观察差值 B-A", f"{micro_wer(rows, 'B')-micro_wer(rows, 'A'):+.2%}")
"""
            ),
            md(
                """
## 2. 配对 Cluster Bootstrap

两个系统必须在同一重采样中使用同一批说话人，这保留配对关系。每轮有放回抽取说话人，连同其全部句子，计算 `WER_B - WER_A`。

区间跨 0 不等于“两个系统完全一样”，只表示当前数据和方法不足以排除零差异。区间窄且整体落在业务最小改进阈值之外，才更有决策力。
"""
            ),
            code(
                """
def paired_speaker_bootstrap(data, n_resamples=4000, seed=0):
    speakers = sorted({row["speaker"] for row in data})
    by_speaker = {s: [row for row in data if row["speaker"] == s] for s in speakers}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_resamples):
        sampled = rng.choice(speakers, size=len(speakers), replace=True)
        sample_rows = [row for s in sampled for row in by_speaker[int(s)]]
        differences.append(micro_wer(sample_rows, "B") - micro_wer(sample_rows, "A"))
    return np.asarray(differences)

boot = paired_speaker_bootstrap(rows)
low, high = np.quantile(boot, [0.025, 0.975])
observed = micro_wer(rows, "B") - micro_wer(rows, "A")
print(f"差值={observed:+.2%}, 95% percentile CI=[{low:+.2%}, {high:+.2%}]")
print("B 更好的 bootstrap 比例:", f"{np.mean(boot < 0):.1%}")
assert len(boot) == 4000
"""
            ),
            md(
                """
## 3. 统计差异与实际价值是两道门

发布规则示例：

1. 质量：95% 区间上界小于 0（B 大概率不退化）；
2. 价值：差值中位数至少改善 0.5pp；
3. 守门：关键切片不退化超过 1pp；
4. 系统：P95 延迟不超过预算。

样本巨大时 0.05pp 也可能“显著”，却不值得部署；样本很小时 2pp 也可能区间很宽，需要更多数据而不是武断宣布失败。
"""
            ),
            code(
                """
import matplotlib.pyplot as plt

confidence = np.array([.05,.12,.18,.25,.34,.42,.48,.55,.62,.68,.73,.79,.84,.88,.91,.94,.96,.97,.98,.99])
correct = np.array([0,0,0,1,0,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1])

def reliability_bins(confidence, correct, edges=np.linspace(0, 1, 6)):
    result = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            result.append((lo, hi, int(mask.sum()), float(confidence[mask].mean()), float(correct[mask].mean())))
    return result

bins = reliability_bins(confidence, correct)
ece = sum(n/len(confidence) * abs(avg_c-acc) for _,_,n,avg_c,acc in bins)
for item in bins:
    print(f"[{item[0]:.1f},{item[1]:.1f}) n={item[2]:2d} conf={item[3]:.2f} acc={item[4]:.2f}")
print("ECE=", round(ece, 3))

plt.plot([0,1], [0,1], "k--", label="perfect calibration")
plt.scatter([x[3] for x in bins], [x[4] for x in bins], s=[20*x[2] for x in bins])
plt.xlabel("mean confidence"); plt.ylabel("empirical accuracy"); plt.legend(); plt.grid(True); plt.show()
"""
            ),
            md(
                """
## 4. 置信度必须回答“100 次中对多少次”

准确率和校准是不同轴：模型可以准确但过度自信，也可以准确率一般但置信度诚实。ECE 依赖分箱，不能单独作为证明；还要看 reliability diagram、NCE/Brier、关键阈值附近和分布外切片。
"""
            ),
        ],
        "选两组同一样本上的系统输出，从空白实现按说话人配对 Bootstrap。报告点估计、95% 区间、最小有意义改进和关键切片守门；禁止只写‘显著/不显著’。",
        "第 5 课用不变量、变形测试和反事实更快定位根因。",
    ),
    Lesson(
        5,
        "因果排错不变量与反事实",
        "因果排错、不变量与反事实",
        "面对十几个可能原因，怎样用最少实验定位故障阶段？",
        "适用于数据管线、训练、流式状态、服务契约和普通软件调试。",
        (
            ("NIST：测量系统与过程稳定性是实验前提", "https://www.itl.nist.gov/div898/handbook/pri/pri.htm"),
            ("NIST AI RMF：测试、监控与可追溯测量", "https://airc.nist.gov/airmf-resources/airmf/5-sec-core/"),
        ),
        [
            md(
                """
## 1. 先画边界，再猜原因

```text
WAV 字节 → 解码数组 → 重采样/归一化 → 特征 → 模型 logits
        → 解码状态 → 文本规范化 → API 响应
```

在每个边界检查合同：shape、dtype、单位、范围、长度、状态归属、版本。目标不是多打印日志，而是找到**最后一个正确边界**和**第一个错误边界**。
"""
            ),
            code(
                """
cases = {
    "healthy":    [1, 1, 1, 1, 1, 1],
    "bad_wav":    [0, 0, 0, 0, 0, 0],
    "resample":   [1, 0, 0, 0, 0, 0],
    "feature":    [1, 1, 0, 0, 0, 0],
    "model":      [1, 1, 1, 0, 0, 0],
    "decoder":    [1, 1, 1, 1, 0, 0],
    "postprocess": [1, 1, 1, 1, 1, 0],
}
stages = ["decode", "frontend", "features", "model", "decoder", "postprocess"]

def first_failed_boundary(checks):
    for stage, passed in zip(stages, checks):
        if not passed:
            return stage
    return None

for name, checks in cases.items():
    print(f"{name:12s} -> {first_failed_boundary(checks)}")
assert first_failed_boundary(cases["decoder"]) == "decoder"
"""
            ),
            md(
                """
## 2. 不变量：不依赖具体答案的强断言

- 增加 batch padding 不应改变有效帧 logits；
- 把同一音频切成不同合法 chunk，最终文本应一致或满足明确误差界；
- 保存再加载 checkpoint，固定输入输出应一致；
- 对 peak-normalized 前端，输入乘正增益后特征应近似不变；
- 两个 WebSocket 会话的 cache 必须隔离。

当不知道正确转写时，这些关系仍然可测，称为变形测试（metamorphic test）。
"""
            ),
            code(
                """
import numpy as np

def normalized_energy(x):
    x = np.asarray(x, dtype=float)
    peak = np.max(np.abs(x))
    if peak == 0:
        return 0.0
    y = x / peak
    return float(np.mean(y ** 2))

x = np.array([0.1, -0.4, 0.2, 0.3])
for gain in [0.25, 1.0, 4.0]:
    print(gain, normalized_energy(gain * x))
assert np.isclose(normalized_energy(x), normalized_energy(4*x))

# 反例：加入硬剪切后，“只是增益变化”的关系被破坏。
clipped = np.clip(4*x, -0.5, 0.5)
print("clipped energy", normalized_energy(clipped))
assert not np.isclose(normalized_energy(x), normalized_energy(clipped))
"""
            ),
            md(
                """
## 3. 反事实测试要让竞争原因给出不同答案

现象：流式输出重复词。

| 假设 | 最小干预 | 预测 |
|---|---|---|
| CTC 状态未跨 chunk | 保持 chunk，只修复 last-token 状态 | 重复消失 |
| 音频重叠送入 | 记录样本区间并去重 | 重复消失 |
| PGS 应用器重复 apd | 固定 token 流，只替换事件应用器 | 重复消失 |

同时改三处即使修好了，也不知道根因。先保存能稳定复现的最小输入，再逐边界替换为可信参照实现。
"""
            ),
            code(
                """
fault_signatures = {
    "sample_rate": {"duration_wrong", "frequency_scaled", "all_modes_fail"},
    "vad_cut": {"edge_deletions", "offline_ok", "short_utterance_bad"},
    "cache_leak": {"second_session_bad", "restart_fixes", "offline_ok"},
    "decoder_state": {"boundary_repeats", "small_chunks_worse", "offline_ok"},
}

observed = {"boundary_repeats", "small_chunks_worse", "offline_ok"}

def rank_causes(observed, signatures):
    ranked = []
    for cause, expected in signatures.items():
        overlap = len(observed & expected)
        contradictions = len(observed - expected)
        score = overlap - 0.5 * contradictions
        ranked.append((score, cause, observed & expected, observed - expected))
    return sorted(ranked, reverse=True)

for score, cause, matches, misses in rank_causes(observed, fault_signatures):
    print(f"{cause:14s} score={score:+.1f} matches={sorted(matches)} unexplained={sorted(misses)}")
"""
            ),
            md(
                """
## 4. 修复后必须有三份证据

1. 原最小复现从红变绿；
2. 新增回归测试能在旧实现上失败；
3. 邻近合同没有被破坏。

“重启后好了”是线索，不是修复；“调大阈值后不报错”可能只是隐藏症状。
"""
            ),
        ],
        "为一个‘离线正确、流式偶发重复’问题画 6 个边界，写 4 个竞争原因、每个原因的独特预测，以及能用最少运行次数定位根因的检查顺序。",
        "第 6 课把质量、延迟、成本、隐私和风险放进同一决策。",
    ),
    Lesson(
        6,
        "Pareto风险与证据化决策",
        "Pareto、风险与证据化决策",
        "没有一个方案全面最好时，怎样做可解释、可复盘的选择？",
        "适用于模型选型、上线门禁、资源规划、架构决策和个人学习路线。",
        (
            ("NIST AI RMF：风险、测量与权衡", "https://airc.nist.gov/airmf-resources/airmf/5-sec-core/"),
            ("Model Cards：用途、切片、限制与透明报告", "https://research.google/pubs/model-cards-for-model-reporting/"),
            ("Datasheets for Datasets：数据动机、组成、采集和用途", "https://www.microsoft.com/en-us/research/publication/datasheets-for-datasets/"),
        ),
        [
            md(
                """
## 1. 先硬约束，再 Pareto，再偏好

1. **硬约束**：隐私、P95 延迟、安全误接受率、内存上限；不满足就淘汰。
2. **Pareto 前沿**：若方案 X 在所有目标不差且至少一项更好，X 支配 Y。
3. **偏好权重**：只在未被支配且满足约束的候选之间表达业务取舍。
4. **敏感性分析**：权重轻微变化就翻转时，决策应标注脆弱，而不是伪装成唯一答案。
"""
            ),
            code(
                """
candidates = [
    {"name": "tiny-local",  "wer": 14.0, "p95_ms": 90,  "cost": 1.0, "risk": 1.5, "private": True},
    {"name": "base-local",  "wer": 10.5, "p95_ms": 180, "cost": 2.2, "risk": 1.2, "private": True},
    {"name": "large-cloud", "wer": 8.8,  "p95_ms": 420, "cost": 5.5, "risk": 2.8, "private": False},
    {"name": "hybrid",      "wer": 9.6,  "p95_ms": 240, "cost": 3.1, "risk": 1.4, "private": True},
    {"name": "slow-local",  "wer": 11.2, "p95_ms": 310, "cost": 2.8, "risk": 1.8, "private": True},
]

feasible = [c for c in candidates if c["private"] and c["p95_ms"] <= 250]
print("满足硬约束:", [c["name"] for c in feasible])

objectives = ["wer", "p95_ms", "cost", "risk"]  # 全部越低越好
def dominates(a, b):
    return all(a[k] <= b[k] for k in objectives) and any(a[k] < b[k] for k in objectives)

frontier = [c for c in feasible if not any(dominates(other, c) for other in feasible if other is not c)]
print("Pareto 前沿:", [c["name"] for c in frontier])
assert "slow-local" not in frontier
"""
            ),
            md(
                """
## 2. 加权分数会隐藏价值判断

归一化方式、目标方向和权重都会改变排名。必须把原始指标与硬约束一起展示，不能只发布一个“综合分”。
"""
            ),
            code(
                """
def minmax_scores(items, weights):
    bounds = {k: (min(x[k] for x in items), max(x[k] for x in items)) for k in weights}
    result = []
    for item in items:
        score = 0.0
        for key, weight in weights.items():
            lo, hi = bounds[key]
            normalized_badness = 0.0 if hi == lo else (item[key] - lo) / (hi - lo)
            score += weight * normalized_badness
        result.append((score, item["name"]))
    return sorted(result)

quality_first = {"wer": .60, "p95_ms": .20, "cost": .10, "risk": .10}
latency_first = {"wer": .25, "p95_ms": .50, "cost": .15, "risk": .10}
print("质量优先:", minmax_scores(frontier, quality_first))
print("延迟优先:", minmax_scores(frontier, latency_first))
"""
            ),
            md(
                """
## 3. 风险不是错误率的同义词

NIST 将风险关联到事件发生可能性与影响。工程中还常显式考虑暴露量：

`risk priority = likelihood × impact × exposure`

它适合排序，不应伪装成精确概率。高风险语音操作还需要安全回退：确认、拒绝、人工接管、审计日志和可回滚版本。
"""
            ),
            code(
                """
risks = [
    {"name": "音乐请求误识别", "likelihood": .08, "impact": 1, "exposure": 10000},
    {"name": "金额槽位误识别", "likelihood": .01, "impact": 10, "exposure": 1500},
    {"name": "医疗否定词删除", "likelihood": .004, "impact": 10, "exposure": 5000},
]
for item in risks:
    item["priority"] = item["likelihood"] * item["impact"] * item["exposure"]
for item in sorted(risks, key=lambda x: x["priority"], reverse=True):
    print(item["name"], item["priority"])
"""
            ),
            md(
                """
## 4. 最终产物不是排行榜，而是决策记录

```text
目标场景与非目标场景：
候选与最弱合理基线：
硬约束及来源：
同一固定测试集上的原始指标、切片、区间：
Pareto 前沿：
风险、失败回退、监控与回滚：
选择及被放弃方案：
最强反对意见：
触发重新评估的条件：
```

数据表回答“数据为什么存在、由什么组成、怎样采集、适合/不适合什么”；模型卡回答“模型在什么条件下测过、表现和限制是什么”。两者让后来者能审计今天的选择。
"""
            ),
            code(
                """
required_decision_fields = {
    "context", "baseline", "constraints", "metrics", "slices", "uncertainty",
    "risks", "fallback", "rollback", "decision", "dissent", "revisit_trigger"
}

def audit_decision_record(record):
    missing = sorted(required_decision_fields - set(record))
    placeholders = {"", "todo", "tbd", "待补充", "未知"}
    empty = sorted(
        key for key in required_decision_fields & set(record)
        if str(record[key]).strip().lower() in placeholders
    )
    return {"missing": missing, "empty": empty, "ready": not missing and not empty}

draft = {key: "TODO" for key in required_decision_fields}
draft["decision"] = "base-local：满足隐私/延迟门禁，质量与成本处于可接受前沿"
print(audit_decision_record(draft))
assert not audit_decision_record(draft)["ready"]
"""
            ),
            md(
                """
## 5. 六课合并成一个思考操作系统

```text
可证伪假设 → 错误分类与切片 → 对照/消融 → 区间与校准
       → 不变量/反事实定位 → 硬约束、Pareto、风险与决策记录
```

遇到新模型时，不先问“是不是更先进”，而依次问：解决哪个可测问题？相比什么基线？在哪些切片？不确定性多大？代价和风险是什么？什么结果会让我改变选择？
"""
            ),
        ],
        "从仓库任一主题中选择两个候选方案，写完整决策记录。至少包含 4 个原始指标、2 个硬约束、3 个风险、Pareto 判断、权重敏感性、失败回退和重新评估触发条件。",
        "回到真实项目，用六课闭环完成一次从问题定义到可审计决策的独立研究。",
    ),
]


def build_notebook(lesson: Lesson):
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "course": {
            "track": "research_thinking",
            "lesson": lesson.number,
            "title": lesson.title,
            "evidence_model": "claim-evidence-falsifier-boundary-next-test",
        },
    }
    notebook.cells = intro(lesson) + lesson.cells + finish(lesson)
    return notebook


def main() -> None:
    for lesson in LESSONS:
        source = NOTEBOOK_DIR / f"思维训练_{lesson.number:02d}_{lesson.slug}.ipynb"
        notebook = build_notebook(lesson)
        nbf.write(notebook, source)

        executed = copy.deepcopy(notebook)
        client = NotebookClient(
            executed,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()
        sanitize_notebook_outputs(executed)
        target = executed_path(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(executed, target)
        print(f"built {source.name} -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
