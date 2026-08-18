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
    sources: tuple[tuple[str, str], ...]
    cells: list
    challenge: str


def intro(lesson: Lesson) -> list:
    sources = "\n".join(f"- [{name}]({url})" for name, url in lesson.sources)
    return [
        md(
            f"""
# 认知拓展 {lesson.number}/8：{lesson.title}

核心问题：**{lesson.question}**

本课不要求背诵模型名称。你要先写预测，再运行小实验，最后说明证据边界。建议投入 60～120 分钟。

固定动作：

观察 → 明确问题 → 选择模型 → 写出预测 → 检查证据 → 寻找反例 → 更新判断

## 课前预测

1. 不查资料，用两句话回答核心问题。
2. 给自己的答案标一个 0%～100% 置信度。
3. 写出一个会使你改变判断的反例。
4. 预测第一个代码实验的方向；运行后不要删除原预测。
"""
        ),
        md(
            f"""
## 一手资料与课程取舍

{sources}

课程只提取能通过小实验理解的部分。代码输出是教学例子，不是对现实世界的自动裁决。
"""
        ),
    ]


def finish(lesson: Lesson) -> list:
    return [
        md(
            """
## 误用警报与适用边界

- 工具是对问题的压缩，不是现实本身；先检查假设，再相信输出。
- 一个漂亮数字不能替代数据来源、测量误差、替代解释和失败代价。
- 不要用术语给别人贴标签；把“某某偏差”改写成可检查的判断步骤。
- 不能量化时可以做定性判断，但必须把不知道什么写出来。

## 迁移练习

对同一个工具各写一个例子：

1. ASR/音频：它能防止哪一种误判？
2. 工作：它能改进哪一个会议、指标或项目决策？
3. 日常生活：它能让哪个选择更可逆、更可检验？

每个例子都要包含：原判断、工具带来的新问题、更新后的行动。
"""
        ),
        md(
            f"""
## 闭卷挑战

{lesson.challenge}

用下面的判断卡作答：

    问题：
    当前主张：
    概率或置信度：
    关键证据：
    最强替代解释：
    什么结果会让我改主意：
    适用边界：
    下一步最小行动：

## 最小掌握门禁

- [ ] 能不用术语解释本课工具。
- [ ] 能从空白重写至少一个核心函数。
- [ ] 能构造一个让直觉失败的反例。
- [ ] 能指出工具本身的一种误用。
- [ ] 能迁移到 ASR、工作、生活三个场景。
- [ ] 已把一条预测和实际结果写入 LEARNING_LOG.md。
"""
        ),
    ]


def logic_cells() -> list:
    return [
        md(
            """
## 1. 先分开：主张、理由、担保和反驳

“新模型分数更高，所以应该上线”至少漏了三层：分数是否来自可比实验；分数是否覆盖真实目标；收益是否超过延迟、成本和风险。有效形式只保证“若前提真，则结论不能假”，不保证前提真的成立。
"""
        ),
        code(
            """
from itertools import product

def implies(a, b):
    return (not a) or b

arguments = {
    "肯定前件": (lambda p, q: [implies(p, q), p], lambda p, q: q),
    "肯定后件": (lambda p, q: [implies(p, q), q], lambda p, q: p),
}

for name, (premises, conclusion) in arguments.items():
    counterexamples = [
        (p, q) for p, q in product([False, True], repeat=2)
        if all(premises(p, q)) and not conclusion(p, q)
    ]
    print(name, "有效" if not counterexamples else f"无效，反例={counterexamples}")
"""
        ),
        md(
            """
## 2. 找反例比找更多支持例更有区分力

一千次“前提与结论同时为真”仍可能没有检验推理形式。只要找到一次“前提全真、结论为假”，就足以说明该形式无效。现实论证还要检查词义偷换、遗漏变量和证据质量。
"""
        ),
        code(
            """
required = {"claim", "reason", "warrant", "evidence", "objection", "boundary"}

def audit_argument(record):
    missing = sorted(required - set(record))
    empty = sorted(k for k in required & set(record) if not str(record[k]).strip())
    return {"missing": missing, "empty": empty, "ready_to_review": not missing and not empty}

draft = {
    "claim": "模型 B 应上线",
    "reason": "测试集 WER 更低",
    "evidence": "同一测试集：10.8% → 10.1%",
    "warrant": "",
}
print(audit_argument(draft))
"""
        ),
        md(
            """
## 3. 表达方式也会改变判断

图表不是装饰。位置和长度通常比角度、面积、体积或色彩饱和度更容易精确比较。展示“提升”时同时给绝对值、差值、分母和不确定性，避免截断坐标制造视觉夸张。
"""
        ),
        code(
            """
old, new = 0.108, 0.101
relative_change = (old - new) / old
absolute_change = old - new
print(f"旧 WER={old:.1%}，新 WER={new:.1%}")
print(f"绝对改善={absolute_change:.1%}，相对改善={relative_change:.1%}")
print("只说“降低 6.5%”会隐藏基线和绝对幅度。")
"""
        ),
    ]


def bayes_cells() -> list:
    return [
        md(
            """
## 1. 后验同时取决于证据强度和基础率

灵敏度 90% 不等于“阳性后有 90% 概率为真”。先把概率改写为 10,000 人中的自然频数，分母就不容易丢失。
"""
        ),
        code(
            """
def positive_posterior(prevalence, sensitivity, false_positive_rate):
    true_positive = prevalence * sensitivity
    false_positive = (1 - prevalence) * false_positive_rate
    return true_positive / (true_positive + false_positive)

p = positive_posterior(0.01, 0.90, 0.05)
print(f"阳性后的概率={p:.1%}")
print("10,000 人中：约 90 个真阳性，495 个假阳性。")
assert round(p, 3) == 0.154
"""
        ),
        md(
            """
## 2. 用赔率做连续更新

后验赔率 = 先验赔率 × 似然比。每一份证据只有在“真时多常见、假时多常见”不同的情况下才提供信息；重复使用同源证据会造成虚假自信。
"""
        ),
        code(
            """
def probability_to_odds(p):
    return p / (1 - p)

def odds_to_probability(o):
    return o / (1 + o)

prior = 0.10
likelihood_ratios = [4.0, 2.0]
odds = probability_to_odds(prior)
for lr in likelihood_ratios:
    odds *= lr
    print(f"LR={lr:.1f} 后，概率={odds_to_probability(odds):.1%}")
print("边界：两条证据若高度相关，不能简单相乘。")
"""
        ),
        md(
            """
## 3. 概率是可更新的承诺，不是情绪强度

把“很可能”改写为数值区间，并提前定义什么证据会怎样更新。长期记录已解决事件，才能检查 70% 判断是否大约七成发生。
"""
        ),
        code(
            """
priors = [0.01, 0.10, 0.50]
for prior in priors:
    posterior = positive_posterior(prior, 0.90, 0.05)
    print(f"先验={prior:>5.0%} -> 同一阳性证据后的后验={posterior:>6.1%}")
"""
        ),
    ]


def causal_cells() -> list:
    return [
        md(
            """
## 1. 看到相关，先画变量关系

混杂变量同时影响“处理”和“结果”；中介是处理影响结果的路径；碰撞点是两个原因共同指向的结果。该控制什么取决于因果问题，不是“变量越多越科学”。
"""
        ),
        code(
            """
import numpy as np
rng = np.random.default_rng(7)
n = 20000
ability = rng.normal(size=n)
training = (ability + rng.normal(scale=0.8, size=n) > 0).astype(int)
score = 70 + 8 * ability + 2 * training + rng.normal(scale=4, size=n)

naive = score[training == 1].mean() - score[training == 0].mean()
bins = np.quantile(ability, [0, .2, .4, .6, .8, 1])
adjusted_parts = []
for low, high in zip(bins[:-1], bins[1:]):
    mask = (ability >= low) & (ability <= high)
    adjusted_parts.append(score[mask & (training == 1)].mean() - score[mask & (training == 0)].mean())
print(f"未调整差异={naive:.2f}，按能力分层后的平均差异={np.mean(adjusted_parts):.2f}")
"""
        ),
        md(
            """
## 2. 控制碰撞点会制造不存在的关系

如果“能力”和“关系资源”都能提高录取率，那么只看已录取者会使两者出现负相关：能力较弱但被录取的人更可能有资源，反之亦然。这不是总体中的因果关系。
"""
        ),
        code(
            """
rng = np.random.default_rng(11)
n = 50000
ability = rng.normal(size=n)
connections = rng.normal(size=n)
admitted = ability + connections + rng.normal(scale=.5, size=n) > 1.0
all_corr = np.corrcoef(ability, connections)[0, 1]
selected_corr = np.corrcoef(ability[admitted], connections[admitted])[0, 1]
print(f"总体相关={all_corr:.3f}，只看已录取者={selected_corr:.3f}")
"""
        ),
        md(
            """
## 3. 观察问题与干预问题不同

“使用某功能的人表现更好”是观察关联；“给同类人开启功能会怎样”是干预问题。因果图不能凭数据自动确定，必须结合时间顺序、机制知识和设计。
"""
        ),
        code(
            """
roles = {
    "设备档次": "可能同时影响是否启用功能和识别率：候选混杂",
    "降噪后 SNR": "若由降噪产生并影响识别：中介",
    "是否进入人工复核": "若由低置信度和高风险共同决定：碰撞点",
}
for variable, role in roles.items():
    print(f"{variable}: {role}")
"""
        ),
    ]


def systems_cells() -> list:
    return [
        md(
            """
## 1. 存量由流量累积，不由口号改变

待处理问题数是存量；新问题和完成问题是流量。即使团队每天很忙，只要流入大于流出，积压仍会上升。先画边界、存量、流量、反馈和延迟。
"""
        ),
        code(
            """
def backlog(initial, arrivals, capacity):
    stock = initial
    history = [stock]
    for incoming, service in zip(arrivals, capacity):
        stock = max(0, stock + incoming - service)
        history.append(stock)
    return history

history = backlog(20, [12] * 10, [10] * 10)
print("积压轨迹:", history)
print("每天完成 10 个仍然越来越多，因为每天新增 12 个。")
"""
        ),
        md(
            """
## 2. 延迟让合理动作产生振荡

如果只根据几天前的信息猛烈调节，系统容易过冲。扩大反馈增益可能短期看起来积极，长期却造成忽高忽低。
"""
        ),
        code(
            """
def delayed_control(days=24, target=20, gain=.7, delay=3):
    stock = [35.0] * (delay + 1)
    service = 10.0
    for _ in range(days):
        observed = stock[-1 - delay]
        service = max(0, 10 + gain * (observed - target))
        stock.append(max(0, stock[-1] + 10 - service))
    return [round(x, 1) for x in stock]

print("温和调节:", delayed_control(gain=.25))
print("激进调节:", delayed_control(gain=1.10))
"""
        ),
        md(
            """
## 3. 杠杆点不等于“最容易改的参数”

参数、缓冲区、信息流、规则、目标和范式可能位于不同层级。高杠杆点通常更难改变，也更容易产生副作用；Meadows 明确提醒这不是机械配方。
"""
        ),
        code(
            """
interventions = [
    ("每人每天多做 1 个", "参数", 1, 2),
    ("即时暴露失败切片", "信息流", 3, 2),
    ("发布门禁奖励真实质量而非样本数", "规则/目标", 5, 4),
]
for name, level, expected_effect, implementation_risk in interventions:
    print(f"{name}: 层级={level}, 预期影响={expected_effect}, 实施风险={implementation_risk}")
print("排序前必须同时考虑作用、延迟、反作用和可逆性。")
"""
        ),
    ]


def decision_cells() -> list:
    return [
        md(
            """
## 1. 先把状态、行动、结果和价值分开

期望值是各情景结果按概率加权。它不是“最可能结果”，也不能自动表达灾难风险、公平、硬约束或效用的非线性。
"""
        ),
        code(
            """
states = {"正常流量": .70, "高峰": .20, "故障": .10}
payoffs = {
    "方案A": {"正常流量": 8, "高峰": 3, "故障": -20},
    "方案B": {"正常流量": 6, "高峰": 5, "故障": -4},
}
expected = {
    action: sum(states[s] * value for s, value in outcomes.items())
    for action, outcomes in payoffs.items()
}
print("期望值:", expected)
"""
        ),
        md(
            """
## 2. 后悔值和鲁棒性提供另一种视角

当概率很脆弱时，可比较最坏结果、最大后悔和概率敏感性。不同规则回答不同价值问题，不应把它们混成一个“客观总分”。
"""
        ),
        code(
            """
best_by_state = {s: max(payoffs[a][s] for a in payoffs) for s in states}
max_regret = {
    action: max(best_by_state[s] - outcomes[s] for s in states)
    for action, outcomes in payoffs.items()
}
worst_case = {action: min(outcomes.values()) for action, outcomes in payoffs.items()}
print("最大后悔:", max_regret)
print("最坏结果:", worst_case)
"""
        ),
        md(
            """
## 3. 信息只有在可能改变行动时才有决策价值

完美信息价值 EVPI = 知道状态后每次选最好方案的期望值 − 现在最佳方案的期望值。真实测试还要扣除时间、金钱和延迟决策的成本。
"""
        ),
        code(
            """
without_information = max(expected.values())
with_perfect_information = sum(
    states[s] * max(payoffs[a][s] for a in payoffs) for s in states
)
evpi = with_perfect_information - without_information
print(f"无信息最佳值={without_information:.2f}")
print(f"完美信息值={with_perfect_information:.2f}，EVPI={evpi:.2f}")
print("若测试总成本高于 EVPI，就不值得只为这项决策购买。")
"""
        ),
    ]


def information_cells() -> list:
    return [
        md(
            """
## 1. 信息量衡量不确定性的减少

低概率事件发生时“惊讶度”更高；熵是平均惊讶度。它描述符号分布，不自动描述内容是否真实、有用或有意义。
"""
        ),
        code(
            """
import math

def entropy(probabilities):
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

for probs in ([.5, .5], [.9, .1], [.25] * 4):
    print(probs, "entropy=", round(entropy(probs), 3), "bits")
"""
        ),
        md(
            """
## 2. 好编码利用可预测性

理想码长约为 −log2(p)。常见符号用短码、罕见符号用长码，平均长度可下降。压缩迫使我们看见表示中哪些结构是重复的。
"""
        ),
        code(
            """
symbols = {"静音": .70, "语音": .25, "告警": .05}
for symbol, probability in symbols.items():
    ideal_length = -math.log2(probability)
    print(f"{symbol}: p={probability:.2f}, 理想信息量={ideal_length:.2f} bits")
print("固定三类编码需要约 2 bits/符号；利用分布可降低平均码长。")
"""
        ),
        md(
            """
## 3. 互信息看共享不确定性，不看语义好坏

若两个变量独立，知道一个不会减少另一个的不确定性，互信息接近 0。互信息高也可能来自泄漏、共同原因或无意义标识符，不能直接推出因果或实用价值。
"""
        ),
        code(
            """
from collections import Counter

def mutual_information(xs, ys):
    n = len(xs)
    joint, cx, cy = Counter(zip(xs, ys)), Counter(xs), Counter(ys)
    return sum(
        count / n * math.log2((count * n) / (cx[x] * cy[y]))
        for (x, y), count in joint.items()
    )

x = [0, 0, 1, 1] * 100
y_same = x[:]
y_independent = [0, 1, 0, 1] * 100
print("完全共享:", mutual_information(x, y_same))
print("独立排列:", mutual_information(x, y_independent))
"""
        ),
    ]


def forecast_cells() -> list:
    return [
        md(
            """
## 1. 概率判断要留下可评分记录

Brier 分数是预测概率与 0/1 结果的平方误差平均，越低越好。它会同时惩罚方向错误和过度自信；但只看一个事件不能判断校准。
"""
        ),
        code(
            """
def brier(probabilities, outcomes):
    return sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(outcomes)

outcomes = [1, 0, 1, 1, 0, 0, 1, 0]
cautious = [.7, .3, .7, .7, .3, .3, .7, .3]
overconfident = [.95, .05, .95, .05, .05, .95, .95, .05]
print("稳健预测:", round(brier(cautious, outcomes), 3))
print("过度自信:", round(brier(overconfident, outcomes), 3))
"""
        ),
        md(
            """
## 2. 校准与区分能力不是一回事

校准问“报 70% 的事件是否约七成发生”；区分能力问“能否给更可能发生的事件更高概率”。全报基础率可能校准却没有区分力。
"""
        ),
        code(
            """
def calibration_bins(probabilities, outcomes):
    groups = {}
    for p, y in zip(probabilities, outcomes):
        key = round(p, 1)
        groups.setdefault(key, []).append(y)
    return {p: (len(values), sum(values) / len(values)) for p, values in groups.items()}

probs = [.2, .2, .2, .2, .7, .7, .7, .7, .7, .7]
ys =    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
print("预测档位: (样本数, 实际发生率)", calibration_bins(probs, ys))
"""
        ),
        md(
            """
## 3. 偏差清单要变成流程护栏

可得性、代表性和锚定是常见启发式；启发式往往有用，但在某些条件下产生系统误差。实用护栏是先写基础率、独立估计、反方证据和解决日期，而不是事后给错误命名。
"""
        ),
        code(
            """
forecast_record = {
    "question": "下周线上 WER 回退是否超过 1 个百分点？",
    "probability": 0.35,
    "base_rate": "过去 20 次发布中有 5 次",
    "inside_view": "本次改动触及 VAD",
    "outside_view": "同类 VAD 改动 6 次中 2 次回退",
    "resolution_date": "2026-08-25",
    "resolution_rule": "固定线上样本、同一规范下 WER 差值 > 0.01",
}
required = {"question", "probability", "base_rate", "inside_view", "outside_view",
            "resolution_date", "resolution_rule"}
print("预测记录完整?", required <= forecast_record.keys())
"""
        ),
    ]


def learning_cells() -> list:
    return [
        md(
            """
## 1. 流畅感不是长期记忆

反复阅读常让材料当下看起来熟悉；闭卷提取会暴露缺口，并在延迟测试中常比同等时间的重读更有利。提取之后必须核对反馈，避免巩固错误。
"""
        ),
        code(
            """
cards = [
    {"question": "为什么幅值 1000 不能跨位深判断声音小？", "recalled": False},
    {"question": "混杂变量与碰撞点有什么区别？", "recalled": True},
    {"question": "Brier 分数怎样计算？", "recalled": False},
]
for card in cards:
    action = "今天重学并闭卷再答" if not card["recalled"] else "延后复习并做迁移题"
    print(card["question"], "->", action)
"""
        ),
        md(
            """
## 2. 间隔长度要服务于保留目标

间隔练习总体优于挤在一起，但不存在适合所有任务的神奇天数；最佳间隔与希望记多久有关。下面只是可编辑日程，不是大脑的精确模型。
"""
        ),
        code(
            """
from datetime import date, timedelta

def review_dates(start, gaps=(1, 3, 7, 16, 35)):
    return [start + timedelta(days=gap) for gap in gaps]

for day in review_dates(date(2026, 8, 18)):
    print(day.isoformat())
"""
        ),
        md(
            """
## 3. 训练要包含变化、反馈和迁移

只在同一道题上变快可能是记住答案。交错不同题型、改变表面条件、解释错误原因并延迟复测，才能积累迁移证据。
"""
        ),
        code(
            """
skills = ["幅值/dB", "贝叶斯", "因果图", "系统反馈"]
schedule = []
for round_id in range(3):
    offset = round_id % len(skills)
    schedule.append(skills[offset:] + skills[:offset])
for index, block in enumerate(schedule, 1):
    print(f"第 {index} 轮:", " → ".join(block))

evidence_levels = {
    0: "看过", 1: "能识别", 2: "能闭卷解释",
    3: "能从空白实现", 4: "能在新问题中迁移并解释边界",
}
print("真正掌握目标:", evidence_levels[4])
"""
        ),
    ]


LESSONS = [
    Lesson(
        1,
        "论证反例与表达",
        "论证、反例与数据表达",
        "怎样判断一个结论是被理由支持，还是只听起来顺耳？",
        (
            ("forall x: Calgary：有效性与反例", "https://forallx.openlogicproject.org/bookml/Ch2.html"),
            ("Cleveland & McGill：图形感知实验", "https://www.tandfonline.com/doi/abs/10.1080/01621459.1984.10478080"),
        ),
        logic_cells(),
        "拆解“准确率更高，所以产品一定更好”：补齐前提、担保、证据、反对意见和边界，再构造一个前提真而结论假的现实反例。",
    ),
    Lesson(
        2,
        "基础率贝叶斯与自然频数",
        "基础率、贝叶斯与自然频数",
        "同一条证据为什么会在不同背景下支持不同结论？",
        (
            ("Gigerenzer 等：自然频数与贝叶斯推理", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4604268/"),
            ("Max Planck 研究记录", "https://pure.mpg.de/view/item_2228574"),
        ),
        bayes_cells(),
        "为一个“ASR 置信度低就一定识别错”的判断构造 1,000 条自然频数表，并解释基础错误率改变时后验怎样变化。",
    ),
    Lesson(
        3,
        "因果图混杂中介与碰撞点",
        "因果图：混杂、中介与碰撞点",
        "为什么控制更多变量有时反而让结论更错？",
        (
            ("Hernán & Robins：Causal Inference: What If", "https://miguelhernan.org/whatifbook"),
            ("Pearl、Glymour、Jewell：因果推断入门", "https://web.cs.ucla.edu/~kaoru/primer-complete-2019.pdf"),
        ),
        causal_cells(),
        "画一张“使用降噪功能→识别率”的因果图，至少包含一个混杂、一个中介和一个选择变量；分别说明控制它们会回答什么问题。",
    ),
    Lesson(
        4,
        "存量流量反馈延迟与杠杆",
        "系统思维：存量、反馈、延迟与杠杆",
        "为什么局部看似正确的动作会让整体长期变差？",
        (
            ("Donella Meadows：系统杠杆点", "https://donellameadows.org/archives/leverage-points-places-to-intervene-in-a-system/"),
        ),
        systems_cells(),
        "选择一个积压问题，画出至少一个存量、两个流量、一个增强回路、一个平衡回路和一个延迟；提出低层与高层干预各一个。",
    ),
    Lesson(
        5,
        "期望值后悔值与信息价值",
        "决策论：期望值、后悔与信息价值",
        "信息不全时，怎样知道该行动、该等待还是该再做实验？",
        (
            ("MIT OCW：Risk and Decision Analysis", "https://ocw.mit.edu/courses/ids-333-risk-and-decision-analysis-fall-2021/"),
            ("MIT OCW：Value of Information", "https://ocw.mit.edu/courses/ids-333-risk-and-decision-analysis-fall-2021/resources/unit-9-value-of-info-video-3/"),
        ),
        decision_cells(),
        "为一次可逆选择和一次不可逆选择各建一棵决策树，比较期望值、最坏结果、最大后悔、EVPI 和延迟成本。",
    ),
    Lesson(
        6,
        "熵编码互信息与表示",
        "信息论：熵、编码、互信息与表示",
        "什么叫信息更多，为什么它不等于内容更有意义？",
        (
            ("Claude Shannon：A Mathematical Theory of Communication", "https://bayes.wustl.edu/Manual/shannon1948.pdf"),
        ),
        information_cells(),
        "构造两段熵完全相同但意义、真实性或行动价值完全不同的消息；解释 Shannon 信息量为什么无法替你判断语义。",
    ),
    Lesson(
        7,
        "认知偏差预测校准与复盘",
        "认知偏差、概率预测与校准",
        "怎样把“我觉得”变成可追踪、可纠错的判断记录？",
        (
            ("Tversky & Kahneman：不确定判断的启发式与偏差", "https://pubmed.ncbi.nlm.nih.gov/17835457/"),
            ("Brier：概率预测评分", "https://journals.ametsoc.org/doi/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2"),
            ("Mellers 等：预测竞赛中的校准实践", "https://pubmed.ncbi.nlm.nih.gov/28154049/"),
        ),
        forecast_cells(),
        "建立 10 条有明确解决日期的预测，强制写基础率、内部视角和反方证据；解决后计算 Brier 分数并按概率档检查校准。",
    ),
    Lesson(
        8,
        "提取间隔反馈与迁移",
        "学习科学：提取、间隔、反馈与迁移",
        "怎样区分“看懂了”与“以后能独立用出来”？",
        (
            ("Roediger & Karpicke：测试效应", "https://journals.sagepub.com/doi/pdf/10.1111/j.1467-9280.2006.01693.x"),
            ("Cepeda 等：间隔练习元分析", "https://pubmed.ncbi.nlm.nih.gov/16719566/"),
            ("Dunlosky 等：学习技术综述", "https://www.psychologicalscience.org/publications/journals/pspi/learning-techniques.html"),
            ("National Academies：可复现性与可重复性", "https://nap.nationalacademies.org/initiative/committee-on-reproducibility-and-replicability-in-science"),
        ),
        learning_cells(),
        "选一个本周学过的概念，设计第 1/3/7/16/35 天闭卷提取；每次换表面场景，并定义 0～4 级掌握证据和失败后的反馈动作。",
    ),
]


def build_notebook(lesson: Lesson):
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "course": {
            "track": "cognitive_toolkit",
            "lesson": lesson.number,
            "title": lesson.title,
            "evidence_model": "question-model-prediction-evidence-boundary-update",
        },
    }
    notebook.cells = intro(lesson) + lesson.cells + finish(lesson)
    return notebook


def main() -> None:
    for lesson in LESSONS:
        source = NOTEBOOK_DIR / f"认知拓展_{lesson.number:02d}_{lesson.slug}.ipynb"
        notebook = build_notebook(lesson)
        nbf.write(notebook, source)

        executed = copy.deepcopy(notebook)
        NotebookClient(
            executed,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        sanitize_notebook_outputs(executed)
        target = executed_path(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(executed, target)
        print(f"built {source.name} -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
