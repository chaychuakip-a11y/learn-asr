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
    concepts: str
    practice: str
    cells: list
    challenge: str


def intro(lesson: Lesson) -> list:
    sources = "\n".join(f"- [{name}]({url})" for name, url in lesson.sources)
    return [
        md(
            f"""
# 研究进阶 {lesson.number}/8：{lesson.title}

核心问题：**{lesson.question}**

目标不是“搜到很多”，而是让另一个人能够检查你为何这样问、在哪里找、为何纳入、如何判断、何时停止，以及什么会改变结论。建议投入 75～120 分钟。

固定研究链：

问题边界 → 竞争解释 → 检索设计 → 来源溯源 → 证据矩阵 → 冲突处理 → 暂定结论 → 更新触发器

## 课前预测

1. 闭卷写出你现在的答案与 0%～100% 置信度。
2. 写出最强替代解释，以及它和你的解释产生不同预测的地方。
3. 预测第一个代码实验的结果；运行后保留原预测。
4. 写出：什么结果会让我改主意：
"""
        ),
        md(
            f"""
## 一手资料与课程取舍

{sources}

课程把标准、方法手册和原始研究转成小实验。一手资料也可能过时或不适用于当前情境；权威取决于问题、证据生成过程和适用范围。
"""
        ),
        md(
            f"""
## 核心概念

{lesson.concepts}
"""
        ),
    ]


def finish(lesson: Lesson) -> list:
    return [
        md(
            f"""
## 信息搜集实作

{lesson.practice}

必须保存：问题范围、检索式、平台、日期、纳入与排除理由、来源家族、支持与反对证据、停止规则。搜索摘要、转载和 AI 总结只作导航线索，不能直接进入证据矩阵。

## 误用警报与适用边界

- 形式逻辑能检查结论是否随前提而来，不能自动证明前提真实。
- 来源名气、引用量和网页排名都是线索，不是质量判决。
- 多个页面可能来自同一研究或同一通讯稿；独立来源家族数比链接数更重要。
- 资源受限的停止规则不是“已经找全”的证明；正式系统综述要遵循学科协议。
- 证据矩阵压缩了细节；高风险结论必须回到原文的方法、样本、版本和局限。

## 迁移练习

分别选择 ASR、工作决策、日常生活中的一个问题。对每个问题写出：

    可操作问题：
    关键量词或条件：
    最强替代解释：
    正向检索式：
    反方检索式：
    独立来源家族：
    暂定结论与置信度：
    什么结果会让我改主意：
    下一步最小行动：
"""
        ),
        md(
            f"""
## 闭卷挑战

{lesson.challenge}

不联网，用空白纸完成第一版；然后联网验证，并用不同颜色标出被证据改变的部分。

## 最小掌握门禁

- [ ] 能不用术语讲清本课方法。
- [ ] 能从空白重写至少一个核心函数。
- [ ] 能构造反例或竞争解释，而不只收集支持材料。
- [ ] 能分开“报告数、来源家族数、直接证据数”。
- [ ] 能指出结论的适用边界和至少一个更新触发器。
- [ ] 已把检索式、排除理由和一次置信度更新写入 LEARNING_LOG.md。
"""
        ),
    ]


LESSONS = [
    Lesson(
        1,
        "问题定义问题树与费米估算",
        "问题定义、问题树与费米估算",
        "怎样把一个含糊大问题拆成能够检索、估算和验证的子问题？",
        (
            ("ACRL 信息素养框架：Research as Inquiry", "https://www.ala.org/acrl/standards/ilframework"),
            ("NASA Systems Engineering Handbook", "https://www.nasa.gov/reference/systems-engineering-handbook/"),
        ),
        """
先写决策，再写问题。一个可操作问题至少说明对象、干预或现象、比较、结果指标、时间和约束。问题树要尽量满足“互不重复、合起来不漏掉当前决策所需部分”，但 MECE 只是检查提示，不是世界必然能被整齐切开。

费米估算不追求假装精确，而是暴露数量级、假设和敏感项。先给每个输入一个低—中—高范围，再优先验证会改变决策的输入。
""",
        "把“怎样提高 ASR”改写成一个有对象、场景、指标、基线、预算与期限的问题。画问题树，做低—中—高估算，并先检索最敏感的假设。",
        [
            md("## 1. 从决策反推问题，而不是从关键词堆积开始"),
            code(
                """
question = {
    "decision": "是否投入一周改善车载远场数字识别",
    "population": "8 kHz 车载远场中文数字命令",
    "baseline": "说话人隔离测试 WER 18%",
    "target": "WER <= 12%",
    "constraints": "单人一周；实时率 <= 0.5",
}
required = {"decision", "population", "baseline", "target", "constraints"}
print("缺失字段:", sorted(required - question.keys()))
print(question)
"""
            ),
            md("## 2. 问题树把“为什么”变成可分别取证的叶节点"),
            code(
                """
tree = {
    "数据": ["说话人覆盖", "噪声覆盖", "标签错误"],
    "表示": ["采样率", "特征参数", "增益归一化"],
    "模型": ["容量", "过拟合", "解码约束"],
    "评估": ["数据泄漏", "切片差异", "置信区间"],
}
for branch, leaves in tree.items():
    print(f"{branch}: " + " / ".join(leaves))
print("叶节点数:", sum(map(len, tree.values())))
"""
            ),
            md("## 3. 区间估算和敏感性决定先查什么"),
            code(
                """
assumptions = {
    "每天可标注小时": (0.5, 1.0, 1.5),
    "有效工作天": (4, 5, 6),
    "每小时音频带来的相对错误下降": (0.01, 0.03, 0.05),
}
for name, values in assumptions.items():
    print(name, "低/中/高 =", values)
low = 0.5 * 4 * 0.01
mid = 1.0 * 5 * 0.03
high = 1.5 * 6 * 0.05
print("预期相对错误下降范围:", (round(low, 3), round(mid, 3), round(high, 3)))
print("跨度最大且最不确定的输入应优先验证。")
"""
            ),
        ],
        "把“AI 会不会取代我的工作”拆成决策问题、至少三层问题树和三个可观测指标；给出数量级估算，并指出最值得先买信息的一个假设。",
    ),
    Lesson(
        2,
        "必要充分量词作用域与反例",
        "必要条件、充分条件、量词、作用域与反例",
        "怎样防止把“只有、只要、所有、存在、相关”读成错误的逻辑方向？",
        (
            ("Open Logic Project：Conditionals", "https://forallx.openlogicproject.org/bookml/Ch5.html"),
            ("Open Logic Project：Quantifier Negation", "https://forallx.openlogicproject.org/html/Ch24.html"),
            ("Open Logic Project：Multiple Quantifiers", "https://forallx.openlogicproject.org/html/Ch25.html"),
        ),
        """
“A 是 B 的充分条件”写作 A → B；“A 是 B 的必要条件”写作 B → A。否定全称不是“全都不”，而是“至少有一个不”。量词顺序也会改变含义：每个用户都有某个可用模型，不等于存在一个模型适合所有用户。

检验逻辑形式的最快办法是寻找“前提都真、结论为假”的反例。自然语言还要检查范围、时态、比较组和词义是否悄悄变化。
""",
        "从一篇产品文档或新闻中摘录五个带“所有、任何、只有、只要、可能”的句子，形式化后为每句找反例。把无法确定作用域的句子重写得可检验。",
        [
            md("## 1. 必要和充分的方向不能靠语感"),
            code(
                """
from itertools import product

def implies(a, b):
    return (not a) or b

for noise_reduced, accurate in product([False, True], repeat=2):
    sufficient_holds = implies(noise_reduced, accurate)
    necessary_holds = implies(accurate, noise_reduced)
    print(noise_reduced, accurate, "充分式", sufficient_holds, "必要式", necessary_holds)
"""
            ),
            md("## 2. “并非所有”只需要一个反例"),
            code(
                """
scores = {"speaker_a": 0.92, "speaker_b": 0.91, "speaker_c": 0.61}
claim = all(score >= 0.80 for score in scores.values())
negation = any(score < 0.80 for score in scores.values())
print("所有说话人都达标:", claim)
print("并非所有说话人都达标:", negation)
print("见证反例:", [name for name, score in scores.items() if score < 0.80])
"""
            ),
            md("## 3. 量词顺序改变产品承诺"),
            code(
                """
works = {
    "model_a": {"quiet", "car"},
    "model_b": {"meeting", "street"},
    "model_c": {"quiet", "meeting"},
}
contexts = {"quiet", "car", "meeting", "street"}
every_context_has_a_model = all(any(c in covered for covered in works.values()) for c in contexts)
one_model_covers_every_context = any(contexts <= covered for covered in works.values())
print("∀场景∃模型:", every_context_has_a_model)
print("∃模型∀场景:", one_model_covers_every_context)
"""
            ),
        ],
        "形式化并反驳：“只要平均准确率高，所有真实用户都会满意；没有使用降噪的系统不可能准确。”至少区分两个量词顺序和必要/充分方向。",
    ),
    Lesson(
        3,
        "演绎归纳溯因与竞争解释",
        "演绎、归纳、溯因与竞争解释",
        "看到一个现象时，怎样避免把第一个解释误当成唯一原因？",
        (
            ("Stanford Encyclopedia：Abduction", "https://plato.stanford.edu/entries/abduction/"),
            ("NIST：Uncertainty Machine Learning", "https://www.nist.gov/programs-projects/uncertainty-machine-learning"),
        ),
        """
演绎问“前提若真，结论是否必然”；归纳问“样本对总体支持多强”；溯因问“哪个解释最能说明现象”。溯因结论永远需要竞争解释和区分性预测。

好的假设不只是能解释已经看到的结果，还应提前预测尚未观察的切片。不要只问哪个故事更顺；问各解释在哪个新观测上会给出不同结果。
""",
        "选择一个异常指标，至少提出三个竞争解释。为每个解释记录预测、反证、成本最低的区分实验和证据更新；专门检索失败案例与替代解释。",
        [
            md("## 1. 有效演绎与真实前提是两次不同检查"),
            code(
                """
arguments = [
    {"name": "有效形式", "if_p_then_q": True, "p": True, "q": True},
    {"name": "肯定后件陷阱", "if_p_then_q": True, "p": False, "q": True},
]
for item in arguments:
    conclusion_forced = item["if_p_then_q"] and item["p"]
    print(item["name"], "形式是否迫使 q:", conclusion_forced)
"""
            ),
            md("## 2. 竞争解释必须给出区分性预测"),
            code(
                """
hypotheses = {
    "输入音量低": {"低RMS": 0.9, "换模型仍差": 0.7, "只在新说话人差": 0.2},
    "说话人域偏移": {"低RMS": 0.2, "换模型仍差": 0.6, "只在新说话人差": 0.9},
    "标签错误": {"低RMS": 0.2, "换模型仍差": 0.9, "只在新说话人差": 0.3},
}
observed = {"低RMS": False, "换模型仍差": True, "只在新说话人差": True}
for name, predictions in hypotheses.items():
    score = sum(p if observed[k] else 1-p for k, p in predictions.items())
    print(name, round(score, 2))
"""
            ),
            md("## 3. 先选能最大区分解释的观测"),
            code(
                """
tests = {
    "测RMS": [0.9, 0.2, 0.2],
    "按说话人切片": [0.2, 0.9, 0.3],
    "人工复核标签": [0.1, 0.2, 0.9],
}
for name, predictions in tests.items():
    spread = max(predictions) - min(predictions)
    print(name, "区分度", round(spread, 2))
"""
            ),
        ],
        "解释“新模型离线集提升、线上投诉却增加”。提出至少四个竞争解释，为每个写区分性预测，设计一次成本最低且最可能改变排序的观测。",
    ),
    Lesson(
        4,
        "论证图钢人化与反方测试",
        "论证图、钢人化与反方测试",
        "怎样把支持与反对理由画成可审查的结构，而不是互相堆观点？",
        (
            ("Toulmin：The Uses of Argument 出版信息", "https://www.cambridge.org/core/books/uses-of-argument/26CF801BC12004587B66778297D5567C"),
            ("Center for Open Science：Registered Reports", "https://www.cos.io/initiatives/registered-reports"),
        ),
        """
论证图把最终主张、子主张、证据、隐含担保、反驳和限定词分开。先钢人化：把对方观点改写成其支持者会接受的最强版本；再红队测试：寻找能区分双方的失败条件。

“有证据支持”不是二值标签。要标明证据支持哪条边、依赖什么担保、是否与其他证据同源，以及结论应该用多强的限定词。
""",
        "选一个你强烈赞同的技术主张，先写对方最强版本。用论证图连接至少三个支持节点、两个反对节点、两个隐含假设，再做一次反方检索。",
        [
            md("## 1. 证据必须连接到具体主张"),
            code(
                """
nodes = {
    "C0": ("claim", "模型 B 应上线"),
    "C1": ("subclaim", "它改善目标用户的错误率"),
    "E1": ("evidence", "盲测 WER 降低 1.2 个百分点"),
    "A1": ("assumption", "盲测分布代表线上用户"),
    "O1": ("objection", "长尾口音退化"),
}
edges = [("E1", "C1", "supports"), ("A1", "C1", "warrants"),
         ("C1", "C0", "supports"), ("O1", "C0", "attacks")]
for edge in edges:
    print(edge, ":", nodes[edge[0]][1], "->", nodes[edge[1]][1])
"""
            ),
            md("## 2. 钢人化先过“对方会签字吗”测试"),
            code(
                """
drafts = [
    {"text": "他们就是害怕新技术", "evidence": False, "charitable": False},
    {"text": "若长尾回退和回滚成本未测，暂缓全量上线", "evidence": True, "charitable": True},
]
for draft in drafts:
    ready = draft["evidence"] and draft["charitable"]
    print("可进入论证图" if ready else "需要重写", draft["text"])
"""
            ),
            md("## 3. 红队搜索词不应只是原主张加“缺点”"),
            code(
                """
claim_terms = ["ASR deployment", "lower WER"]
red_team_facets = {
    "失败": ["failure analysis", "regression", "distribution shift"],
    "外部效度": ["real world", "speaker subgroup", "out of domain"],
    "复现": ["replication", "benchmark leakage"],
}
for facet, terms in red_team_facets.items():
    print(facet, "=>", " OR ".join(terms))
"""
            ),
        ],
        "为“开源模型一定比商业 API 更适合我们的 ASR 项目”画论证图。先钢人化正反两方，再指出哪条证据能真正切断争议。",
    ),
    Lesson(
        5,
        "分面检索布尔查询与停止规则",
        "分面检索、布尔查询与停止规则",
        "怎样从一条随手搜索升级为可复查、能迭代的检索策略？",
        (
            ("ACRL 信息素养框架：Searching as Strategic Exploration", "https://www.ala.org/acrl/standards/ilframework"),
            ("Crossref REST API 官方文档", "https://github.com/CrossRef/rest-api-doc"),
            ("Cochrane Handbook Chapter 4", "https://training.cochrane.org/handbook/current/chapter-04"),
        ),
        """
检索是非线性迭代：先把问题拆成概念分面，每个分面收集同义词、旧称、缩写、上位词和受控词；分面内 OR 扩召回，分面间 AND 收精度。记录每次修改为何发生。

种子来源可以向后查参考文献、向前查被引、横向查相关研究和作者。停止前至少做一次反方检索，并声明预算、饱和规则和遗漏风险。
""",
        "用至少三个分面建立查询词典，在两个不同来源库各运行两轮。每轮记录检索式、筛选数、新增合格数和新发现同义词；执行向前、向后引文追踪与一次反方检索。",
        [
            md("## 1. 分面内 OR，分面间 AND"),
            code(
                """
from research_workspace.workbench import build_boolean_query

facets = {
    "对象": ["automatic speech recognition", "ASR"],
    "问题": ["accent robustness", "speaker generalization"],
    "方法": ["evaluation", "benchmark", "error analysis"],
}
query = build_boolean_query(facets)
print(query)
"""
            ),
            md("## 2. API URL 让查询参数和结果上限可保存"),
            code(
                """
from research_workspace.workbench import build_crossref_url

url = build_crossref_url(query, rows=20, from_year=2020)
print(url)
print("本课只生成官方 Crossref URL，不自动联网；先审查检索式再请求。")
"""
            ),
            md("## 3. 停止规则必须连同边界一起报告"),
            code(
                """
from research_workspace.workbench import should_stop_searching

rounds = [9, 4, 2, 0, 0, 0]
decision = should_stop_searching(rounds, window=3, threshold=0)
print(decision)
"""
            ),
        ],
        "围绕“8 kHz 音频是否足以训练可靠语音识别”建立分面词典、主查询、反方查询、引文追踪路线与停止规则；列出你最可能漏掉的来源类型。",
    ),
    Lesson(
        6,
        "横向阅读来源溯源与独立性",
        "横向阅读、来源溯源与独立性",
        "看到一个可信外观的页面时，怎样查清谁在说、证据从哪来、是否真正独立？",
        (
            ("Civic Online Reasoning：研究与横向阅读", "https://cor.inquirygroup.org/research/"),
            ("Google Scholar 官方帮助：Cited by 与版本", "https://scholar.google.com/intl/us/scholar/help.html"),
            ("Zotero 官方快速入门", "https://www.zotero.org/support/quick_start_guide"),
        ),
        """
横向阅读意味着暂时离开当前页面，在其他标签中调查发布者、作者、资金、声誉和原始证据。不要被页面设计困住。溯源时尽量从转载回到通讯社、机构报告、预印本或论文，再区分版本。

来源独立性是证据合成的核心：十家媒体复制同一通讯稿仍只有一个证据家族；同一研究的预印本、会议稿和期刊版也不能当三次独立重复。
""",
        "选择一个传播广的技术数字，打开至少四个横向标签：发布者背景、原始报告、其他独立报道、反方材料。画来源谱系，标出复制关系、资金、版本和独立来源家族。",
        [
            md("## 1. 先问来源角色，再问页面是否漂亮"),
            code(
                """
sources = [
    {"id": "A", "role": "原始研究", "family": "study-1", "direct": True},
    {"id": "B", "role": "大学新闻稿", "family": "study-1", "direct": False},
    {"id": "C", "role": "媒体转载", "family": "study-1", "direct": False},
    {"id": "D", "role": "独立复现", "family": "study-2", "direct": True},
]
for source in sources:
    print(source)
"""
            ),
            md("## 2. 链接数不等于独立证据数"),
            code(
                """
from research_workspace.workbench import evidence_family_summary

summary = evidence_family_summary([
    {"source_family": s["family"], "direct_evidence": s["direct"]}
    for s in sources
])
print(summary)
"""
            ),
            md("## 3. 来源谱系还要记录版本和派生关系"),
            code(
                """
lineage = [
    ("study-preprint", "study-journal", "修订并同行评审"),
    ("study-journal", "press-release", "机构摘要"),
    ("press-release", "news-a", "近乎逐字转载"),
    ("press-release", "news-b", "近乎逐字转载"),
]
roots = {parent for parent, _, _ in lineage} - {child for _, child, _ in lineage}
print("候选根来源:", roots)
for relation in lineage:
    print(" -> ".join(relation[:2]), relation[2])
"""
            ),
        ],
        "查证“某 AI 系统达到人类水平”这一表述。回到原始基准和任务定义，画出来源家族，说明“人类水平”在哪些量词、样本或条件上被扩大了。",
    ),
    Lesson(
        7,
        "证据矩阵冲突消解与综合",
        "证据矩阵、冲突消解与综合",
        "来源彼此矛盾时，怎样比较而不是挑选最顺眼的一条？",
        (
            ("Cochrane Handbook：Risk of Bias", "https://training.cochrane.org/handbook/current/chapter-08"),
            ("National Academies：Reproducibility and Replicability", "https://www.nationalacademies.org/read/25303/chapter/10"),
        ),
        """
证据矩阵让每一行对应一个来源家族，每一列对应研究对象、方法、比较组、指标、结果、局限和它支持的主张。冲突可能来自问题不同、样本不同、测量不同、分析不同、偏倚或随机误差，而不一定是谁撒谎。

综合不是数票。先判断来源是否真正回答同一问题，再考虑直接性、方法质量、精度、一致性和适用性。结论应逐主张给置信度，不用一个总分掩盖致命缺陷。
""",
        "为一个争议建立至少六行证据矩阵，其中至少两行反对你的初始判断。逐项解释冲突来自哪一层，并写出“支持什么、不支持什么、对谁适用”。",
        [
            md("## 1. 把来源压成可比较字段，而不抹去局限"),
            code(
                """
evidence = [
    {"id": "S1", "family": "lab-a", "population": "朗读语音", "effect": -2.1, "risk": "中"},
    {"id": "S2", "family": "field-b", "population": "真实车载", "effect": 0.4, "risk": "低"},
    {"id": "S3", "family": "lab-c", "population": "朗读语音", "effect": -1.7, "risk": "高"},
]
for row in evidence:
    print(row)
"""
            ),
            md("## 2. 先按问题与人群切片，再讨论一致性"),
            code(
                """
from collections import defaultdict

by_population = defaultdict(list)
for row in evidence:
    by_population[row["population"]].append(row["effect"])
for population, effects in by_population.items():
    print(population, "结果", effects, "方向一致", all(x < 0 for x in effects) or all(x >= 0 for x in effects))
"""
            ),
            md("## 3. 综合句同时包含方向、置信度、范围与更新条件"),
            code(
                """
conclusion = {
    "claim": "该方法可能改善朗读语音离线 WER",
    "confidence": "中",
    "supports": "两个实验室来源家族方向一致",
    "against": "真实车载直接证据未显示改善",
    "boundary": "不能外推到车载部署",
    "update_trigger": "新增预注册车载盲测或发现数据泄漏",
}
for key, value in conclusion.items():
    print(f"{key}: {value}")
"""
            ),
        ],
        "三篇文章说方案有效，两篇说无效。禁止按 3:2 表决；建立证据矩阵，解释冲突来源，为至少两个不同人群分别下结论并给更新条件。",
    ),
    Lesson(
        8,
        "系统综述开放科学与研究档案",
        "系统综述、开放科学与可更新研究档案",
        "怎样让一次研究在三个月后仍能复查、复现和更新？",
        (
            ("PRISMA 2020 Statement", "https://www.prisma-statement.org/prisma-2020-statement"),
            ("PRISMA-S：文献检索报告扩展", "https://www.prisma-statement.org/prisma-search"),
            ("Center for Open Science：OSF", "https://www.cos.io/products/osf"),
        ),
        """
系统综述的价值不仅是结论，更是透明的问题、协议、检索、筛选、排除和综合过程。PRISMA 是报告指南，不是研究质量评分器。日常研究可借用其透明性，但不要声称完成了正式系统综述。

可更新档案至少保存问题版本、协议、完整检索式与日期、候选来源、排除理由、来源家族、证据矩阵、分析代码、结论、局限、指纹和下次更新时间。
""",
        "复制示例研究档案，为你自己的问题替换所有字段。请另一人仅凭档案复查一条主张；记录其无法复现之处，修订后保存新指纹。",
        [
            md("## 1. 选择流程的算术也应可审计"),
            code(
                """
from research_workspace.workbench import prisma_flow_errors

flow = {
    "identified": 18, "duplicates_removed": 3, "screened": 15,
    "excluded_screening": 8, "full_text_assessed": 7,
    "excluded_full_text": 4, "included": 3,
}
print("流程错误:", prisma_flow_errors(flow))
"""
            ),
            md("## 2. 自动审计不能替代实质判断，但能抓住缺项和断链"),
            code(
                """
import json
from pathlib import Path
from research_workspace.workbench import audit_dossier

path = Path("research_workspace/example_dossier.json")
dossier = json.loads(path.read_text(encoding="utf-8"))
report = audit_dossier(dossier)
print(json.dumps(report, ensure_ascii=False, indent=2))
"""
            ),
            md("## 3. 指纹使“后来改过什么”成为可检查的问题"),
            code(
                """
from research_workspace.workbench import research_fingerprint

before = research_fingerprint(dossier)
revised = dict(dossier)
revised["limitations"] = dossier["limitations"] + ["尚未由第二名研究者复核"]
after = research_fingerprint(revised)
print("原指纹:", before)
print("新指纹:", after)
print("内容改变:", before != after)
"""
            ),
        ],
        "设计一个最小可复现研究包：问题、协议、查询日志、PRISMA 风格流程、排除表、来源家族、证据矩阵、结论、局限、更新日期和指纹；解释它仍然缺少什么。",
    ),
]


def build_notebook(lesson: Lesson):
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "course": {
            "track": "information_reasoning",
            "lesson": lesson.number,
            "title": lesson.title,
            "evidence_model": "question-query-provenance-matrix-synthesis-update",
        },
    }
    notebook.cells = intro(lesson) + lesson.cells + finish(lesson)
    return notebook


def main() -> None:
    for lesson in LESSONS:
        source = NOTEBOOK_DIR / f"研究进阶_{lesson.number:02d}_{lesson.slug}.ipynb"
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
