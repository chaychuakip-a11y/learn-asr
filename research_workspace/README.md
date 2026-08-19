# 信息研究工作台

这是逻辑与信息研究进阶课程的可运行配套工具。它不替你做判断，而是迫使研究过程留下能够复查的记录。

## 能做什么

- 将概念分面组合为“分面内 OR、分面间 AND”的布尔检索式。
- 生成有数量上限的 Crossref 官方 Works API URL；默认不联网。
- 用 DOI 或“标题 + 年份”发现可能的重复来源。
- 将转载、新闻报道和同一研究的不同版本归入同一个来源家族。
- 检查 PRISMA 风格的识别、筛选和纳入计数是否自洽。
- 审计研究档案是否留下问题边界、检索日志、纳入排除标准、主张—证据映射、冲突、停止规则和局限。
- 对研究档案计算 SHA-256 指纹，帮助识别后续改动。

## 快速开始

先审计仓库中的示例研究档案：

    python -m research_workspace.workbench audit research_workspace/example_dossier.json

构造检索式：

    python -m research_workspace.workbench query '{"对象":["automatic speech recognition","ASR"],"问题":["accent robustness"]}'

在 PowerShell 中，复杂 JSON 的引号容易被 Shell 处理；实际研究更推荐在 Notebook 里直接调用 Python 函数。

生成 Crossref 查询 URL（这里只生成，不自动访问）：

    python -m research_workspace.workbench crossref-url "speech recognition robustness" --rows 20 --from-year 2020

## 证据纪律

搜索摘要、AI 总结和转载可以帮助发现关键词与原始来源，但不能直接充当证据。进入证据矩阵前，应打开原始报告，记录版本、日期、作者或机构、方法、限制以及它究竟支持哪个主张。

“连续几轮没有新结果”只是资源受限研究的透明停止启发式，不是已经找全所有证据的证明。正式系统综述应遵循所属领域的协议与报告规范。

示例数据见 example_dossier.json；自动测试见 tests/test_research_workspace.py。
