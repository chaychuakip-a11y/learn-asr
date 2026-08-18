# Codex 从入门到精通

这套课程面向“会提问、会 `/compact`，但还不会稳定指挥 coding agent 完成工程任务”的学习者。课程不要求你先会编程；我们会把编程知识和 Codex 操作能力分开训练。

## 最终目标

学完后，你应能独立做到：

1. 让 Codex 准确理解仓库、目标、约束和验收标准。
2. 区分解释、诊断、修改、审查、研究等任务，并选择合适的工作方式。
3. 看懂 Codex 做了什么，用测试、运行结果和差异审查判断结果是否可信。
4. 管理长任务、上下文、聊天分支、权限和风险。
5. 用 `AGENTS.md`、配置、技能、插件、MCP、自动化建立可复用工作流。
6. 把模糊需求逐步变成设计、实现、测试、文档和交付物。
7. 在真实项目中主导 Codex，而不是被动接受它的答案。

## 课程地图

| 阶段 | 章节 | 你将获得的能力 | 结业证据 |
|---|---:|---|---|
| A. 驾驶基础 | 01–04 | 上下文、提示词、任务边界、验证闭环 | 独立完成一次安全的小改动 |
| B. 工程协作 | 05–08 | 读仓库、调试、测试、审查 Git 差异 | 修复一个真实 bug，并证明无回归 |
| C. 长任务控制 | 09–12 | 计划、线程、compact、分支、工作树、并行 | 完成一个多文件功能并留存决策记录 |
| D. 持久化与扩展 | 13–16 | AGENTS、配置、技能、插件、MCP、自动化 | 建立一条可复用且可验证的工作流 |
| E. 精通实战 | 17–20 | 架构、重构、安全、评测、成本与失败恢复 | 独立交付综合项目并通过闭卷验收 |

每一章的具体目标、实战和验收证据见 [完整教学大纲](完整教学大纲.md)。

教程交付范围、验证结果及其不证明的事项见 [课程完成审计](课程完成审计.md)。

## 章节清单

- [01：先建立正确心智模型](01_心智模型与第一次闭环.md)
- [02：写出稳定有效的任务说明](02_提示词工程.md)
- [03：检查、修改、验证、审查](03_工程闭环.md)
- [04：阶段 A 实战与验收](04_阶段A实战.md)
- [05：让 Codex 快速读懂陌生仓库](05_陌生仓库调查.md)
- [06：从错误现象到根因诊断](06_系统化诊断与修复.md)
- [07：测试策略与证据强度](07_测试与证据设计.md)
- [08：Git、diff、review 与安全回退](08_Git差异与代码审查.md)
- [09：复杂任务的计划与需求访谈](09_复杂任务的计划与需求访谈.md)
- [10：聊天、项目、线程、fork 与 compact](10_聊天上下文与Compact.md)
- [11：长时间任务、检查点与失败恢复](11_长任务控制与失败恢复.md)
- [12：工作树和多 agent 协作](12_多Agent协作.md)
- [13：用 `AGENTS.md` 固化仓库规则](13_AGENTS持久化规则.md)
- [14：配置、模型、推理强度、权限与沙箱](14_配置模型权限与沙箱.md)
- [15：技能、插件和 MCP 的选择与制作](15_技能插件与MCP.md)
- [16：Hooks、自动化与非交互执行](16_Hooks自动化与非交互执行.md)
- [17：架构设计与渐进式重构](17_架构设计与渐进重构.md)
- [18：安全、隐私、依赖与供应链审查](18_安全隐私与供应链.md)
- [19：评测 Codex：成功率、回归率、耗时与成本](19_Codex工作流评测.md)
- [20：综合项目与闭卷毕业考核](20_综合项目与毕业考核.md)

20 章正文现已全部发布。每章都包含概念、示例、反例、练习和验收标准；是否进入下一阶段由验收证据决定，而不是阅读进度。

## 学习方式

每次学习控制在 45–90 分钟：

1. 阅读一节，不要背命令。
2. 在当前 `G:\learn_asr` 项目中新开一个专用聊天完成练习。
3. 要求 Codex 给出证据，不以“它说完成了”为完成。
4. 把结果填入 [学习记录](学习记录.md)。
5. 未通过验收就重做，不急着进入下一课。

建议每周学习 3 次。阶段 A 约 1 周，B 和 C 各 2 周，D 和 E 各 2–3 周。时间只是参考，验收结果才决定能否进阶。

## 你现在从哪里开始

按顺序完成 01、02、03，然后做 04 的闭卷任务。之后依次通过阶段 B–E；第一次练习不要要求 Codex 改大型模型训练代码，先从只读调查和文档级小改动开始。

维护者可在 PowerShell 中运行 `./CODEX_MASTERY/verify_course.ps1`，检查 20 章编号、必需章节、代码围栏、本地链接、首页导航和教学大纲。

## 官方资料

课程以当前官方 OpenAI 文档为准：

- [Codex best practices](https://learn.chatgpt.com/guides/best-practices)
- [Prompting Codex](https://learn.chatgpt.com/docs/prompting)
- [Projects and chats](https://learn.chatgpt.com/docs/projects)
- [Long-running work](https://learn.chatgpt.com/docs/long-running-work)
- [Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents)
- [Codex CLI commands](https://learn.chatgpt.com/docs/developer-commands?surface=cli)
- [Git worktrees](https://learn.chatgpt.com/docs/environments/git-worktrees)
- [AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
- [Codex configuration](https://learn.chatgpt.com/docs/config-file/config-basic)
- [Agent approvals and security](https://learn.chatgpt.com/docs/agent-approvals-security)
- [Skills and plugins](https://learn.chatgpt.com/docs/skills-and-plugins)
- [Hooks](https://learn.chatgpt.com/docs/hooks)
- [Non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
- [Scheduled tasks](https://learn.chatgpt.com/docs/automations)
- [Codex threat models](https://learn.chatgpt.com/docs/security/threat-model)
- [Agent internet access](https://learn.chatgpt.com/docs/cloud/internet-access)

产品界面和功能会更新；遇到课程与当前界面不一致时，以官方文档和你实际可调用的能力为准。
