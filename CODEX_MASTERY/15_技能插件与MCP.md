# 15：Skill、Plugin 与 MCP 的选择和制作

## 本课目标

当你反复复制同一段提示词，或者 Codex 需要访问 GitHub、Slack、数据库等实时系统时，就要选择扩展层。本课帮助你区分 Skill、Plugin 和 MCP，设计第一个可重复工作流，并避免为了“高级”而过度集成。

## 1. 三者的核心区别

| 能力 | 本质 | 适合什么 |
|---|---|---|
| Skill | 一组可复用指令、参考、脚本和资产 | 固定方法与输出格式 |
| MCP server | 提供实时数据和受控动作的工具服务 | 查询/修改外部系统或专用工具 |
| Plugin | 可安装、分发的能力包，可含 skills、connectors/MCP、hooks 等 | 团队或市场分发完整工作流 |

一句话：Skill 说明“怎样做”，MCP 提供“能查什么、能操作什么”，Plugin 负责“如何打包安装和共享”。

## 2. 与提示词、`AGENTS.md` 的边界

| 需求 | 选择 |
|---|---|
| 只在这次任务使用 | 提示词 |
| 当前仓库每次都要遵守 | `AGENTS.md` |
| 多仓库反复执行同一多步流程 | Skill |
| 读取会变化的私有 GitHub/Slack/数据库数据 | MCP/connector |
| 把 skill、工具、hook 一起分发 | Plugin |

例子：“这个项目使用 `uv`”属于 `AGENTS.md`；“每周把提交整理成固定格式发布说明”属于 Skill；“读取 GitHub PR 实时评论”需要 GitHub connector/MCP。

## 3. Skill 的结构

一个 Skill 是包含 `SKILL.md` 的目录，可以附带：

- `references/`：按需读取的详细资料；
- `scripts/`：稳定、机械、可执行步骤；
- 模板和示例；
- 其他工作流需要的资源。

最小 `SKILL.md`：

```md
---
name: asr-test-audit
description: Audit ASR test evidence when the user asks whether a model, decoder, or audio pipeline change is sufficiently verified.
---

# ASR test audit

1. Identify the observable requirement and affected contract.
2. Map existing unit, integration, and end-to-end evidence.
3. Run only safe, relevant checks authorized by the task.
4. Report proven, contradicted, missing, and untested claims.
5. Never treat a documented command as an executed result.
```

真正 Skill 需要更细的输入、失败处理和输出规范，但第一版应小而聚焦。

## 4. Skill 如何触发

Codex 初始只看到 Skill 的名称和 description，任务匹配后才读取完整 `SKILL.md`。所以 description 必须说清：

- 用户目标；
- 何时应该触发；
- 何时不应触发；
- 常见真实说法或对象。

弱描述：“用于测试。”

强描述：

```text
Audit test evidence for ASR model, decoder, frontend, and streaming changes.
Use when the user asks whether an ASR change is sufficiently tested or wants a requirement-to-test coverage report.
Do not use merely to run an already-specified test command.
```

description 过宽会乱触发，过窄会永远不触发。

## 5. Skill 的安装位置

当前官方约定包括：

- 仓库级：从当前目录向项目根寻找 `.agents/skills/`；
- 个人级：`$HOME/.agents/skills/`；
- 管理员和系统级：由环境提供。

仓库 Skill 适合团队共享并随代码 review；个人 Skill 适合自己的跨项目流程。路径和支持能力可能随 Codex 版本更新，以当前官方文档为准。

## 6. 用 `$skill-creator` 创建第一版

Codex 中可以显式调用：

```text
$skill-creator

创建一个名为 asr-test-audit 的个人 Skill。
目标：当我询问某项 ASR 修改是否验证充分时，生成需求→测试证据映射。
输入：任务说明、相关 diff、测试命令或结果。
输出：已证明、被反驳、缺失、未运行四类结论，附文件和命令证据。
边界：只读审计为默认；除非明确授权，不修改文件、不运行训练、不联网。
请先采访我补齐触发条件和失败处理，再起草并用真实案例测试。
```

不要一开始就加入复杂脚本。先用指令版验证方法有价值；只有重复的机械步骤明显不稳定时才加脚本。

## 7. Skill 的测试集

至少准备：

1. 应自动触发的典型请求；
2. 同义表达；
3. 不应触发的近似请求；
4. 输入缺失、应追问的请求；
5. 工具失败或证据矛盾；
6. 明确手动调用 `$skill-name`。

对每次结果检查：触发是否正确、步骤是否完整、输出格式是否稳定、是否越权、是否承认缺口。

## 8. MCP 是工具边界

MCP server 可以暴露：

- Tools：查询或执行动作；
- Resources：可读取的数据或文档；
- Prompts/其他能力：由实现和客户端支持。

适合 MCP 的情况：

- 数据频繁变化；
- 数据私有且已有授权系统；
- 需要结构化调用而不是复制粘贴；
- 希望服务端执行鉴权、参数校验和审计；
- 多个用户或工作流复用同一集成。

不适合 MCP：

- 几页静态说明可以放在 Skill reference；
- 一次性、本地、简单脚本；
- 只是想让提示词看起来高级；
- 没有明确数据所有者和权限模型。

## 9. 读工具和写工具必须清楚区分

MCP 工具元数据可以声明只读提示，但你仍应从实际行为判断：

- list/get/search 通常倾向只读；
- create/update/delete/send/run/trigger 通常会改变状态；
- “生成预览”可能写日志或创建临时资源；
- 查询也可能暴露敏感数据。

一个安全的写工具应：

- 参数明确且可验证；
- 服务端重新鉴权；
- 使用最小 OAuth scope；
- 对危险动作支持预览或幂等；
- 返回结构化结果和资源 ID；
- 记录审计但不泄露秘密；
- 不把“模型说允许”当成授权。

## 10. Connector 与网页搜索

访问私有 GitHub、Slack、Notion 等工作区数据时，应使用已认证 connector/MCP，而不是网页搜索或模型记忆。网页搜索无法访问可靠的私有权限上下文，也不应被用来绕过登录。

连接服务前检查：

- 请求哪些权限；
- 能读哪些空间；
- 是否能写、发送或删除；
- 数据会传到哪里；
- 谁能撤销连接；
- 当前任务是否真的需要。

## 11. Plugin 何时值得做

先从 Skill 开始，满足以下情况再考虑 Plugin：

- 需要让其他人方便安装；
- 需要同时分发 Skill 和 MCP/connector；
- 需要 hooks、命令或额外资产；
- 有清楚版本、权限和维护责任；
- 已经通过多组真实案例测试。

Plugin 是分发单元，不会自动让一个糟糕工作流变好。

## 12. 安装第三方 Plugin 前的审查

检查：

- 发布者和来源；
- 包含的 skills、MCP、hooks 和 browser 能力；
- OAuth scopes 与外部服务；
- 是否有工作区管理员限制；
- 安装后是否要重启或新开聊天；
- hooks 是否需要信任；
- 卸载是否会保留 connector 授权；
- 数据适用的第三方条款。

先选择一个能消除真实手工循环的插件，不要一次安装全部推荐插件。

## 13. 选择练习

为每个需求选择 Prompt、`AGENTS.md`、Skill、MCP 或 Plugin：

1. “这次不要改测试。”
2. “本仓库始终使用 `uv`。”
3. “每次 ASR 变更生成同格式证据审计。”
4. “读取当前 GitHub PR 的最新评论。”
5. “把审计 Skill、GitHub 工具和审批 Hook 分发给全团队。”
6. “阅读一份不再变化的内部术语表。”

参考：1 Prompt；2 AGENTS；3 Skill；4 MCP/connector；5 Plugin；6 Skill reference 或普通文档。

## 14. 实战：设计但不急着安装

```text
只读分析我过去在 CODEX_MASTERY 学习记录中的重复工作，不创建 Skill、不安装 Plugin。

找出至少重复出现 3 次、输入输出稳定、值得复用的工作流。
为每项判断应该使用 Prompt 模板、AGENTS.md、Skill、MCP 还是 Plugin；
说明选择、拒绝其他层的理由、所需权限和最小测试集。
最后只推荐一个最适合作为第一个 Skill 的候选。
```

确认候选后使用 `$skill-creator`。创建或安装是持久化修改，应检查产生的路径和内容，并用新聊天测试触发。

## 15. 故障排查

Skill 不触发：

- 检查路径、名称和 description；
- 新开会话确认重新发现；
- 用应触发/不应触发对照测试；
- 显式 `$skill-name` 调用以区分发现问题和流程问题。

MCP 不可用：

- 检查服务器是否启用和启动；
- 检查认证、scope 和连接状态；
- 检查客户端是否需要重启；
- 用只读健康/列举工具验证；
- 不要把工具故障误诊为提示词问题。

Plugin 不可用：

- 检查安装、启用、支持界面和新聊天要求；
- 检查 connector 是否单独授权；
- 检查工作区策略；
- 检查 bundled hooks 是否等待信任。

## 本课验收

你必须：

- 对 6 个场景正确选择扩展层；
- 设计一个范围明确的 Skill；
- 写出强 description 和不触发边界；
- 用至少 5 个触发测试验证；
- 解释 Skill、MCP、Plugin 的职责；
- 为一个写 MCP 工具设计授权和审计边界。

只有当 Skill 在新聊天中既能正确触发、又不会误触发，并且结果可重复，才算通过。
