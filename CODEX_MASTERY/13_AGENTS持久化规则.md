# 13：用 `AGENTS.md` 固化仓库规则

## 本课目标

你已经会在提示词中写目标和约束。本课学习把真正长期、仓库相关、反复使用的规则放进 `AGENTS.md`，让新聊天自动获得正确的构建命令、目录地图和工程边界。

学完后，你应能：

1. 判断某条信息该留在当前提示词、普通文档还是 `AGENTS.md`。
2. 理解全局、仓库和子目录指令的发现与覆盖关系。
3. 写出短、准、可执行、可验证的 agent 指令。
4. 验证 Codex 真的加载并正确应用了指令。
5. 通过真实失败复盘逐步改进规则，而不是堆砌愿望。

## 1. `AGENTS.md` 是什么

可以把它理解为“给 coding agent 的仓库操作说明”。Codex 开始工作前会读取适用的 `AGENTS.md`，用于了解：

- 仓库结构与重要目录；
- 安装、运行、测试、格式化命令；
- 编码和架构约定；
- 禁止事项与风险边界；
- 修改完成后如何验证；
- 特定子目录的特殊规则。

它不是替代 README。README 面向人类用户，`AGENTS.md` 面向执行任务的 agent；二者可以引用彼此，但职责不同。

## 2. 哪些内容应该放哪里

| 内容 | 推荐位置 | 原因 |
|---|---|---|
| “这次只修一个测试” | 当前提示词 | 一次性范围 |
| “所有 Python 变更都运行指定测试” | 仓库 `AGENTS.md` | 每次任务重复 |
| “我个人喜欢简短回复” | 全局指导/个人配置 | 跨仓库个人偏好 |
| 用户使用教程 | README/docs | 面向人类读者 |
| 某子系统必须保持流式因果性 | 子目录 `AGENTS.md` | 局部工程约束 |
| 一套可重复审查流程 | Skill | 多步骤工作流 |
| CI 强制格式和测试 | CI/Hook | 需要机械执行，而非文字提醒 |

一条规则若只对当前任务成立，写进仓库级 `AGENTS.md` 会污染未来任务。

## 3. 指令发现与优先级

官方行为可概括为：

1. Codex 先读取 Codex home 下的全局 `AGENTS.override.md`，没有时读取全局 `AGENTS.md`；同一级只取第一个非空文件。
2. 再从项目根目录沿路径走到当前工作目录。
3. 每一级优先寻找 `AGENTS.override.md`，再找 `AGENTS.md`，最后找配置的备用文件名。
4. 越靠近当前目录的指令出现在后面，因此能覆盖较上层指导。
5. 空文件会被忽略，合并内容超过配置上限时会被截断；当前官方默认上限为 32 KiB。

示意：

```text
~/.codex/AGENTS.md                 个人通用规则
G:/learn_asr/AGENTS.md             整个仓库规则
G:/learn_asr/deployment/AGENTS.md  部署目录补充规则
```

当 Codex 从 `deployment/` 工作时，会合并前两层，再应用更具体的部署规则。

## 4. `AGENTS.md` 应写什么

推荐骨架：

```md
# Repository guidance

## Repository map

- `acoustic_engine/`: core offline and streaming recognition code.
- `tests/`: unit and integration tests.
- `notebooks/`: course material; do not rewrite generated outputs manually.

## Environment and commands

- Requires Python 3.13 and `uv`.
- Focused acoustic-engine tests:
  `uv run python -m unittest discover -s tests -p "test_acoustic_engine.py"`

## Change rules

- Preserve public engine and checkpoint contracts unless the task explicitly changes them.
- Keep training, inference, and tests consistent when preprocessing changes.
- Do not regenerate model artifacts unless the task explicitly requires it.

## Verification

- Run the narrowest relevant test first, then the affected test file.
- Report commands, test counts, exit codes, and untested scope.
```

这只是教学示例，不能未经调查直接作为本仓库最终规则。每一项都要与实际代码和团队需求核对。

## 5. 好规则的五个特征

### 具体

弱：“写高质量代码。”

强：“修改音频预处理时，同时检查训练和推理调用者，并运行 `FrontendTests`。”

### 可执行

告诉 Codex 该做什么、使用什么命令，而不只是表达价值观。

### 可验证

规则的应用结果能通过测试、diff 或输出检查。

### 有范围

说明对哪些文件、任务或条件生效。

### 有安全替代

若禁止某行为，尽量给出正确路径。例如“不直接删除模型产物；若确认废弃，先列出引用和恢复方案并请求授权”。

## 6. 常见反模式

### 把所有提示词都塞进去

数百条低价值规则会挤占上下文，互相冲突，关键约束反而不明显。

### 写无法执行的口号

“永远完美”“绝不出错”“使用最佳实践”没有操作定义。

### 复制过期命令

错误测试命令会让每个新聊天重复失败。规则必须实际验证。

### 过度限定实现

把一次成功补丁的每一步固化，可能阻止未来更合适的方案。固定不变式和验收，而不是偶然实现细节。

### 与 CI 重复但更弱

格式、许可证头、禁止提交密钥等机械规则应由工具强制；`AGENTS.md` 可以告诉 Codex如何运行工具，不应成为唯一防线。

### 相互冲突

根规则说“所有测试必须全跑”，子目录又说“禁止运行全量测试”，却没有说明适用条件。Codex只能猜。

## 7. `AGENTS.override.md` 的用途

override 适合临时或更强的替代指导，例如：

- 个人短期实验环境；
- 某个子目录暂时冻结；
- 自动化账户使用不同命令；
- 紧急事故期间禁止部署。

override 不是日常随意堆叠规则的地方。临时条件结束后应删除或更新，否则未来会出现“为什么普通 AGENTS 不生效”的困惑。

## 8. 从真实摩擦生成规则

最可靠的改进循环：

```text
Codex 重复犯错
→ 收集两次以上具体证据
→ 找共同原因
→ 选择正确持久化层
→ 写一条最小规则
→ 用正例、反例、无关例测试
→ 保留、修订或删除
```

不要因为一次偶然失败就立刻添加十条规则。先问：这是提示词不清、仓库文档缺失、工具检查缺失，还是确实需要持久指令？

## 9. 如何验证加载

新会话启动后要求：

```text
只读说明本次会话加载了哪些 agent 指令来源，按从全局到当前目录的顺序列出。
然后总结对当前目录最重要的构建、测试、修改和安全规则；不要执行任务。
```

CLI 还可从目标目录运行只读命令验证。关键是使用**新会话**或重启，因为指令链通常在会话开始时构建。

验证三类行为：

1. 正例：修改相关文件时应触发规则；
2. 安全反例：不应被误判为违规；
3. 无关任务：规则不应制造额外噪音。

## 10. 本仓库的调查练习

先不要创建文件。发送：

```text
只读调查 G:\learn_asr 是否需要仓库级 AGENTS.md，不要创建或修改文件。

从 README.md、CONTRIBUTING.md、pyproject.toml、acoustic_engine/README.md、
tests/ 和现有脚本提取真正重复使用的：目录地图、环境要求、快速测试、
昂贵/危险操作、生成文件边界和完成标准。

每条候选规则给出来源证据、适用范围、为什么不能只留在普通文档、
一个应触发示例和一个不应触发示例。拒绝口号和未经验证的命令。
```

你负责删掉不必要的候选项。目标不是写得多，而是让新聊天少犯高代价错误。

## 11. 起草与盲测

调查后另发：

```text
根据我接受的候选项，在回复中起草一个不超过 120 行的根级 AGENTS.md，
先不要写入文件。按仓库地图、环境命令、修改边界、验证、禁止事项组织。
每条规则必须具体、可执行、可验证；能引用现有文档的不要复制长段内容。
最后列出仍需我决定的问题。
```

确认后才写入。然后开启三个新聊天或用有界只读 subagent 测试：

- 正常修改场景；
- 不应触发特殊规则的文档场景；
- `deployment/` 的局部场景。

记录加载来源、行为和噪音。测试失败时修订一条最小规则。

## 12. 维护清单

- [ ] 命令仍能运行且测试数量合理。
- [ ] 目录和入口没有过期。
- [ ] 没有临时任务规则残留。
- [ ] 根级与子目录规则不冲突。
- [ ] 重要规则未因大小上限被截断。
- [ ] 机械约束有 CI/Hook 等真正执行机制。
- [ ] 新成员能理解规则为何存在。

## 本课验收

提交一个经过调查和三类测试的 `AGENTS.md` 草案，并能回答：

1. 全局、根级、子目录指令如何合并？
2. `AGENTS.override.md` 何时适合使用？
3. 为什么测试命令必须实际验证？
4. 什么内容更适合 Skill 或 CI？
5. 如何证明一条规则既能触发又不过度触发？

在用户确认前不要把教学草案直接写入生产仓库。是否采纳规则本身也是需要审查的工程变更。
