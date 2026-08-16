# 贡献课程内容

## 开发环境

```powershell
uv sync --locked
uv run jupyter lab
```

## 修改 Notebook

1. 只修改无“已运行”后缀的源 Notebook。
2. 保留课号、标题和 `course` metadata。
3. 新概念必须配最小数值例子、边界条件和至少一道排错题。
4. 图必须标注数量、单位和坐标轴；不能只依赖颜色表达含义。
5. 教学模拟、训练集过拟合和生产结果必须明确区分。
6. 不提交密钥、个人音频或无许可数据。

## 提交前检查

```powershell
uv run python scripts/enrich_course.py
uv run python scripts/validate_course.py
```

若代码发生变化，应重新执行受影响 Notebook，并更新对应 `_已运行.ipynb`。

## Commit 建议

- `course: expand CTC forward exercises`
- `fix: correct streaming cache boundary`
- `docs: clarify FSDD attribution`
- `ci: validate notebook structure`

一次提交只处理一个主题，避免把课程内容、依赖升级和大规模输出混在一起。

除非贡献中另有明确且被接受的说明，代码贡献按 Apache-2.0 提交，课程内容贡献按 CC BY 4.0 提交。第三方材料必须单独说明来源和许可证。
