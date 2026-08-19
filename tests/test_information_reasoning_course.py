from __future__ import annotations

from pathlib import Path
import re
import unittest

import nbformat

from scripts.notebook_layout import executed_path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


class InformationReasoningCourseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = sorted(NOTEBOOKS.glob("研究进阶_[0-9][0-9]_*.ipynb"))

    def test_eight_ordered_lessons_exist(self) -> None:
        numbers = [int(re.match(r"研究进阶_(\d\d)_", path.name).group(1)) for path in self.sources]
        self.assertEqual(numbers, list(range(1, 9)))

    def test_each_lesson_is_auditable_and_transferable(self) -> None:
        required_markers = (
            "课前预测",
            "一手资料与课程取舍",
            "核心概念",
            "信息搜集实作",
            "误用警报",
            "适用边界",
            "迁移练习",
            "闭卷挑战",
            "最小掌握门禁",
            "最强替代解释",
            "什么结果会让我改主意",
            "检索式",
            "来源家族",
            "停止规则",
        )
        for lesson, path in enumerate(self.sources, start=1):
            notebook = nbformat.read(path, as_version=4)
            metadata = notebook.metadata.get("course", {})
            self.assertEqual(metadata.get("track"), "information_reasoning", path.name)
            self.assertEqual(metadata.get("lesson"), lesson, path.name)
            self.assertEqual(
                metadata.get("evidence_model"),
                "question-query-provenance-matrix-synthesis-update",
                path.name,
            )
            text = "\n".join(cell.source for cell in notebook.cells)
            for marker in required_markers:
                self.assertIn(marker, text, f"{path.name}: {marker}")
            self.assertIn("https://", text, path.name)
            self.assertGreaterEqual(sum(cell.cell_type == "code" for cell in notebook.cells), 3)

    def test_source_clean_and_executed_copy_complete(self) -> None:
        for source in self.sources:
            source_notebook = nbformat.read(source, as_version=4)
            executed = executed_path(source)
            self.assertTrue(executed.exists(), executed)
            executed_notebook = nbformat.read(executed, as_version=4)
            self.assertEqual(len(source_notebook.cells), len(executed_notebook.cells))
            for source_cell, executed_cell in zip(source_notebook.cells, executed_notebook.cells):
                self.assertEqual(source_cell.source, executed_cell.source)
                if source_cell.cell_type == "code":
                    self.assertIsNone(source_cell.execution_count)
                    self.assertEqual(source_cell.outputs, [])
                    self.assertIsNotNone(executed_cell.execution_count)
                    self.assertFalse(
                        any(output.get("output_type") == "error" for output in executed_cell.outputs)
                    )

    def test_handbook_and_workspace_cover_research_chain(self) -> None:
        handbook = (ROOT / "LOGIC_INFORMATION_RESEARCH.md").read_text(encoding="utf-8")
        for topic in (
            "必要条件与充分条件",
            "量词与作用域",
            "溯因",
            "横向阅读",
            "来源家族",
            "布尔检索",
            "证据矩阵",
            "PRISMA",
            "停止规则",
            "AI 总结",
        ):
            self.assertIn(topic, handbook)
        self.assertTrue((ROOT / "research_workspace" / "example_dossier.json").exists())


if __name__ == "__main__":
    unittest.main()
