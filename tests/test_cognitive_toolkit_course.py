from __future__ import annotations

from pathlib import Path
import re
import unittest

import nbformat

from scripts.notebook_layout import executed_path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


class CognitiveToolkitCourseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = sorted(NOTEBOOKS.glob("认知拓展_[0-9][0-9]_*.ipynb"))

    def test_eight_ordered_lessons_exist(self) -> None:
        numbers = [int(re.match(r"认知拓展_(\d\d)_", path.name).group(1)) for path in self.sources]
        self.assertEqual(numbers, list(range(1, 9)))

    def test_each_lesson_has_sources_boundaries_and_transfer(self) -> None:
        required_markers = (
            "课前预测",
            "一手资料与课程取舍",
            "误用警报",
            "适用边界",
            "迁移练习",
            "闭卷挑战",
            "最小掌握门禁",
            "最强替代解释：",
            "什么结果会让我改主意：",
        )
        for lesson, path in enumerate(self.sources, start=1):
            notebook = nbformat.read(path, as_version=4)
            metadata = notebook.metadata.get("course", {})
            self.assertEqual(metadata.get("track"), "cognitive_toolkit", path.name)
            self.assertEqual(metadata.get("lesson"), lesson, path.name)
            self.assertEqual(
                metadata.get("evidence_model"),
                "question-model-prediction-evidence-boundary-update",
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

    def test_handbook_covers_tools_and_boundaries(self) -> None:
        handbook = (ROOT / "COGNITIVE_TOOLKIT.md").read_text(encoding="utf-8")
        for topic in (
            "论证结构与反例",
            "自然频数",
            "碰撞点",
            "存量、流量和延迟",
            "信息价值",
            "Shannon 熵",
            "Brier",
            "提取和迁移",
            "指标、代理与激励",
            "可复现性与可重复性",
        ):
            self.assertIn(topic, handbook)
        self.assertIn("这套材料没有承诺什么", handbook)


if __name__ == "__main__":
    unittest.main()
