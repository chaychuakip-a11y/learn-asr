from __future__ import annotations

from pathlib import Path
import re
import unittest

import nbformat

from scripts.notebook_layout import executed_path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


class ResearchThinkingCourseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = sorted(NOTEBOOKS.glob("思维训练_[0-9][0-9]_*.ipynb"))

    def test_six_ordered_lessons_exist(self) -> None:
        numbers = [int(re.match(r"思维训练_(\d\d)_", path.name).group(1)) for path in self.sources]
        self.assertEqual(numbers, list(range(1, 7)))

    def test_each_lesson_uses_the_evidence_loop_and_primary_sources(self) -> None:
        required_markers = (
            "课前预测",
            "一手资料与课程取舍",
            "闭卷挑战",
            "主张：",
            "最强替代解释：",
            "什么结果会推翻主张：",
            "适用边界：",
            "最小掌握门禁",
        )
        for lesson, path in enumerate(self.sources, start=1):
            notebook = nbformat.read(path, as_version=4)
            metadata = notebook.metadata.get("course", {})
            self.assertEqual(metadata.get("track"), "research_thinking", path.name)
            self.assertEqual(metadata.get("lesson"), lesson, path.name)
            text = "\n".join(cell.source for cell in notebook.cells)
            for marker in required_markers:
                self.assertIn(marker, text, f"{path.name}: {marker}")
            self.assertIn("https://", text, path.name)
            self.assertGreaterEqual(sum(cell.cell_type == "code" for cell in notebook.cells), 3)

    def test_source_is_clean_and_executed_copy_is_complete(self) -> None:
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

    def test_research_report_contains_ranked_future_directions(self) -> None:
        report = (ROOT / "LEARNING_EXPANSION_RESEARCH.md").read_text(encoding="utf-8")
        for topic in (
            "语音科学与语音学",
            "数据中心 ASR",
            "多语、口音与语言变化",
            "人机交互与语音 UX",
            "隐私、安全与对抗鲁棒",
            "端侧系统与硬件意识",
            "可观测性与 ML 事故响应",
        ):
            self.assertIn(topic, report)
        for authority in ("NIST", "SCTK", "Model Cards", "Datasheets"):
            self.assertIn(authority, report)

    def test_decision_audit_rejects_placeholders(self) -> None:
        decision_notebook = nbformat.read(self.sources[-1], as_version=4)
        text = "\n".join(cell.source for cell in decision_notebook.cells)
        self.assertIn('placeholders = {"", "todo", "tbd", "待补充", "未知"}', text)
        self.assertIn("assert not audit_decision_record(draft)[\"ready\"]", text)


if __name__ == "__main__":
    unittest.main()
