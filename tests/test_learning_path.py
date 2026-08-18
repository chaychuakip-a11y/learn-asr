from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROADMAP = ROOT / "LEARNING_PATH.md"


class LearningPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = ROADMAP.read_text(encoding="utf-8")

    def test_all_dependency_stages_are_present_once(self) -> None:
        headings = re.findall(r"^### 阶段 (\d+)：", self.text, flags=re.MULTILINE)
        self.assertEqual(headings, [str(number) for number in range(13)])

    def test_user_priority_topics_are_mandatory_and_routed(self) -> None:
        for topic in (
            "CTC",
            "流式",
            "PGS",
            "RTF",
            "语言模型",
            "WFST",
            "量化",
            "部署",
            "麦克风前端",
            "语义",
        ):
            self.assertIn(topic, self.text)
        self.assertIn("全部属于必修主干", self.text)

    def test_every_local_markdown_link_resolves(self) -> None:
        failures: list[str] = []
        for raw_target in re.findall(r"\]\(([^)]+)\)", self.text):
            target = raw_target.split("#", 1)[0].strip()
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (ROADMAP.parent / target).resolve().exists():
                failures.append(target)
        self.assertEqual(failures, [])

    def test_repository_entry_points_use_the_unique_path(self) -> None:
        root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        notebook_readme = (ROOT / "notebooks" / "README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("[唯一学习路径](LEARNING_PATH.md)", root_readme)
        self.assertIn("[`LEARNING_PATH.md`](../LEARNING_PATH.md)", notebook_readme)


if __name__ == "__main__":
    unittest.main()
