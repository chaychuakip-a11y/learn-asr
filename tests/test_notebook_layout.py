from __future__ import annotations

from pathlib import Path
import re
import unittest

from scripts.notebook_layout import EXECUTED_DIR, NOTEBOOK_DIR, executed_path


CAPSTONE_STEM = "结课项目_实时数字CTC声学引擎_从WAV到流式文本"
ACTIVE_INDEXES = (
    NOTEBOOK_DIR / "README.md",
    NOTEBOOK_DIR / "核心课程索引_第01到41课.md",
    NOTEBOOK_DIR / "PyTorch零基础课程索引.md",
    NOTEBOOK_DIR / "音频零基础课程索引.md",
    NOTEBOOK_DIR / "研究与工程思维课程索引.md",
    NOTEBOOK_DIR / "认知拓展课程索引.md",
    NOTEBOOK_DIR / "逻辑与信息研究进阶课程索引.md",
    NOTEBOOK_DIR / "语言模型零基础_课程索引.md",
    EXECUTED_DIR / "README.md",
)


def paired_sources() -> list[Path]:
    patterns = (
        "[0-9][0-9]_*.ipynb",
        "基础_[0-9][0-9]_*.ipynb",
        "音频基础_[0-9][0-9]_*.ipynb",
        "思维训练_[0-9][0-9]_*.ipynb",
        "认知拓展_[0-9][0-9]_*.ipynb",
        "研究进阶_[0-9][0-9]_*.ipynb",
        "语言模型零基础_[0-9][0-9]_*.ipynb",
        "专题_*.ipynb",
        "学习中枢_*.ipynb",
        "代码伴读_*.ipynb",
        f"{CAPSTONE_STEM}.ipynb",
    )
    found = {path for pattern in patterns for path in NOTEBOOK_DIR.glob(pattern)}
    return sorted(path for path in found if not path.stem.endswith("_已运行"))


class NotebookLayoutTests(unittest.TestCase):
    def test_source_root_contains_no_executed_copies(self) -> None:
        self.assertEqual(list(NOTEBOOK_DIR.glob("*_已运行.ipynb")), [])

    def test_executed_outputs_do_not_publish_local_machine_paths(self) -> None:
        forbidden = ("G:\\learn_asr", "G:\\\\learn_asr", "C:\\Users\\", "C:\\\\Users\\\\")
        failures = []
        for path in EXECUTED_DIR.rglob("*_已运行.ipynb"):
            text = path.read_text(encoding="utf-8")
            if any(marker in text for marker in forbidden):
                failures.append(path.relative_to(NOTEBOOK_DIR).as_posix())
        self.assertEqual(failures, [])

    def test_every_course_source_has_one_categorized_executed_copy(self) -> None:
        sources = paired_sources()
        expected = {executed_path(source).resolve() for source in sources}
        actual = {path.resolve() for path in EXECUTED_DIR.rglob("*_已运行.ipynb")}
        self.assertEqual(len(sources), 101)
        self.assertSetEqual(actual, expected)

    def test_active_indexes_have_stable_names(self) -> None:
        for path in ACTIVE_INDEXES:
            self.assertTrue(path.exists(), path)
        stale = sorted(NOTEBOOK_DIR.glob("课程索引_第01到*课.md"))
        self.assertEqual(stale, [])

    def test_active_markdown_links_resolve(self) -> None:
        link_pattern = re.compile(r"\]\(([^)]+)\)")
        failures: list[str] = []
        for document in ACTIVE_INDEXES:
            text = document.read_text(encoding="utf-8")
            for raw_target in link_pattern.findall(text):
                target = raw_target.split("#", 1)[0].strip()
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (document.parent / target).resolve()
                if not resolved.exists():
                    failures.append(f"{document.relative_to(NOTEBOOK_DIR)} -> {target}")
        self.assertEqual(failures, [])

    def test_historical_indexes_are_archived(self) -> None:
        archive = NOTEBOOK_DIR / "_archive"
        names = {path.name for path in archive.glob("课程索引_第01到*课.md")}
        self.assertSetEqual(
            names,
            {
                "课程索引_第01到24课.md",
                "课程索引_第01到30课.md",
                "课程索引_第01到43课.md",
                "课程索引_第01到45课.md",
            },
        )


if __name__ == "__main__":
    unittest.main()
