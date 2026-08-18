"""Canonical paths for source and executed course notebooks.

Source notebooks intentionally stay directly under ``notebooks/``.  Several
lessons support being launched with that directory as the kernel working
directory, so moving the sources would silently break their data paths.
Executed reference copies are generated below ``notebooks/_executed`` and are
grouped by learning track to keep the source directory readable.
"""

from __future__ import annotations

from pathlib import Path
import re

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
EXECUTED_DIR = NOTEBOOK_DIR / "_executed"
ARCHIVE_DIR = NOTEBOOK_DIR / "_archive"

EXECUTED_CATEGORIES = {
    "asr_core": "ASR 主线 01～46",
    "pytorch_foundations": "Python/PyTorch 桥梁课",
    "language_models": "语言模型专修课",
    "labs": "学习中枢与专题实验室",
    "capstone": "结课项目",
}

CAPSTONE_STEM = "结课项目_实时数字CTC声学引擎_从WAV到流式文本"

WINDOWS_USER_HOME = re.compile(r"(?i)[A-Z]:\\Users\\[^\\\r\n]+")


def executed_category(source: Path | str) -> str:
    """Return the stable executed-copy category for a source notebook."""

    stem = Path(source).stem
    if re.match(r"^\d{2}_", stem):
        return "asr_core"
    if stem.startswith(("基础_", "代码伴读_")):
        return "pytorch_foundations"
    if stem.startswith("语言模型零基础_"):
        return "language_models"
    if stem == CAPSTONE_STEM:
        return "capstone"
    return "labs"


def executed_path(source: Path | str) -> Path:
    """Map one source notebook to its canonical executed reference copy."""

    source_path = Path(source)
    return (
        EXECUTED_DIR
        / executed_category(source_path)
        / f"{source_path.stem}_已运行.ipynb"
    )


def ensure_executed_directories() -> None:
    """Create only the known, bounded output directories."""

    for category in EXECUTED_CATEGORIES:
        (EXECUTED_DIR / category).mkdir(parents=True, exist_ok=True)


def _sanitize_output_value(value: object) -> tuple[object, int]:
    if isinstance(value, str):
        sanitized = value.replace(str(ROOT), "<REPO_ROOT>")
        sanitized = sanitized.replace(str(ROOT).replace("\\", "/"), "<REPO_ROOT>")
        sanitized = WINDOWS_USER_HOME.sub("<USER_HOME>", sanitized)
        return sanitized, int(sanitized != value)
    if isinstance(value, list):
        changed = 0
        sanitized_items = []
        for item in value:
            sanitized, item_changed = _sanitize_output_value(item)
            sanitized_items.append(sanitized)
            changed += item_changed
        return sanitized_items, changed
    if isinstance(value, dict):
        changed = 0
        sanitized_mapping = {}
        for key, item in value.items():
            sanitized, item_changed = _sanitize_output_value(item)
            sanitized_mapping[key] = sanitized
            changed += item_changed
        return sanitized_mapping, changed
    return value, 0


def sanitize_notebook_outputs(notebook: nbformat.NotebookNode) -> int:
    """Remove machine-specific paths from saved outputs, preserving evidence."""

    sanitized_metadata, changed = _sanitize_output_value(dict(notebook.metadata))
    notebook["metadata"] = nbformat.from_dict(sanitized_metadata)
    for cell in notebook.cells:
        sanitized_cell_metadata, metadata_changes = _sanitize_output_value(
            dict(cell.metadata)
        )
        cell["metadata"] = nbformat.from_dict(sanitized_cell_metadata)
        changed += metadata_changes
        if cell.cell_type != "code":
            continue
        sanitized, output_changes = _sanitize_output_value(list(cell.get("outputs", [])))
        cell["outputs"] = [nbformat.from_dict(output) for output in sanitized]
        changed += output_changes
    return changed
