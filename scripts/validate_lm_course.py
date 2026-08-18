from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import nbformat

from notebook_layout import executed_path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"

LESSONS = [
    "语言模型零基础_01_从计数到Bigram",
    "语言模型零基础_02_平滑回退OOV与困惑度",
    "语言模型零基础_03_FSA_FST与第一张OpenFst图",
    "语言模型零基础_04_OpenFst组合确定化与最小化",
    "语言模型零基础_05_从语料到ARPA与Gfst",
    "语言模型零基础_06_词典L消歧与HCLG_CTC_TLG",
    "语言模型零基础_07_Nbest_Lattice分数融合与二遍重打分",
    "语言模型零基础_08_综合项目与闭卷验收",
    "语言模型零基础_09_前沿ASR语言模型系统设计实验室",
]

REQUIRED_DOCUMENTS = [
    "README.md",
    "ASR_LM_ENVIRONMENT.md",
    "ASR_LM_OPENFST_KENLM_CHEATSHEET.md",
    "FRONTIER_ASR_LM_READING.md",
    "notebooks/语言模型零基础_课程索引.md",
]

SOURCE_MARKERS = {
    1: ["Bigram", "Add-k", "自动判题", "离场票"],
    2: ["回退", "困惑度", "<unk>", "自动判题"],
    3: ["OpenFst", "Tropical", "最短路径", "自动判题"],
    4: ["Composition", "fstarcsort", "Determinize", "Minimize"],
    5: ["KenLM", "ARPA", "G.fst", "自动判题"],
    6: ["#0", "#1", "HCLG", "CTC-TLG"],
    7: ["N-best", "Lattice", "Oracle WER", "二遍重打分"],
    8: ["开发集", "测试集", "闭卷代码题", "OpenFst"],
    9: ["contextual ASR", "Coverage", "Pareto", "实验计划"],
}

EXECUTION_EVIDENCE = {
    3: ["默认路径顺序校验通过"],
    4: ["默认组合最短路径校验通过"],
    5: ["ARPA=4.553500 FST=4.553500"],
    6: ["删除 #0:#0 后： 路径消失"],
    7: ["oracle WER@8 = 0.000"],
    8: ["选择 LM scale=0.7", "融合系统 test WER=0.286"],
    9: ['"status": "planned-not-run"', "threshold=0.9 coverage=0.30 risk=0.00"],
}

RUNTIME_ARTIFACTS = [
    "openfst_lab/lesson05/words.txt",
    "openfst_lab/lesson05/tiny.1gram.arpa",
    "openfst_lab/lesson05/tiny.2gram.arpa",
    "openfst_lab/lesson05/tiny.3gram.arpa",
    "openfst_lab/lesson05/G.kaldi.fst",
    "openfst_lab/lesson06/L.fst",
    "openfst_lab/lesson06/LG.fst",
    "openfst_lab/lesson07/tiny_lattice.fst",
    "openfst_lab/lesson08/capstone_report.json",
    "openfst_lab/lesson09/frontier_experiment_plan.json",
]


def notebook_output_text(notebook: nbformat.NotebookNode) -> str:
    chunks: list[str] = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "stream":
                chunks.append(str(output.get("text", "")))
            elif output_type in {"execute_result", "display_data"}:
                value = output.get("data", {}).get("text/plain", "")
                chunks.append("".join(value) if isinstance(value, list) else str(value))
    return "\n".join(chunks)


def read_notebook(path: Path, errors: list[str]) -> nbformat.NotebookNode | None:
    try:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        return notebook
    except Exception as exc:
        errors.append(f"{path.relative_to(ROOT)}: invalid notebook: {exc}")
        return None


def validate_cell_ids(
    path: Path, notebook: nbformat.NotebookNode, errors: list[str]
) -> None:
    ids = [cell.get("id") for cell in notebook.cells]
    missing = [index for index, cell_id in enumerate(ids) if not cell_id]
    duplicates = sorted({cell_id for cell_id in ids if cell_id and ids.count(cell_id) > 1})
    if missing:
        errors.append(f"{path.name}: missing cell ids at indexes {missing}")
    if duplicates:
        errors.append(f"{path.name}: duplicate cell ids {duplicates}")


def validate_pair(lesson: int, stem: str, errors: list[str]) -> tuple[int, int]:
    source_path = NOTEBOOK_DIR / f"{stem}.ipynb"
    executed_output = executed_path(source_path)
    for path in (source_path, executed_output):
        if not path.exists():
            errors.append(f"missing notebook: {path.relative_to(ROOT)}")
    if not source_path.exists() or not executed_output.exists():
        return 0, 0

    source = read_notebook(source_path, errors)
    executed = read_notebook(executed_output, errors)
    if source is None or executed is None:
        return 0, 0

    validate_cell_ids(source_path, source, errors)
    validate_cell_ids(executed_output, executed, errors)

    if len(source.cells) != len(executed.cells):
        errors.append(
            f"lesson {lesson}: source/executed cell count differs "
            f"({len(source.cells)} != {len(executed.cells)})"
        )
    else:
        for index, (source_cell, executed_cell) in enumerate(
            zip(source.cells, executed.cells, strict=True)
        ):
            if (
                source_cell.cell_type != executed_cell.cell_type
                or source_cell.source != executed_cell.source
            ):
                errors.append(
                    f"lesson {lesson}: source differs from executed copy at cell {index}"
                )
                break

    source_code = [cell for cell in source.cells if cell.cell_type == "code"]
    executed_code = [cell for cell in executed.cells if cell.cell_type == "code"]
    dirty = [
        index
        for index, cell in enumerate(source_code)
        if cell.execution_count is not None or cell.outputs
    ]
    if dirty:
        errors.append(f"{source_path.name}: source code cells store outputs {dirty}")

    unexecuted = [
        index
        for index, cell in enumerate(executed_code)
        if cell.execution_count is None
    ]
    if unexecuted:
        errors.append(f"{executed_output.name}: unexecuted code cells {unexecuted}")

    for cell_index, cell in enumerate(executed.cells):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(
                    f"{executed_output.name}: cell {cell_index} stores "
                    f"{output.get('ename')}: {output.get('evalue')}"
                )

    source_text = "\n".join(cell.source for cell in source.cells)
    for marker in SOURCE_MARKERS[lesson]:
        if marker not in source_text:
            errors.append(f"{source_path.name}: missing course marker {marker!r}")

    output_text = notebook_output_text(executed)
    for marker in EXECUTION_EVIDENCE.get(lesson, []):
        if marker not in output_text:
            errors.append(f"{executed_output.name}: missing execution evidence {marker!r}")

    return len(source.cells), len(source_code)


def validate_frontier_reading(errors: list[str]) -> None:
    path = ROOT / "FRONTIER_ASR_LM_READING.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    paper_numbers = [int(value) for value in re.findall(r"(?m)^(\d+)\. \[", text)]
    if paper_numbers != list(range(1, 18)):
        errors.append(
            "FRONTIER_ASR_LM_READING.md: expected consecutively numbered papers 1..17, "
            f"got {paper_numbers}"
        )
    for marker in [
        "截至 2026-08-18",
        "音频条件",
        "检索式上下文",
        "幻觉",
        "coverage-risk",
    ]:
        if marker not in text:
            errors.append(f"FRONTIER_ASR_LM_READING.md: missing marker {marker!r}")


def run_runtime_checks(errors: list[str]) -> None:
    commands = [
        (["wsl", "-d", "Ubuntu", "--", "which", "fstcompile"], "fstcompile"),
        (["wsl", "-d", "Ubuntu", "--", "which", "fstcompose"], "fstcompose"),
        (
            ["wsl", "-d", "Ubuntu", "--", "test", "-x", "/opt/kenlm/build/bin/lmplz"],
            "KenLM lmplz",
        ),
        (
            [
                "wsl",
                "-d",
                "Ubuntu",
                "--",
                "test",
                "-x",
                "/opt/kenlm/build/bin/build_binary",
            ],
            "KenLM build_binary",
        ),
        (
            [
                "wsl",
                "-d",
                "Ubuntu",
                "--",
                "/opt/kaldilm-venv/bin/python",
                "-m",
                "kaldilm",
                "--help",
            ],
            "kaldilm",
        ),
    ]
    for command, label in commands:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"runtime tool check failed for {label}: {exc}")
            continue
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no output"
            errors.append(f"runtime tool check failed for {label}: {detail}")

    for relative in RUNTIME_ARTIFACTS:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty runtime artifact: {relative}")

    plan_path = ROOT / "openfst_lab/lesson09/frontier_experiment_plan.json"
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid frontier experiment plan: {exc}")
        else:
            if plan.get("status") != "planned-not-run":
                errors.append(
                    "frontier experiment plan must remain honestly labeled 'planned-not-run'"
                )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the standalone 9-lesson ASR language-model course."
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="also check WSL tools and artifacts generated by lessons 5-9",
    )
    args = parser.parse_args()

    errors: list[str] = []
    for relative in REQUIRED_DOCUMENTS:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required document: {relative}")

    total_cells = 0
    total_code_cells = 0
    for lesson, stem in enumerate(LESSONS, start=1):
        cells, code_cells = validate_pair(lesson, stem, errors)
        total_cells += cells
        total_code_cells += code_cells

    validate_frontier_reading(errors)
    if args.runtime:
        run_runtime_checks(errors)

    if errors:
        print("LM COURSE VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    suffix = ", WSL tools and generated artifacts" if args.runtime else ""
    print(
        "LM COURSE VALIDATION PASSED: "
        f"9 source + 9 executed notebooks, {total_cells} paired cells, "
        f"{total_code_cells} code cells, 0 stored execution errors{suffix}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
