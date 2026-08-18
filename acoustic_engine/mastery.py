from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import nbformat

from .challenge import CHALLENGES, coding_passed_ids
from .tutor import QUESTIONS, load_progress, passed_ids


@dataclass(frozen=True)
class MasteryGate:
    name: str
    passed: bool
    evidence: str
    next_action: str


def build_gates(progress: dict[str, object]) -> tuple[MasteryGate, ...]:
    knowledge_count = len(passed_ids(progress))
    coding_count = len(coding_passed_ids(progress))
    capstone = progress.get("capstone_notebook", {})
    capstone_passed = isinstance(capstone, dict) and capstone.get("verified") is True
    oral = progress.get("oral_exam", {})
    oral_passed = isinstance(oral, dict) and oral.get("verified") is True
    migration = progress.get("migration_task", {})
    migration_passed = isinstance(migration, dict) and migration.get("verified") is True
    return (
        MasteryGate(
            "知识检查",
            knowledge_count == len(QUESTIONS),
            f"{knowledge_count}/{len(QUESTIONS)}",
            "uv run python -m acoustic_engine.tutor",
        ),
        MasteryGate(
            "亲手编码",
            coding_count == len(CHALLENGES),
            f"{coding_count}/{len(CHALLENGES)}",
            "uv run python -m acoustic_engine.challenge --status",
        ),
        MasteryGate(
            "结课 Notebook",
            capstone_passed,
            str(capstone.get("path", "尚无已执行证据")) if isinstance(capstone, dict) else "尚无已执行证据",
            "完整运行并保存结课 Notebook，再用 --verify-notebook 验证",
        ),
        MasteryGate(
            "迁移与故障注入",
            migration_passed,
            str(migration.get("evidence", "尚未验证")) if isinstance(migration, dict) else "尚未验证",
            "在新 WAV/chunk/错误配置上完成迁移和排错任务",
        ),
        MasteryGate(
            "闭卷口述答辩",
            oral_passed,
            str(oral.get("evidence", "尚未验证")) if isinstance(oral, dict) else "尚未验证",
            "在对话中闭卷解释完整数据流、三类状态、指标和证据边界",
        ),
    )


def format_report(progress: dict[str, object]) -> str:
    gates = build_gates(progress)
    lines = ["ASR 掌握度门禁"]
    for gate in gates:
        lines.append(f"{'通过' if gate.passed else '待完成'}｜{gate.name}｜{gate.evidence}")
        if not gate.passed:
            lines.append(f"  下一步：{gate.next_action}")
    passed = sum(gate.passed for gate in gates)
    lines.append(f"总门禁：{passed}/{len(gates)}")
    if passed != len(gates):
        lines.append("结论：尚不能证明已经学会；引擎可运行不等于学习者掌握。")
    else:
        lines.append("结论：五类证据齐全，可以进入最终人工审计。")
    return "\n".join(lines)


def validate_executed_notebook(path: Path) -> dict[str, object]:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells:
        raise ValueError("Notebook 没有代码单元")
    unexecuted = [index for index, cell in enumerate(code_cells) if cell.execution_count is None]
    if unexecuted:
        raise ValueError(f"仍有未执行代码单元: {unexecuted}")
    errors = []
    for index, cell in enumerate(code_cells):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(f"cell {index}: {output.get('ename')}: {output.get('evalue')}")
    if errors:
        raise ValueError("Notebook 保存了执行错误: " + "; ".join(errors))
    raw = path.read_bytes()
    return {
        "verified": True,
        "path": str(path.resolve()),
        "code_cells": len(code_cells),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "verified_at": datetime.now().isoformat(timespec="seconds"),
    }


def record_capstone(progress_path: Path, evidence: dict[str, object]) -> None:
    progress = load_progress(progress_path)
    progress["capstone_notebook"] = evidence
    progress["updated_at"] = evidence["verified_at"]
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def run_system_audit(root: Path) -> bool:
    commands = (
        [sys.executable, "scripts/validate_course.py"],
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_acoustic_engine.py"],
    )
    passed = True
    for command in commands:
        print("运行：", " ".join(command))
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        passed = passed and result.returncode == 0
    return passed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR 学习掌握度门禁")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--status", action="store_true", help="显示五类掌握证据")
    actions.add_argument("--verify-notebook", type=Path, metavar="PATH", help="验证并记录已执行结课 Notebook")
    actions.add_argument("--system-audit", action="store_true", help="运行课程和引擎自动化审计")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    progress_path = root / "learning_progress.json"
    if args.verify_notebook:
        try:
            evidence = validate_executed_notebook(args.verify_notebook)
        except (OSError, ValueError, nbformat.ValidationError) as exc:
            print("Notebook 验证失败：", exc)
            return 1
        record_capstone(progress_path, evidence)
        print(f"Notebook 验证通过：{evidence['code_cells']} 个代码单元，已保存学习证据。")
        return 0
    if args.system_audit:
        return 0 if run_system_audit(root) else 1
    print(format_report(load_progress(progress_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
