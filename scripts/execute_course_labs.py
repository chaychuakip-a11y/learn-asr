from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import nbformat
from nbclient import NotebookClient

from notebook_layout import (
    ensure_executed_directories,
    executed_path,
    sanitize_notebook_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
HEADLESS_RUNNER = ROOT / "scripts" / "run_notebook_headless.py"

# 教学 Notebook 的矩阵都很小。Windows 上多个 Jupyter 内核同时初始化
# OpenMP/MKL 时可能发生运行时死锁，因此在启动子进程或内核前固定单线程。
# 用户显式设置过的值仍然优先。
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

CORE_LABS = [
    "学习中枢_诊断与掌握度仪表盘.ipynb",
    "专题_CTC可视化实验室_从路径到流式解码.ipynb",
    "专题_流式ASR实验室_Chunk缓存PGS与实时率.ipynb",
    "专题_WFST实验室_从L与G到流式TokenPassing.ipynb",
    "专题_量化部署实验室_ONNX_INT8性能与服务验收.ipynb",
    "专题_音频前端实验室_质量VAD_AEC与波束形成.ipynb",
    "专题_语义后处理实验室_时间戳ITN置信度与安全执行.ipynb",
    "结课项目_实时数字CTC声学引擎_从WAV到流式文本.ipynb",
]

EXTERNAL_DATA_LABS = [
    "专题_FSDD说话人泛化实验_数据划分增强与盲测.ipynb",
    "专题_FSDD六折LOSO_嵌套选择与说话人统计.ipynb",
    "专题_AudioMNIST外部盲测_冻结协议跨域失败与适配边界.ipynb",
]

def execute(path: Path, timeout: int, write_executed: bool = False) -> float:
    started = time.perf_counter()
    if not write_executed:
        environment = os.environ.copy()
        environment.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"})
        result = subprocess.run(
            [sys.executable, str(HEADLESS_RUNNER), str(path)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        if result.returncode:
            detail = (result.stdout + "\n" + result.stderr).strip()
            raise RuntimeError(detail[-6000:])
        return time.perf_counter() - started

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    if write_executed:
        ensure_executed_directories()
        sanitize_notebook_outputs(notebook)
        nbformat.write(notebook, executed_path(path))
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute course labs in memory without changing source notebooks."
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Also execute FSDD/AudioMNIST labs; first prepare .local_data as documented.",
    )
    parser.add_argument(
        "--write-executed",
        action="store_true",
        help="Write each successful run to its categorized notebooks/_executed copy.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-notebook timeout normally; per-cell timeout with --write-executed.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="TEXT",
        help="Execute only notebook names containing TEXT; may be repeated.",
    )
    args = parser.parse_args()

    selected = CORE_LABS + (EXTERNAL_DATA_LABS if args.external else [])
    if args.only:
        selected = [name for name in selected if any(text in name for text in args.only)]
        if not selected:
            parser.error("--only did not match any selected notebook")
    failures: list[tuple[str, str]] = []
    for index, name in enumerate(selected, start=1):
        path = NOTEBOOKS / name
        print(f"[{index}/{len(selected)}] {name}", flush=True)
        if not path.exists():
            failures.append((name, "file does not exist"))
            continue
        try:
            elapsed = execute(path, args.timeout, write_executed=args.write_executed)
            suffix = (
                f", wrote {executed_path(path).relative_to(ROOT)}"
                if args.write_executed
                else ""
            )
            print(f"  PASS ({elapsed:.1f}s{suffix})", flush=True)
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL: {type(exc).__name__}: {exc}", flush=True)

    if failures:
        print("COURSE LAB EXECUTION FAILED")
        for name, reason in failures:
            print(f"- {name}: {reason}")
        return 1
    print(f"COURSE LAB EXECUTION PASSED: {len(selected)} notebooks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
