from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
import warnings

import matplotlib
import nbformat


matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute notebook code cells in one Python namespace.")
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()

    path = args.notebook.resolve()
    notebook = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__", "__file__": str(path)}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or "skip-execution" in cell.metadata.get("tags", []):
            continue
        try:
            exec(compile(cell.source, f"{path.name}:cell-{index}", "exec"), namespace)
        except Exception:
            print(f"FAILED {path.name} cell {index}", file=sys.stderr)
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
