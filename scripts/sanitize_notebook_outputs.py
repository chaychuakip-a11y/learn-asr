"""Sanitize machine-specific paths in every published executed notebook."""

from __future__ import annotations

import nbformat

from notebook_layout import EXECUTED_DIR, sanitize_notebook_outputs


def main() -> None:
    paths = sorted(EXECUTED_DIR.rglob("*_已运行.ipynb"))
    changed_files = 0
    changed_values = 0
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        changes = sanitize_notebook_outputs(notebook)
        if changes:
            nbformat.write(notebook, path)
            changed_files += 1
            changed_values += changes
    print(
        f"SANITIZED NOTEBOOK OUTPUTS: {len(paths)} checked, "
        f"{changed_files} files changed, {changed_values} output values changed"
    )


if __name__ == "__main__":
    main()
