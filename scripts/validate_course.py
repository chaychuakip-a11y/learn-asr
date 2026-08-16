from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import nbformat


ROOT=Path(__file__).resolve().parents[1]
NB_DIR=ROOT/"notebooks"
EXPECTED=set(range(1,42))
REQUIRED=["README.md","pyproject.toml","uv.lock","DATA_SOURCES.md","COURSE_AUDIT.md","LICENSE","LICENSE-CONTENT","LICENSE-SCOPE.md","NOTICE","notebooks/课程索引_第01到41课.md"]


def lesson_number(path:Path):
    m=re.match(r"(\d\d)_",path.name)
    return int(m.group(1)) if m else None


def validate_notebook(path:Path,require_upgrade:bool):
    errors=[]
    try:
        nb=nbformat.read(path,as_version=4)
        nbformat.validate(nb)
    except Exception as exc:
        return [f"{path.name}: invalid notebook: {exc}"]
    if not nb.cells or nb.cells[0].cell_type!="markdown" or not nb.cells[0].source.lstrip().startswith("#"):
        errors.append(f"{path.name}: first cell must be a Markdown heading")
    if require_upgrade:
        tags=sum("course-upgrade-v2" in c.metadata.get("tags",[]) for c in nb.cells)
        if tags!=4:errors.append(f"{path.name}: expected 4 course-upgrade cells, got {tags}")
        if "course" not in nb.metadata:errors.append(f"{path.name}: missing course metadata")
    for i,c in enumerate(nb.cells):
        for output in c.get("outputs",[]):
            if output.get("output_type")=="error":
                errors.append(f"{path.name}: cell {i} stores error {output.get('ename')}: {output.get('evalue')}")
    return errors


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-only",action="store_true",help="Do not require or inspect executed copies")
    args=parser.parse_args()
    errors=[]
    for rel in REQUIRED:
        if not (ROOT/rel).exists():errors.append(f"missing required file: {rel}")
    sources=sorted(p for p in NB_DIR.glob("[0-9][0-9]_*.ipynb") if not p.stem.endswith("_已运行"))
    numbers=[lesson_number(p) for p in sources]
    if set(numbers)!=EXPECTED:
        errors.append(f"lesson numbers mismatch: missing={sorted(EXPECTED-set(numbers))}, extra={sorted(set(numbers)-EXPECTED)}")
    if len(numbers)!=len(set(numbers)):errors.append("duplicate source lesson numbers")
    for path in sources:errors.extend(validate_notebook(path,True))
    if not args.source_only:
        executed=sorted(NB_DIR.glob("[0-9][0-9]_*_已运行.ipynb"))
        if {lesson_number(p) for p in executed}!=EXPECTED:errors.append("executed notebook set is incomplete")
        for path in executed:errors.extend(validate_notebook(path,False))
    for path in ROOT.rglob("*"):
        if ".venv" in path.parts or not path.is_file():continue
        if path.stat().st_size>95*1024*1024:errors.append(f"file is close to GitHub 100 MB limit: {path.relative_to(ROOT)}")
    if errors:
        print("COURSE VALIDATION FAILED")
        for error in errors:print("-",error)
        return 1
    print(f"COURSE VALIDATION PASSED: {len(sources)} source lessons"+("" if args.source_only else ", 41 executed copies"))
    return 0


if __name__=="__main__":sys.exit(main())
