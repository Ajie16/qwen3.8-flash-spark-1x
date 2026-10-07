#!/usr/bin/env python3
"""Put the pf=2 tile candidates first so default_config can actually reach them.

The first attempt appended (8,4,*,2) after the existing list, which changed nothing: gate_up's
GLM_GATEUP=(8,4,4,1) and down's GLM_DOWN=(8,4,1,1) both satisfy the divisibility test first, so
the pf=2 entries were never examined. Both attempts measured 769-773 tok/s, which is the baseline.

experts_grouped.cuh:336-338 compiles (8,4,1), (8,4,2) and (4,4,2). SK is a runtime argument, not a
template parameter, so (8,4,4,2) and (8,4,1,2) both hit the single TF_LAUNCH(8,4,2,...) instance.
"""
import shutil
import time
from pathlib import Path

PATH = Path("/home/xujie/tensorfold-venv/lib/python3.12/site-packages/tensorfold/cuda/exl3/experts.py")
STAMP = "%Y%m%d-%H%M%S"

OLD = """    cands = [GLM_GATEUP if gateup else GLM_DOWN, (8, 4, 2, 1), (8, 4, 1, 1), (4, 4, 2, 2), (4, 4, 1, 2),
             (8, 4, 4, 2), (8, 4, 2, 2), (8, 4, 1, 2)]   # pf=2: experts_grouped.cuh:337 compiles it; nothing selected it"""
NEW = """    cands = [(8, 4, 4, 2), (8, 4, 2, 2), (8, 4, 1, 2),   # pf=2 first: experts_grouped.cuh:337 compiles it, earlier lists never reached it
             GLM_GATEUP if gateup else GLM_DOWN, (8, 4, 2, 1), (8, 4, 1, 1), (4, 4, 2, 2), (4, 4, 1, 2)]"""


def main() -> int:
    text = PATH.read_text()
    if NEW in text:
        print("SKIP  already applied")
        return 0
    if text.count(OLD) != 1:
        print(f"FAIL  anchor found {text.count(OLD)} times")
        return 1
    backup = Path(f"/home/xujie/tensorfold-patches/pre-pf2b-{time.strftime(STAMP)}")
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PATH, backup / "cuda__exl3__experts.py")
    print(f"backup: {backup}")
    PATH.write_text(text.replace(OLD, NEW, 1))
    import py_compile
    py_compile.compile(str(PATH), doraise=True)
    print("  ok    pf=2 candidates now precede the pf=1 ones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())