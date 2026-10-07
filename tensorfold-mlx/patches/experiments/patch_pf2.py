#!/usr/bin/env python3
"""Revert the nt=16 experiment and enable the already-compiled pf=2 tile.

nt=16 is not instantiated: experts_grouped.cuh:336-338 only compiles (8,4,1), (8,4,2) and
(4,4,2), so GLM_DOWN=(16,4,1,1) made every routed() call raise
  RuntimeError: unsupported tile setting nt=16 warps=4 pf=1

But that dispatch table also shows (8,4,2) IS compiled, and default_config can never return it -
every candidate in its list carries pf=1. SK is a runtime argument rather than a template
parameter (grouped_kernel<CB,NT,W,PF,LO,HI>), so gate_up and down already share one instance;
moving both to pf=2 needs no new CUDA at all, only a different candidate list. PF is the
register-level prefetch depth in warp_tiles, so this is the cheapest pipelining knob available.
"""
import shutil
import sys
import time
from pathlib import Path

PATH = Path("/home/xujie/tensorfold-venv/lib/python3.12/site-packages/tensorfold/cuda/exl3/experts.py")

REVERT_NT = "GLM_DOWN = (16, 4, 1, 1)      # N tile 256 instead of 128; legal for down (K=640, N=2560)"
GLM_DOWN = "GLM_DOWN = (8, 4, 1, 1)"

OLD_CANDS = "cands = [GLM_GATEUP if gateup else GLM_DOWN, (8, 4, 2, 1), (8, 4, 1, 1), (4, 4, 2, 2), (4, 4, 1, 2)]"
NEW_CANDS = ("cands = [GLM_GATEUP if gateup else GLM_DOWN, (8, 4, 2, 1), (8, 4, 1, 1), (4, 4, 2, 2), (4, 4, 1, 2),\n"
             "             (8, 4, 4, 2), (8, 4, 2, 2), (8, 4, 1, 2)]   # pf=2: experts_grouped.cuh:337 compiles it; nothing selected it")


def main() -> int:
    text = PATH.read_text()
    if "(8, 4, 1, 2)]" in text:
        print("SKIP  already applied")
        return 0
    changed = []
    if REVERT_NT in text:
        text = text.replace(REVERT_NT, GLM_DOWN, 1)
        changed.append("reverted GLM_DOWN to (8,4,1,1)")
    if text.count(OLD_CANDS) != 1:
        print(f"FAIL  candidate list anchor found {text.count(OLD_CANDS)} times")
        return 1
    text = text.replace(OLD_CANDS, NEW_CANDS, 1)
    changed.append("default_config now reaches pf=2 tiles")

    backup = Path(f"/home/xujie/tensorfold-patches/pre-pf2-{time.strftime('%Y%m%d-%H%M%S')}")
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PATH, backup / "cuda__exl3__experts.py")
    print(f"backup: {backup}")
    PATH.write_text(text)
    import py_compile
    py_compile.compile(str(PATH), doraise=True)
    for line in changed:
        print(f"  ok    {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())