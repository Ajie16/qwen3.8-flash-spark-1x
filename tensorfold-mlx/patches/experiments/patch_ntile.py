#!/usr/bin/env python3
"""Try widening the down-projection N tile from 128 to 256 columns.

GLM_DOWN = (nt=8, w=4, sk=1, pf=1) pins the down projection's N tile at 8*16 = 128 columns.
nt=16 is arithmetically legal for this model (K=640 % 64 == 0, N=2560 % 256 == 0) and costs
128 accumulators per thread instead of 64.

The gate_up projection cannot widen at all: its N is 640, and default_config requires
N % (16*nt) == 0, so nt <= 8. For reference, exllamav3 sits on the same N_TILE=128 for this
model for the same reason (exl3_moe.cu notes intermediate_dim 640 % 256 != 0).

This changes both prefill and decode: the Scratch holding cfg_d is built once for the model,
so the tile is shared. Measure both.
"""
import shutil
import sys
import time
from pathlib import Path

PATH = Path("/home/xujie/tensorfold-venv/lib/python3.12/site-packages/tensorfold/cuda/exl3/experts.py")
OLD = "GLM_DOWN = (8, 4, 1, 1)"
NEW = "GLM_DOWN = (16, 4, 1, 1)      # N tile 256 instead of 128; legal for down (K=640, N=2560)"


def main() -> int:
    text = PATH.read_text()
    if NEW.split("#")[0].strip() in text:
        print("SKIP  already applied")
        return 0
    if text.count(OLD) != 1:
        print(f"FAIL  anchor found {text.count(OLD)} times")
        return 1
    backup = Path(f"/home/xujie/tensorfold-patches/pre-ntile-{time.strftime('%Y%m%d-%H%M%S')}")
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PATH, backup / "cuda__exl3__experts.py")
    print(f"backup: {backup}")
    PATH.write_text(text.replace(OLD, NEW, 1))
    import py_compile
    py_compile.compile(str(PATH), doraise=True)
    print("  ok    GLM_DOWN (8,4,1,1) -> (16,4,1,1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())