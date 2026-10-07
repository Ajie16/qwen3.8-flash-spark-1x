#!/usr/bin/env python3
"""Raise the EXL3 routed-expert window from 1024 to 2048 rows.

Why this and not the prompt chunk: total expert sweeps = passes x windows-per-pass
= (tokens / chunk) x (chunk / MOE_WINDOW) = tokens / MOE_WINDOW. The chunk cancels out,
so MOE_WINDOW is the only lever on how often the expert weights are re-decoded. At 1024
with a 2048-row pass the whole expert set is swept twice per pass.

2048 is the ceiling, not an arbitrary pick: the grouping keeps R*slots*4 bytes of shared
memory (experts.cu raises cudaFuncSetAttribute against sharedMemPerBlockOptin). Measured on
this GB10: optin = 101,376 B, so R=2048 needs 90,112 B and fits, R=4096 needs 180,224 B
and does not.
"""
import shutil
import sys
import time
from pathlib import Path

PATH = Path("/home/xujie/tensorfold-venv/lib/python3.12/site-packages/tensorfold/"
            "families/qwen4_exp/cuda/exl3_pack.py")
OLD = "MOE_WINDOW = 1024        # most rows a routed-expert call takes (its grouping keeps every pick in 48 KB of shared memory)"
NEW = ("MOE_WINDOW = 2048        # most rows a routed-expert call takes; its grouping keeps every pick in\n"
       "                         # R*slots*4 bytes of shared memory, and cudaFuncSetAttribute raises the\n"
       "                         # limit to sharedMemPerBlockOptin (101,376 B measured on a GB10: 2048 fits,\n"
       "                         # 4096 would need 180,224 B and cannot). Halving the window halves how\n"
       "                         # often a prompt pass re-decodes the expert weights.")

PROBE = "MOE_WINDOW = 2048"


def main() -> int:
    if not PATH.exists():
        print(f"FAIL  {PATH} not found")
        return 1
    text = PATH.read_text()
    if PROBE in text:
        print("SKIP  already applied")
        return 0
    if text.count(OLD) != 1:
        print(f"FAIL  anchor found {text.count(OLD)} times")
        return 1

    backup = Path(f"/home/xujie/tensorfold-patches/pre-moewindow-{time.strftime('%Y%m%d-%H%M%S')}")
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PATH, backup / "families__qwen4_exp__cuda__exl3_pack.py")
    print(f"backup: {backup}")

    PATH.write_text(text.replace(OLD, NEW, 1))
    import py_compile
    py_compile.compile(str(PATH), doraise=True)
    print("  ok    MOE_WINDOW 1024 -> 2048")
    print("  compiles  families/qwen4_exp/cuda/exl3_pack.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())