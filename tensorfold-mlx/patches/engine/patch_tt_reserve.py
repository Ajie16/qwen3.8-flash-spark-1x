#!/usr/bin/env python3
"""Refine the thinking-budget clamp: reserve a quarter of the ceiling for the answer.

The first clamp (budget <= max_tokens - len(close)) stops the reply from ending inside its own
think block, but it spends everything else on thinking: a clamped request came back with the two
newlines of the close and nothing else. Budget must also leave room to answer.
"""
import shutil
import sys
import time
from pathlib import Path

PATH = Path("/home/xujie/tensorfold-venv/lib/python3.12/site-packages/tensorfold/cuda/server.py")
STAMP = time.strftime("%Y%m%d-%H%M%S")

OLD_CLAMP = """        budget = min(prepared.think_budget, max(0, prepared.max_tokens - len(close)))
        if budget <= 0:
            return None
        return ThinkBudget(budget, close, think_end)"""

NEW_CLAMP = """        # Reserve a quarter of the ceiling for the answer, not merely the length of the close:
        # a clamp that only fits the close hands the client two newlines and nothing else.
        answer = max(_MIN_ANSWER, prepared.max_tokens // 4)
        budget = min(prepared.think_budget, max(0, prepared.max_tokens - answer))
        if budget < len(close):
            return None                            # too little to think and still answer: do not cut
        return ThinkBudget(budget, close, think_end)"""

OLD_CONST = "from __future__ import annotations"
NEW_CONST = "from __future__ import annotations\n\n_MIN_ANSWER = 256          # reply tokens a thinking budget never takes"

PROBE = "answer = max(_MIN_ANSWER, prepared.max_tokens // 4)"


def main() -> int:
    backup = Path(f"/home/xujie/tensorfold-patches/pre-061-reserve-{STAMP}")
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PATH, backup / "cuda__server.py")
    print(f"backup: {backup}")

    text = PATH.read_text()
    if PROBE in text:
        print("  SKIP  already applied")
        return 0
    for label, old, new in (("the clamp", OLD_CLAMP, NEW_CLAMP), ("the reserve constant", OLD_CONST, NEW_CONST)):
        if text.count(old) != 1:
            print(f"  FAIL  {label}: anchor found {text.count(old)} times")
            return 1
        text = text.replace(old, new, 1)
        print(f"  ok    {label}")
    PATH.write_text(text)

    import py_compile
    py_compile.compile(str(PATH), doraise=True)
    print("  compiles  cuda/server.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())