#!/usr/bin/env python3
"""Restrict the expert-row dump to the live prefix of `members`.

Self-check failure: the busiest row showed 426 live entries while slots = top_k + shared = 11
(cuda/exl3/experts.py:26 via exl3_mm.py:25-26). members_buf is a reused scratch that ext.group
only writes over its active ids, so entries past the live count are stale from an earlier pass.

Fix: pass s.count (the live length of ids) and read only members[:n].
"""
import shutil
import time
from pathlib import Path

PATH = Path("/home/xujie/tensorfold-venv/lib/python3.12/site-packages/tensorfold/cuda/exl3/experts.py")

CALL_OLD = "        _dump_expert_rows(members, R)"
CALL_NEW = "        _dump_expert_rows(members, R, s.count)"

SIG_OLD = "def _dump_expert_rows(members: torch.Tensor, R: int) -> None:"
SIG_NEW = "def _dump_expert_rows(members: torch.Tensor, R: int, count) -> None:"

BODY_OLD = """    live = (members >= 0)
    counts = live.sum(dim=1).to(torch.int32).cpu()            # per expert"""
BODY_NEW = """    n = int(count.item())                    # ids' live length: only the first n rows are written
    live = members[:n] >= 0
    counts = live.sum(dim=1).to(torch.int32).cpu()            # per expert"""


def main() -> int:
    text = PATH.read_text()
    if "n = int(count.item())" in text:
        print("SKIP  already applied")
        return 0
    for name, old in (("call", CALL_OLD), ("sig", SIG_OLD), ("body", BODY_OLD)):
        if text.count(old) != 1:
            print(f"FAIL  {name} anchor found {text.count(old)} times")
            return 1
    backup = Path(f"/home/xujie/tensorfold-patches/pre-expertrows3-{time.strftime('%Y%m%d-%H%M%S')}")
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PATH, backup / "cuda__exl3__experts.py")
    print(f"backup: {backup}")
    text = text.replace(CALL_OLD, CALL_NEW, 1)
    text = text.replace(SIG_OLD, SIG_NEW, 1)
    text = text.replace(BODY_OLD, BODY_NEW, 1)
    PATH.write_text(text)
    import py_compile
    py_compile.compile(str(PATH), doraise=True)
    print("  ok    members sliced to the live expert count")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())