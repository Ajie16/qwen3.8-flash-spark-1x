#!/usr/bin/env python3
"""Dump the per-expert row counts of every prefill pass, to size the dequant-amortization lever.

Why: ext.group writes the grouping as `ids` (unique expert ids) + `members` (maxu x R, -1 for
non-members). The per-expert row count is therefore (members >= 0).sum(dim=1), and nothing in
TensorFold keeps it. That number decides how much of the prompt could be amortized if hot experts
were decoded once and handed to cuBLAS instead of re-running the trellis decode per 16-row block -
the tier exllamav3 uses above 256 rows (moe_batch_recon.py:128).

Collection stays on the GPU; only when 48 layers (one pass) have arrived does it move to the CPU
and write. One D2H copy per prefill, not per layer.

Set TENSORFOLD_DUMP_EXPERT_ROWS=<path> to arm it.

The first attempt of this patch inserted the helper immediately after the call site, so the
column-0 def ended routed() early and the rest of routed() fell into the helper's body
(NameError: ext). Hence the helper goes before routed() and the call site stays a bare line.
"""
import shutil
import sys
import time
from pathlib import Path

PATH = Path("/home/xujie/tensorfold-venv/lib/python3.12/site-packages/tensorfold/cuda/exl3/experts.py")
SOURCE = Path("/home/xujie/patch_expert_rows.py")

HELPER = '''
# TEMPORARY MEASUREMENT HOOK (2026-10-03) - remove with this file's revert
_DUMP_PATH = os.environ.get("TENSORFOLD_DUMP_EXPERT_ROWS")
_dump_pending: list = []


def _dump_expert_rows(members: torch.Tensor, R: int) -> None:
    """Append one layer's per-expert row counts; write the file once a full prefill is collected."""

    if not _DUMP_PATH:
        return
    counts = (members >= 0).sum(dim=1).to(torch.int32).cpu()
    _dump_pending.append((R, counts))
    if len(_dump_pending) >= 48:                      # one pass over all 48 layers
        torch.save([(r, c.tolist()) for r, c in _dump_pending], _DUMP_PATH)
        _dump_pending.clear()


def routed('''

PROBE = "def _dump_expert_rows("

# the only change inside routed(): one call, right after ext.group
CALL_OLD = """    if group:
        ext.group(pick, ids, s.count, members, R, slots, E)
"""
CALL_NEW = """    if group:
        ext.group(pick, ids, s.count, members, R, slots, E)
        _dump_expert_rows(members, R)
"""


def main() -> int:
    text = PATH.read_text()
    if PROBE in text:
        print("SKIP  already applied")
        return 0
    if text.count("def routed(") != 1:
        print(f"FAIL  found {text.count('def routed(')} def routed")
        return 1
    if text.count(CALL_OLD) != 1:
        print(f"FAIL  call anchor found {text.count(CALL_OLD)} times")
        return 1

    backup = Path(f"/home/xujie/tensorfold-patches/pre-expertrows2-{time.strftime('%Y%m%d-%H%M%S')}")
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PATH, backup / "cuda__exl3__experts.py.broken")
    print(f"backup: {backup}")

    text = text.replace("def routed(", HELPER, 1)
    text = text.replace(CALL_OLD, CALL_NEW, 1)
    if "\nimport os\n" not in text:
        lines = text.split("\n")
        last = max(i for i, ln in enumerate(lines) if ln.startswith(("import ", "from ")))
        lines.insert(last + 1, "\nimport os")
        text = "\n".join(lines)

    PATH.write_text(text)
    import py_compile
    py_compile.compile(str(PATH), doraise=True)
    print("  ok    helper defined before routed(); one call after ext.group")
    return 0


if __name__ == "__main__":
    sys.exit(main())