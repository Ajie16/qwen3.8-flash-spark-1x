#!/usr/bin/env python3
"""Time-slice a prefill pass across the prompts waiting to fill.

Measured on the live EXL3 server: four cold prompts issued together all prefilled at once
(/health shows prefilling=4) yet aggregate throughput stayed at 786 tok/s and the last one took
149.7 s - exactly one prompt's fill time each, run back to back. Overlap factor 0.99x.

Why: _pieces handed the whole 2048-row window to the first prompt in _order(), and _order()
sorts by "fewest rows left" - which is precisely the prompt that just consumed the most rows.
So the same prompt led every pass. FILL_GUARD=8 was meant to rotate a starved prompt, but
sorting `due` before `not due` made a prompt due forever once it reached the guard, so the
guard inverted instead of rotating.

The fix divides the pass among the waiting prompts. A single filling prompt still takes the
whole window, so single-stream prefill is bit-identical; only concurrent fills are affected.
"""
import shutil
import sys
import time
from pathlib import Path

PATH = Path("/home/xujie/tensorfold-venv/lib/python3.12/site-packages/tensorfold/"
            "families/qwen4_exp/cuda/multi.py")

OLD = '''        pieces, room = [], self._pass_rows() if rows is None else rows
        for s in self._order():
            e, mtp, start, _ = self.fills[s.sid]
            n = min(next((p for p in e.stops if p > start), len(s.prompt)) - start, room)
            ends = sum(1 for x, a, k in pieces if a + k == len(x.prompt))
            if n == 0 or (start + n == len(s.prompt) and ends == ENDS):
                break
            pieces.append((s, start, n))
            room -= n
        return pieces'''

NEW = '''        pieces, room = [], self._pass_rows() if rows is None else rows
        order = self._order()
        # Time-slice the pass across the prompts waiting to fill. Handing the whole window to the
        # first prompt made "fewest rows left" pick the same prompt every pass - it is the one that
        # just consumed the most rows - so N prompts arriving together filled strictly one after
        # another and the last of them waited for all the others. One prompt still takes the whole
        # window, so a single filling request is unchanged.
        share = max(PASS_MIN, room // len(order)) if order else room
        for s in order:
            e, mtp, start, _ = self.fills[s.sid]
            n = min(next((p for p in e.stops if p > start), len(s.prompt)) - start, room, share)
            ends = sum(1 for x, a, k in pieces if a + k == len(x.prompt))
            if n == 0 or (start + n == len(s.prompt) and ends == ENDS):
                break
            pieces.append((s, start, n))
            room -= n
        return pieces'''

PROBE = "share = max(PASS_MIN, room // len(order)) if order else room"


def main() -> int:
    text = PATH.read_text()
    if PROBE in text:
        print("SKIP  already applied")
        return 0
    if text.count(OLD) != 1:
        print(f"FAIL  anchor found {text.count(OLD)} times")
        return 1

    backup = Path(f"/home/xujie/tensorfold-patches/pre-fillslice-{time.strftime('%Y%m%d-%H%M%S')}")
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PATH, backup / "families__qwen4_exp__cuda__multi.py")
    print(f"backup: {backup}")

    PATH.write_text(text.replace(OLD, NEW, 1))
    import py_compile
    py_compile.compile(str(PATH), doraise=True)
    print("  ok    _pieces time-slices the pass across waiting prompts")
    print("  compiles  families/qwen4_exp/cuda/multi.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())