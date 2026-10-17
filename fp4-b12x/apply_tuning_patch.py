#!/usr/bin/env python3
"""ARCHIVED DEAD END -- do not apply this and expect working 8192-batch serving.

**This patch does not make --max-num-batched-tokens 8192 work.** It gets past the first assertion and
then fails at flashinfer/experimental/b12x/preparation/session.py:1027 with "tuning consolidation does
not match pending races". The B12X preparation is a chain of cross-rank invariants; patching one only
moves the failure. It is kept as a record of what was tried and why the size is not raisable.

Original intent below.

Patch vLLM's B12X preparation consistency check so a tuning-key mismatch is survivable.

Why: `b12x_startup.py` gathers a tuning result from every rank, then keeps only the keys that EVERY
participating rank marked ready:

    choices = [key for key, ranks in participants_by_key.items()
               if ready_by_key.get(key) == set(ranks)]

and immediately after compares that authorised set against every key any rank offered:

    if authorized != pending:
        raise RuntimeError("preparation ranks reached incompatible tuning boundaries")

On this two-Spark pair the comparison fails for --max-num-batched-tokens 8192 and 16384, so the
engine cannot start at all. Only 4096 works. Clearing the compile cache does not help, which rules
out stale artefacts: the two ranks genuinely disagree about one kernel's readiness at those sizes.

Nothing here is a hardcoded size list. The sizes are decided by which kernel candidates each rank can
launch; this check is a consistency assertion over that, and it is the assertion that is fatal rather
than the tuning itself. Downgrading it to a warning keeps the winners that were agreed and lets the
affected kernel run its untuned path: slower for that one kernel, but correct, and the missing key is
logged so the cost is visible.

The patched file is bind-mounted over the image's copy, so the image stays untouched and the change
is undone by dropping the mount.
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

# The vLLM source is not vendored here (it is Apache-2.0 code owned upstream); see
# b12x_startup.py.patch for the exact change. Extract the file from the image first:
#   docker run --rm -v "$PWD/b12x-patch:/out" --entrypoint sh eugr/spark-vllm-b12x:latest \
#     -c 'cp /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/b12x_startup.py /out/'
SRC = Path(__file__).parent / "b12x-patch" / "b12x_startup.py"

OLD = '''            if authorized != pending:
                raise RuntimeError(
                    "preparation ranks reached incompatible tuning boundaries"
                )
'''

NEW = '''            if authorized != pending:
                # PATCHED (see tensorfold-mlx/scripts/b12x-patch/apply_tuning_patch.py): a key that
                # some rank offered but not every participant marked ready is dropped from `choices`
                # above. Raising made any batch size whose kernel grid leaves one rank unready
                # unusable -- that is what blocks --max-num-batched-tokens 8192/16384 on this TP=2
                # pair. Log the mismatch and keep the winners we do have; the affected kernel falls
                # back to its untuned path.
                logger.warning(
                    "b12x preparation tuning key mismatch: unready=%s unexpected=%s",
                    sorted(pending - authorized),
                    sorted(authorized - pending),
                )
'''


def main() -> int:
    if not SRC.is_file():
        print("FAIL  %s missing -- extract it from the image first" % SRC)
        return 1
    src = SRC.read_text(encoding="utf-8")

    if "PATCHED (see tensorfold-mlx" in src:
        print("already patched")
        return 0
    if src.count(OLD) != 1:
        print("FAIL  anchor found %d times (expected 1)" % src.count(OLD))
        return 1

    SRC.write_text(src.replace(OLD, NEW, 1), encoding="utf-8")
    try:
        py_compile.compile(str(SRC), doraise=True, cfile="/tmp/_b12x_check.pyc")
    except py_compile.PyCompileError as exc:
        print("FAIL  patched file does not compile: %s" % exc)
        return 1
    print("patched and compiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
