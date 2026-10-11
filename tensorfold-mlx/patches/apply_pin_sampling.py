#!/usr/bin/env python3
"""Pin the served checkpoint's own sampling settings, ignoring what a client asks for.

Why: the pack's generation_config.json pins temperature 1.0, top_k 20 and top_p 0.95, which is
what Qwen recommends for this model and what TensorFold already uses as its default. But
``App.sampling_for`` reads the client's values first --

    temp = float(fields.get("temperature", self.sampling["temperature"]))

-- so any client that sends its own numbers silently overrides the model's recommendation, and the
same conversation answers differently depending on which client sent it. A benchmark asking for
greedy and an agent asking for 0.7 are then not comparing the same model. There is no supported
switch for this in TensorFold 0.6.6, hence the patch.

``seed`` is deliberately left to the client: it controls reproducibility, not the output
distribution, and TensorFold already derives a seed from the prompt when none is sent.

Set TENSORFOLD_PIN_SAMPLING=0 in the server's environment to restore the per-request behaviour.
That escape hatch is server-side on purpose: benchmarks that need greedy can have it, clients
cannot reach it.

Idempotent, and safe to re-run after a TensorFold upgrade overwrites server.py. It does not touch
the running process -- the module is already imported -- so it takes effect on the next restart.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TARGET = Path("/home/xujie/tensorfold-venv/lib/python3.12/site-packages/tensorfold/cuda/server.py")

IMPORT_ANCHOR = "import json\nimport threading\n"
IMPORT_NEW = "import json\nimport os\nimport threading\n"

FIELDS_ANCHOR = '_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "seed")\n'
FIELDS_NEW = FIELDS_ANCHOR + """
# Serve the checkpoint's own recommended sampling instead of whatever a client sends. The pack's
# generation_config.json pins temperature 1.0 / top_k 20 / top_p 0.95; a client asking for greedy or
# 0.7 otherwise changes the model silently, so one conversation answers differently per client.
# TENSORFOLD_PIN_SAMPLING=0 restores the per-request behaviour (server-side only, so benchmarks that
# want greedy can have it and clients cannot).
PIN_SAMPLING = os.environ.get("TENSORFOLD_PIN_SAMPLING", "1") != "0"
"""

OLD_BODY = '''        fields = parse_numbers({k: body[k] for k in _SAMPLING_FIELDS if body.get(k) is not None})
        temp = float(fields.get("temperature", self.sampling["temperature"]))
        if temp <= 0:
            return None
        seed = fields.get("seed")
        top_k = fields.get("top_k", self.sampling["top_k"])
        top_p = fields.get("top_p", self.sampling["top_p"])
        min_p = fields.get("min_p", self.sampling.get("min_p", 0.0))
'''

NEW_BODY = '''        fields = parse_numbers({k: body[k] for k in _SAMPLING_FIELDS if body.get(k) is not None})
        if PIN_SAMPLING:
            # the checkpoint's numbers win; only `seed` below stays the client's
            temp = float(self.sampling["temperature"])
            top_k, top_p = self.sampling["top_k"], self.sampling["top_p"]
            min_p = self.sampling.get("min_p", 0.0)
        else:
            temp = float(fields.get("temperature", self.sampling["temperature"]))
            top_k = fields.get("top_k", self.sampling["top_k"])
            top_p = fields.get("top_p", self.sampling["top_p"])
            min_p = fields.get("min_p", self.sampling.get("min_p", 0.0))
        if temp <= 0:
            return None
        seed = fields.get("seed")
'''

OLD_DOC = '        """Keyed sampling (the seed, else one drawn from the prompt), or None for greedy; RequestError if malformed."""'
NEW_DOC = ('        """Keyed sampling (the seed, else one drawn from the prompt), or None for greedy; RequestError if malformed.\n'
           '        The distribution is the checkpoint\'s while PIN_SAMPLING holds; the seed is still the request\'s."""')


def main() -> int:
    if not TARGET.is_file():
        print("FAIL  %s not found" % TARGET)
        return 1
    src = TARGET.read_text(encoding="utf-8")

    if "PIN_SAMPLING" in src:
        print("already applied (PIN_SAMPLING present); nothing to do")
        return 0

    for label, old, new in (
        ("import os", IMPORT_ANCHOR, IMPORT_NEW),
        ("PIN_SAMPLING constant", FIELDS_ANCHOR, FIELDS_NEW),
        ("docstring", OLD_DOC, NEW_DOC),
        ("sampling_for body", OLD_BODY, NEW_BODY),
    ):
        if old not in src:
            print("FAIL  anchor not found: %s" % label)
            print("      the file differs from the expected 0.6.6 text; inspect before patching")
            return 1
        if src.count(old) != 1:
            print("FAIL  anchor is not unique: %s (%d matches)" % (label, src.count(old)))
            return 1
        src = src.replace(old, new)
        print("patched: %s" % label)

    TARGET.write_text(src, encoding="utf-8")

    # verify by compiling; a syntax error here would take the service down at the next restart
    import py_compile
    try:
        py_compile.compile(str(TARGET), doraise=True, cfile="/tmp/_srv_check.pyc")
    except py_compile.PyCompileError as exc:
        print("FAIL  patched file does not compile: %s" % exc)
        return 1
    print("compiles cleanly")

    body = TARGET.read_text(encoding="utf-8")
    for probe in ("import os", "PIN_SAMPLING = os.environ.get", "if PIN_SAMPLING:"):
        print("  %-40s %s" % (probe, "present" if probe in body else "MISSING"))
    print("\nTakes effect on the next restart; the running process keeps the old behaviour.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
