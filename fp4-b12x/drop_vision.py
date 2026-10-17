#!/usr/bin/env python3
"""Drop --vision from the EXL3-on-TensorFold launcher and record why.

The pack's vision tower is EXL3 trellis (vision_bits 5, mul1 coding) and TensorFold's CUDA
geometry check refuses it: "invalid or unsupported vision tensor range: blocks.0.attn.k_proj.mul1".
TabbyAPI did load that tower, so this is a real loss against that stack -- images and video stop
working, text does not. Recording it in the launcher so the next reader knows it was tried and
failed, not overlooked.
"""
from pathlib import Path

P = Path("/home/xujie/tf-serve-exl3-on-tensorfold.sh")
src = P.read_text(encoding="utf-8")

FLAG = "  --vision --vision-max-images 16 \\\n"
NOTE_LINES = [
    "# --vision is off: this pack's vision tower is EXL3 trellis (vision_bits 5, mul1 coding) and the",
    "# CUDA geometry check rejects it -- \"invalid or unsupported vision tensor range:",
    "# blocks.0.attn.k_proj.mul1\". TabbyAPI loaded that tower through EXL3's own kernels, so this is a",
    "# real loss against that stack: images and video do not work here, text does. A 0.6.1 tree in",
    "# ~/tensorfold-venv-exl3 serves this pack with --vision, so the check is newer than the pack's",
    "# format support, not the pack's fault.",
    "#",
]

if "--vision" not in src:
    print("already removed")
elif FLAG not in src:
    print("FAIL  flag line not found verbatim")
    raise SystemExit(1)
else:
    src = src.replace(FLAG, "", 1)
    anchor = "# --ple-on-ssd is MLX-only"
    if anchor not in src:
        print("FAIL  note anchor missing")
        raise SystemExit(1)
    src = src.replace(anchor, "\n".join(NOTE_LINES) + anchor, 1)
    P.write_text(src, encoding="utf-8")
    print("removed --vision, recorded why")

print()
for line in P.read_text(encoding="utf-8").split("\n"):
    s = line.strip()
    if s.startswith("exec ") or "--" in s and "tensorfold serve" not in s and s.startswith("--"):
        print("  %s" % s[:112])
    elif s.startswith("--"):
        print("  %s" % s[:112])
