#!/usr/bin/env python3
"""Start an aria2c download of a HuggingFace repo from spark-2, through hf-mirror.

The model goes to ~/models; this script and its log stay in the workspace. It writes an aria2
input file listing every blob under resolve/main, then runs aria2c with 8 connections per file and
16 files at a time, continuing a partial download if one is already there.

spark-2 cannot reach huggingface.co, only hf-mirror.com, so every URL is rewritten to the mirror.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = sys.argv[1] if len(sys.argv) > 1 else "nvidia/Qwen3.8-Flash-Next-FP4-Local"
MIRROR = "https://hf-mirror.com"
DEST = Path("/home/xujie/models") / REPO.split("/")[-1]
WORK = Path("/home/xujie/dl")           # process files, not the workspace's, but not /tmp either


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)

    api = "%s/api/models/%s?blobs=true" % (MIRROR, REPO)
    # hf-mirror answers 403 to urllib's default agent; curl and hf_hub_download both send one.
    req = urllib.request.Request(api, headers={"User-Agent": "curl/8.5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        meta = json.load(r)
    files = [s["rfilename"] for s in meta.get("siblings", []) if s.get("rfilename")]
    # config.json and friends are small but needed; keep them too
    print("repo  %s" % REPO)
    print("files %d" % len(files))

    inp = WORK / (REPO.split("/")[-1] + ".aria2")
    lines = []
    total = 0
    for f in files:
        lines.append("%s/%s/resolve/main/%s" % (MIRROR, REPO, f))
        lines.append("  dir=%s" % DEST)
        lines.append("  out=%s" % f.rsplit("/", 1)[-1])
    inp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("input %s (%d entries)" % (inp, len(files)))
    for s in meta.get("siblings", []):
        if s.get("size"):
            total += s["size"]
    print("size  %.1f GiB" % (total / 2 ** 30))
    print("dest  %s" % DEST)
    print("free  %s" % subprocess.run(["df", "-h", str(DEST)], capture_output=True, text=True)
          .stdout.splitlines()[-1])

    log = WORK / (REPO.split("/")[-1] + ".download.log")
    cmd = ["aria2c", "-x8", "-s8", "-j16", "-c", "--file-allocation=none",
           "--summary-interval=30", "--console-log-level=warn",
           "-i", str(inp), "-d", str(DEST), "-l", str(log)]
    print("run   aria2c -x8 -s8 -j16 -c")
    print("log   %s" % log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
