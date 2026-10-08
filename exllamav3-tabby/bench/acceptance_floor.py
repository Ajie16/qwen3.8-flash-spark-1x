#!/usr/bin/env python3
"""Drive acceptance toward zero and see whether decode lands on the log's slow end.

The mechanism is decode tok/s = (accepted + 1) / round_time, with round_time set by the verify
forward (~35-46 ms measured) and accepted = acceptance_rate x draft_window. Six content types
correlated +0.99 and the formula matched all six, but they only spanned 46-65 tok/s. The service log
has lone requests at 14.8 tok/s, which the formula reaches only if about one token comes out per
~68 ms round - i.e. acceptance near zero with a full-length round.

Content the drafter cannot anticipate should do that: unpredictable values such as random
identifiers have no exploitable structure, so the draft proposes and the target rejects.

Runs predictable and unpredictable tasks and reports acceptance, tok/s and the round medians.
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid

MT = int(sys.argv[1]) if len(sys.argv) > 1 else 500
LOG = "/home/xujie/serve-exl3-native.log"
URL = "http://10.100.65.1:8899/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Local"

TASKS = {
    # predictable: structure the drafter can copy
    "json_plan": "Produce a JSON array of 30 user records with fields id, name, email, city, "
                 "signup_date and plan. Make the values realistic and varied.",
    # unpredictable: values with no exploitable structure
    "uuid_list": "Generate 40 random UUIDv4 identifiers, one per line, in lowercase hex with "
                 "hyphens. Do not number them or add commentary.",
    "rand_hex": "Generate 60 random 32-character lowercase hexadecimal strings, one per line. "
                "No commentary, no numbering.",
    "rand_words": "Generate 200 random English words, comma separated, all different, chosen "
                  "arbitrarily with no theme. No commentary.",
}


def marker():
    try:
        return int(subprocess.run(["grep", "-c", r"\[iter\]", LOG],
                                  capture_output=True, text=True).stdout.strip() or 0)
    except Exception:
        return 0


def run(name):
    body = {"model": MODEL, "max_tokens": MT, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"[{uuid.uuid4().hex[:8]}] {TASKS[name]}"}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                s = raw.decode("utf-8", "replace").strip()
                if not s.startswith("data:"):
                    continue
                p = s[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    c = json.loads(p)
                except json.JSONDecodeError:
                    continue
                if c.get("usage"):
                    usage = c["usage"]
    except Exception as e:  # noqa: BLE001
        return None, type(e).__name__
    return usage, None


print(f"##### 接受率下限探针 · max_tokens={MT} #####")
print(f"  {'任务':<12}{'接受率':>8}{'T/s':>8}{'输出':>7}{'轮数':>7}{'中位win':>9}{'轮时ms':>9}")
print("  " + "-" * 60)
for name in TASKS:
    n0 = marker()
    usage, err = run(name)
    time.sleep(1.5)
    if err or not usage:
        print(f"  {name:<12} 失败: {err}")
        continue
    det = usage.get("completion_tokens_details") or {}
    a = det.get("accepted_prediction_tokens") or 0
    r = det.get("rejected_prediction_tokens") or 0
    pct = 100 * a / (a + r) if (a + r) else 0
    tps = usage.get("completion_tokens_per_sec", 0)
    lines = subprocess.run(["grep", r"\[iter\]", LOG], capture_output=True, text=True).stdout.splitlines()
    mine = []
    for ln in lines[n0:]:
        m = re.search(r"win=(-?\d+).*?total_ms=([\d.]+)", ln)
        if m and int(m.group(1)) > 0:
            mine.append((int(m.group(1)), float(m.group(2))))
    w = sorted(x[0] for x in mine)[len(mine) // 2] if mine else 0
    t = sorted(x[1] for x in mine)[len(mine) // 2] if mine else 0
    print(f"  {name:<12}{pct:>7.1f}%{tps:>8.1f}{usage['completion_tokens']:>7}"
          f"{len(mine):>7}{w:>9}{t:>9.1f}")
print()
print("  对照：服务日志里独占长请求的最低是 14.8 T/s，本机正常档 46-65 T/s")
