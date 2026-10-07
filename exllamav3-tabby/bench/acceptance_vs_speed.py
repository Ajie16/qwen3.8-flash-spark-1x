#!/usr/bin/env python3
"""Verify the mechanism end to end: does draft acceptance predict decode rate?

The round instrumentation established the accounting - decode rate is (accepted + 1) per round over
the round's wall time, and the verify forward costs ~40 ms whether the draft window is 5 or 1. So if
the mechanism is right, acceptance measured on a prompt must predict that prompt's tok/s, and the
spread across content types must cover the range seen in the service log (14.8-60.3 tok/s on lone
requests).

Runs a spread of content types, one at a time, service idle, and reports for each: acceptance from
the server's own counters, decode tok/s, and the round-log medians over that request's window
(rounds are partitioned by the [iter] n= counter, so each request's slice is identifiable).

If a low-acceptance prompt really does land near 20 tok/s and a high-acceptance one near 60, the
question "why did decode drop" is answered: content drove the draft acceptance down, and the verify
forward's fixed cost turned that into lost throughput.

Usage: acceptance_vs_speed.py [max_tokens]
"""
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

MT = int(sys.argv[1]) if len(sys.argv) > 1 else 700
LOG = "/home/xujie/serve-exl3-native.log"
URL = "http://10.100.64.1:8899/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Local"

PROMPTS = {
    "code": "Write a Python function that parses an nginx access log line into a dict with fields ip, "
            "timestamp, method, path, status, bytes. Add type hints and a docstring, then explain "
            "each regex group.",
    "prose": "Write a vivid 400-word short story about a lighthouse keeper on a remote Alaskan "
             "island who finds something unexpected washed ashore after a storm. Vary sentence "
             "structure and use specific sensory detail.",
    "math": "Solve step by step: find all real x such that x^3 - 6x^2 + 11x - 6 = 0, then prove the "
            "roots are the only ones, and generalise to x^3 - 6x^2 + 11x - c.",
    "json": "Produce a JSON array of 25 records, each with fields id, name, email, city, signup_date, "
            "and plan. Make the values realistic and varied.",
    "zh_tech": "请详细说明分布式数据库两阶段提交协议的状态机、协调者与参与者各自故障的恢复流程、"
               "网络分区下的阻塞问题，以及如何避免悬挂事务。",
    "zh_prose": "写一篇约四百字的中文短篇故事，讲一位守灯塔的老人在暴风雨后发现岸边漂来的东西。"
                "要求句式多变，包含具体的感官细节。",
}


def round_marker():
    try:
        out = subprocess.run(["grep", "-c", r"\[iter\]", LOG], capture_output=True, text=True).stdout
        return int(out.strip() or 0)
    except Exception:
        return 0


def request(name):
    body = {"model": MODEL, "max_tokens": MT, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"[{uuid.uuid4().hex[:8]}] {PROMPTS[name]}"}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
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


print(f"##### 接受率 vs decode 速率 · max_tokens={MT} #####")
rows = []
for name in PROMPTS:
    n0 = round_marker()
    usage, err = request(name)
    time.sleep(1.5)
    if err or not usage:
        print(f"  {name:<10} 失败: {err}")
        continue
    det = usage.get("completion_tokens_details") or {}
    acc = det.get("accepted_prediction_tokens") or 0
    rej = det.get("rejected_prediction_tokens") or 0
    rate = 100 * acc / (acc + rej) if (acc + rej) else 0
    tps = usage.get("completion_tokens_per_sec", 0)
    out = usage["completion_tokens"]
    # slice this request's rounds
    lines = subprocess.run(["grep", r"\[iter\]", LOG], capture_output=True, text=True).stdout.splitlines()
    mine = []
    for ln in lines[n0:]:
        m = re.search(r"win=(-?\d+).*?verify_ms=([\d.]+).*?total_ms=([\d.]+) acc=\+(-?\d+)", ln)
        if m:
            mine.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), int(m.group(4))))
    rw = [r[0] for r in mine if r[0] > 0]
    rt = [r[2] for r in mine if r[0] > 0]
    med_w = sorted(rw)[len(rw) // 2] if rw else 0
    med_t = sorted(rt)[len(rt) // 2] if rt else 0
    rows.append((name, rate, tps, out, med_w, med_t, len(mine)))
    print(f"  {name:<10} 接受 {rate:>5.1f}%  {tps:>6.1f} T/s  输出 {out:>5}  "
          f"轮 {len(mine):>4}  中位 win={med_w:>4} 轮时={med_t:>6.1f}ms")

print()
if len(rows) >= 3:
    lo = min(rows, key=lambda r: r[1])
    hi = max(rows, key=lambda r: r[1])
    print(f"  接受率最低 {lo[0]} ({lo[1]:.1f}%) -> {lo[2]:.1f} T/s")
    print(f"  接受率最高 {hi[0]} ({hi[1]:.1f}%) -> {hi[2]:.1f} T/s")
    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    print(f"  接受率与 T/s 的相关系数 r = {num / den:+.2f}" if den else "  方差为零")
    print()
    print("  预测公式 (acc/round + 1) / 轮时:")
    for name, rate, tps, out, w, t, n in rows:
        if t:
            pred = 1000 / t * (1 + (rate / 100) * w)
            print(f"    {name:<10} 实测 {tps:>6.1f}   预测 {pred:>6.1f}   "
                  f"{'✓' if abs(pred - tps) / max(tps, 1) < 0.35 else '✗'}")
