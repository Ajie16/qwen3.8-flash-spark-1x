#!/usr/bin/env python3
"""Prefill throughput against context depth, over HTTP, from the server's own counters.

Small-prompt prefill numbers (100-150 tok/s at 60-120 tokens) are dominated by fixed per-request
overhead and say nothing about the kernel. What matters is the marginal rate at realistic depths,
which is `(new_tokens) / (prefill_time)` on a cold prompt — no prefix cache to hide behind.

Each measurement uses a fresh random salt so nothing is cached, and reports the slope between
successive depths as well as the per-point rate.

Usage: prefill_depth.py [depths] [reps]
"""
import json
import sys
import time
import urllib.request
import uuid

DEPTHS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "4000,16000,32000,64000").split(",")]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
URL = "http://10.100.64.1:8899/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Local"

# ~34 tokens per repetition of this sentence.
FILLER = ("本节为工程背景材料，说明分布式存储系统的设计约束，涵盖一致性哈希、虚拟节点、"
          "数据迁移与再平衡、副本一致性、故障检测与恢复流程。以下内容不含待回答的问题。")


def run(target):
    reps = max(1, target // 34)
    body = {"model": MODEL, "max_tokens": 1, "stream": False,
            "messages": [{"role": "user",
                          "content": f"[{uuid.uuid4().hex[:8]}] " + FILLER * reps + "\n\n只回复一个字：好"}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    wall = time.perf_counter() - t0
    u = d.get("usage") or {}
    return u.get("prompt_tokens", 0), u.get("prompt_tokens_per_sec"), wall


print("  {:>9}{:>12}{:>12}{:>10}".format("目标", "实际prompt", "prefill t/s", "墙钟s"))
print("  " + "-" * 45)
pts = []
for target in DEPTHS:
    for rep in range(REPS):
        try:
            pt, ptps, wall = run(target)
        except Exception as e:  # noqa: BLE001
            print("  {:>9}  失败 {}".format(target, type(e).__name__))
            continue
        pts.append((pt, ptps, wall, target, rep))
        print("  {:>9,}{:>12,}{:>12}{:>10.2f}".format(target, pt, ptps or 0, wall))
        time.sleep(1)

print()
# Marginal rate between the smallest and largest usable points.
clean = [p for p in pts if p[0] and p[1]]
if len(clean) >= 2:
    lo = min(clean, key=lambda p: p[0] if p[3] <= 8000 else 10**9)
    hi = max(clean, key=lambda p: p[0])
    lo = min(clean, key=lambda p: p[0])
    # use the smallest prompt as the fixed-overhead reference
    t_lo = lo[0] / lo[1]
    t_hi = hi[0] / hi[1]
    slope = (t_hi - t_lo) / (hi[0] - lo[0]) * 1000
    print("  边际: {} -> {} tok 用时 {:.2f}s -> {:.2f}s".format(lo[0], hi[0], t_lo, t_hi))
    print("        {:.3f} ms/token  (渐近 {:.0f} tok/s)".format(slope, 1000 / slope if slope else 0))
    print("        截距 {:.2f} s（固定开销）".format(t_lo - slope / 1000 * lo[0]))
