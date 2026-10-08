#!/usr/bin/env python3
"""Prefill-only benchmark reading the server's own counters.

Wall-clock prefill numbers are diluted by HTTP, chat-template rendering and the sampled output
token; TabbyAPI reports `prompt_tokens_per_sec` directly and that is the number to compare knob
settings against. Streaming with `stream_options.include_usage` is what makes the usage block arrive
(a non-streaming request leaves it null).

Cold by construction: every request carries a fresh random salt as the first text, so no prefix can
be reused and the measured work is the whole prompt.

Usage: prefill_bench.py [target_tokens] [reps]
"""
import json
import statistics
import sys
import time
import urllib.request
import uuid

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
URL = "http://10.100.65.1:8899/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Local"

# ~41 tokens per repetition, measured against the server's reported prompt_tokens.
FILLER = ("本节为工程背景材料，说明分布式存储系统的设计约束，涵盖一致性哈希、虚拟节点、"
          "数据迁移与再平衡、副本一致性、故障检测与恢复流程。以下内容不含待回答的问题。")


def run():
    reps = max(1, TARGET // 41)
    body = {"model": MODEL, "max_tokens": 1, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user",
                          "content": f"[{uuid.uuid4().hex[:8]}] " + FILLER * reps + "\n\n只回复一个字"}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    usage = None
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
    return usage


pts, tps = [], []
for i in range(REPS):
    try:
        u = run()
    except Exception as e:  # noqa: BLE001
        print("  rep {} 失败: {}".format(i + 1, type(e).__name__))
        continue
    if not u:
        print("  rep {} 无 usage".format(i + 1))
        continue
    pts.append(u.get("prompt_tokens", 0))
    tps.append(u.get("prompt_tokens_per_sec") or 0)
    print("  rep {}: prompt {:,}  prefill {:.0f} t/s".format(i + 1, pts[-1], tps[-1]))
    time.sleep(1)

if tps:
    print("  ---")
    print("  prompt  {:,}   prefill 中位 {:.0f} t/s   (范围 {:.0f}-{:.0f})".format(
        int(statistics.mean(pts)) if pts else 0, statistics.median(tps), min(tps), max(tps)))
else:
    print("  无有效样本")
