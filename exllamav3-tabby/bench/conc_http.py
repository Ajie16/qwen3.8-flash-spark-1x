#!/usr/bin/env python3
"""Concurrency sweep over HTTP, the way the deployment is actually configured (4 jobs).

Decode rate per stream says little on its own: what matters for an agent host is aggregate
throughput while several requests are in flight, and whether per-stream latency collapses. Each
stream gets a unique salt so nothing shares a prefix, runs in its own thread, and reports the
server's own counters.

Aggregate is total output tokens across all streams divided by the window from the first token to
the last, which is what a user of the box actually experiences.

Usage: conc_http.py [streams] [max_tokens]
"""
import json
import sys
import threading
import time
import urllib.request
import uuid

STREAMS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2,4").split(",")]
MT = int(sys.argv[2]) if len(sys.argv) > 2 else 400
URL = "http://10.100.64.1:8899/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Local"

ASK = ("Write a Python module implementing a consistent-hash ring with virtual nodes: the ring "
       "class, the hashing, add/remove node, key lookup, and a short usage example. Be thorough. ")


def one(results, idx):
    body = {"model": MODEL, "max_tokens": MT, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"[s{idx}-{uuid.uuid4().hex[:6]}] {ASK}"}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    usage, t0, t_first = None, time.perf_counter(), None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
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
                if t_first is None and (c.get("choices") or [{}])[0].get("delta", {}).get("content"):
                    t_first = time.perf_counter()
    except Exception as e:  # noqa: BLE001
        results[idx] = {"err": type(e).__name__}
        return
    t_end = time.perf_counter()
    results[idx] = {"usage": usage, "t0": t0, "t_first": t_first, "t_end": t_end}


print("  {:>7}{:>10}{:>12}{:>12}{:>12}{:>11}".format(
    "并发", "总输出", "窗口s", "聚合T/s", "单流T/s", "首token s"))
print("  " + "-" * 66)
for n in STREAMS:
    results = {}
    ths = [threading.Thread(target=one, args=(results, i)) for i in range(n)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    ok = {k: v for k, v in results.items() if "usage" in v and v["usage"]}
    if not ok:
        print("  {:>7}  全部失败".format(n))
        continue
    total = sum(v["usage"]["completion_tokens"] for v in ok.values())
    first = min(v["t_first"] or v["t0"] for v in ok.values())
    last = max(v["t_end"] for v in ok.values())
    window = last - first
    per = [v["usage"].get("completion_tokens_per_sec", 0) for v in ok.values()]
    ttft = [((v["t_first"] or v["t0"]) - v["t0"]) for v in ok.values()]
    print("  {:>7}{:>10,}{:>12.2f}{:>12.1f}{:>12.1f}{:>11.2f}".format(
        len(ok), total, window, total / window if window else 0,
        sum(per) / len(per), max(ttft)))
    time.sleep(2)
