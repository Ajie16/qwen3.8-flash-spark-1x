#!/usr/bin/env python3
"""Run an agent-shaped request and analyse the round log it produces.

Agent traffic differs from the synthetic benchmarks in three ways that were never combined: a long
continuation prompt (20k-100k), tools in the request, and thinking on. The round instrumentation
can say whether such a request has a regime where verify_ms explodes, the window collapses, or the
acceptance falls - none of which a per-request average can show.

Usage: agent_round_test.py [prompt_tokens] [max_tokens]
"""
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

PT = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
MT = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
LOG = "/home/xujie/serve-exl3-native.log"
URL = "http://10.100.64.1:8899/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Local"

FILLER = ("本节为工程背景材料，说明分布式存储系统的设计约束，涵盖一致性哈希、虚拟节点、"
          "数据迁移与再平衡、副本一致性、故障检测与恢复流程。以下内容不含待回答的问题。")
PARAS = max(1, PT // 34)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the workspace and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the file"}},
            "required": ["path"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": "Run the project test suite and return the summary.",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string", "description": "Test name filter"}},
            "required": [],
        },
    },
}]

ASK = ("\n\n请阅读 src/consistent_hash.py 的实现，找出再平衡逻辑中的问题，"
       "然后写一份修复方案并调用 run_tests 验证。请分步进行。")


def round_lines_before():
    try:
        out = subprocess.run(["grep", "-c", r"\[round\]", LOG], capture_output=True, text=True).stdout
        return int(out.strip() or 0)
    except Exception:
        return 0


def main() -> int:
    n0 = round_lines_before()
    prompt = f"[会话 {uuid.uuid4().hex[:8]}] " + FILLER * PARAS + ASK
    body = {"model": MODEL, "max_tokens": MT, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    print(f"##### agent 形态 · 目标 prompt {PT:,} · max_tokens {MT} · 工具 + xhigh 思考 #####")
    t0 = time.perf_counter()
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
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
    except urllib.error.HTTPError as e:
        print("  HTTP", e.code, e.read().decode()[:200])
        return 1
    except Exception as e:  # noqa: BLE001
        print("  failed:", type(e).__name__, e)
        return 1
    wall = time.perf_counter() - t0
    if usage:
        ptd = usage.get("prompt_tokens_details") or {}
        print(f"  prompt {usage['prompt_tokens']:,} (cached {ptd.get('cached_tokens', 0) or 0:,}) "
              f"prefill {usage.get('prompt_tokens_per_sec', 0):.0f} T/s")
        print(f"  输出 {usage['completion_tokens']:,}  decode {usage.get('completion_tokens_per_sec', 0):.1f} T/s  "
              f"墙钟 {wall:.1f}s")

    time.sleep(2)
    lines = subprocess.run(["grep", r"\[round\]", LOG], capture_output=True, text=True).stdout.splitlines()
    new = lines[n0:]
    print(f"\n  本次产生 {len(new)} 行 round 日志")
    if not new:
        return 1

    rows = []
    for ln in new:
        m = re.search(r"n=(\d+) jobs=(\d+) window=(-?\d+) draft_ms=([\d.]+) verify_ms=([\d.]+) "
                      r"total_ms=([\d.]+) acc=\+(-?\d+) rej=\+(-?\d+)", ln)
        if m:
            rows.append(dict(n=int(m.group(1)), jobs=int(m.group(2)), win=int(m.group(3)),
                             draft=float(m.group(4)), verify=float(m.group(5)),
                             total=float(m.group(6)), acc=int(m.group(7)), rej=int(m.group(8))))
    if not rows:
        print("  解析失败，原始行：")
        for ln in new[:3]:
            print("   ", ln)
        return 1

    def stats(key):
        v = sorted(r[key] for r in rows)
        return v[0], v[len(v) // 2], v[-1]

    print(f"  {'指标':<12}{'最小':>9}{'中位':>9}{'最大':>9}")
    for k, label in (("win", "window"), ("draft", "draft_ms"), ("verify", "verify_ms"),
                     ("total", "total_ms"), ("acc", "acc/round"), ("rej", "rej/round")):
        lo, md, hi = stats(k)
        print(f"  {label:<12}{lo:>9.1f}{md:>9.1f}{hi:>9.1f}")

    print("\n  分段趋势（每 10% 轮数一段）:")
    seg = max(1, len(rows) // 10)
    print(f"  {'段':>4}{'轮数':>6}{'window':>8}{'draft_ms':>10}{'verify_ms':>11}{'total_ms':>10}{'acc':>6}")
    for i in range(0, len(rows), seg):
        ch = rows[i:i + seg]
        print(f"  {i // seg + 1:>4}{len(ch):>6}"
              f"{sum(r['win'] for r in ch) / len(ch):>8.1f}"
              f"{sum(r['draft'] for r in ch) / len(ch):>10.1f}"
              f"{sum(r['verify'] for r in ch) / len(ch):>11.1f}"
              f"{sum(r['total'] for r in ch) / len(ch):>10.1f}"
              f"{sum(r['acc'] for r in ch) / len(ch):>6.2f}")

    worst = sorted(rows, key=lambda r: -r["verify"])[:5]
    print("\n  verify_ms 最慢的 5 轮:")
    for r in worst:
        print(f"    n={r['n']} jobs={r['jobs']} window={r['win']} draft={r['draft']:.1f} "
              f"verify={r['verify']:.1f} total={r['total']:.1f} acc=+{r['acc']} rej=+{r['rej']}")
    outside = [r for r in rows if r["total"] > r["draft"] + r["verify"] + 5]
    print(f"\n  total > draft+verify+5ms 的轮（卡在循环外）: {len(outside)} / {len(rows)}")
    for r in outside[:5]:
        print(f"    n={r['n']} total={r['total']:.1f} vs draft+verify={r['draft'] + r['verify']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
