#!/usr/bin/env python3
"""Reproduce the identity violation on demand using a stop string.

The mechanism, from job.py's checkpoint rewind path:

    off_tokens = self.held_tokens.slice(len(self.checkpoint["held_tokens"]), None)
    self.held_text   = self.checkpoint["held_text"]
    self.held_tokens = self.checkpoint["held_tokens"]
    ...
    if replay_from is not None:
        seq.kv_position = replay_from
        seq.prefill_complete = False

Tokens are held back so a stop string spanning a chunk boundary can be detected. On a match the job
rolls the sequence and the held text back to a checkpoint and replays (re-prefills) the span, while
`accepted_draft_tokens` keeps counting across the rollback - which is why the identity
`accepted <= output` breaks, and why the published tok/s (output / wall) understates the work done.

This drives a long generation that will certainly contain a chosen stop string, so the rewind fires
mid-stream, then checks the identity. `stop` is a per-request parameter, so a client that sends
stop sequences is exactly what triggers this.
"""
import json
import sys
import time
import urllib.request
import uuid

MT = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
STOP = sys.argv[2] if len(sys.argv) > 2 else "</think>"
URL = "http://10.100.64.1:8899/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Local"

ASK = ("请写一份很长的技术文档，主题是分布式键值存储的一致性哈希与再平衡，"
       "逐节详尽展开：背景与问题定义、一致性哈希的数学原理、虚拟节点方案、"
       "数据迁移的增量算法、迁移期间的一致性保证、故障场景分析、容量规划公式、"
       "完整的伪代码实现。内容要尽可能长，不要省略任何细节。")


def run(label, extra):
    body = {"model": MODEL, "max_tokens": MT, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"[{uuid.uuid4().hex[:8]}] {ASK}"}]}
    body.update(extra)
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    usage = None
    t0 = time.perf_counter()
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
    except Exception as e:  # noqa: BLE001
        print(f"  {label:<22} 失败: {type(e).__name__}")
        return None
    wall = time.perf_counter() - t0
    if not usage:
        print(f"  {label:<22} 无 usage")
        return None
    det = usage.get("completion_tokens_details") or {}
    acc = det.get("accepted_prediction_tokens") or 0
    out = usage["completion_tokens"]
    tps = usage.get("completion_tokens_per_sec", 0)
    ok = acc <= out
    print(f"  {label:<22} 输出 {out:>6,}  {tps:>6.1f} T/s  接受 {acc:>7,}  "
          f"{'✓ 恒等式成立' if ok else '★ 违反！丢弃 {:,.0f} tok'.format(acc - out)}  (墙钟 {wall:.0f}s)")
    return ok


print(f"##### stop 串触发回退的验证 · max_tokens={MT} #####")
print(f"  stop 串: {STOP!r}")
print()
a = run("无 stop（对照）", {})
time.sleep(3)
b = run(f"stop={STOP!r}", {"stop": [STOP]})
time.sleep(3)
c = run("stop=长串（必中）", {"stop": ["一致性哈希"]})

print()
if a is not None and b is not None:
    print(f"  对照: {'成立' if a else '违反'}   stop 串: {'成立' if b else '违反'}")
if a is not None and c is not None:
    print(f"  对照: {'成立' if a else '违反'}   长 stop: {'成立' if c else '违反'}")
