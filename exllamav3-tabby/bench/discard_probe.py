#!/usr/bin/env python3
"""Decide between two explanations for the #8 anomaly: discarded work, or a reporting bug.

The round log's own accounting says one request produced 14,163 tokens while TabbyAPI reported
4,111. Two explanations fit:

  (a) The engine really generated 14,163 tokens and threw ~10,000 away. The client got 4,111
      tokens' worth of text, and the GPU time was wasted.
  (b) All 14,163 tokens reached the client; `new_tokens` under-reports, so only the published rate
      is wrong.

They are distinguished by the response text itself. This captures the full streamed text, counts its
tokens with the model's own tokenizer, and prints it beside the reported count and the round-derived
count. If the text holds ~14k tokens it is (b); if it holds ~4k it is (a).

Usage: discard_probe.py [prompt_tokens] [max_tokens]
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
import uuid

PT = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
MT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
LOG = "/home/xujie/serve-exl3-native.log"
URL = "http://10.100.64.1:8899/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Local"

FILLER = ("本节为工程背景材料，说明分布式存储系统的设计约束，涵盖一致性哈希、虚拟节点、"
          "数据迁移与再平衡、副本一致性、故障检测与恢复流程。以下内容不含待回答的问题。")
PARAS = max(1, PT // 34)
ASK = ("\n\n请写一份完整的技术设计文档，主题是分布式键值存储的一致性哈希与再平衡，"
       "逐节详尽展开：背景与问题定义、数学原理、虚拟节点方案、增量迁移算法、"
       "迁移期间的一致性保证、故障场景分析、容量规划公式、完整伪代码。")


def marker():
    try:
        return int(subprocess.run(["grep", "-c", r"\[iter\]", LOG],
                                  capture_output=True, text=True).stdout.strip() or 0)
    except Exception:
        return 0


def main() -> int:
    n0 = marker()
    body = {"model": MODEL, "max_tokens": MT, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"[{uuid.uuid4().hex[:8]}] "
                                                      + FILLER * PARAS + ASK}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    print(f"##### 丢弃探针 · 目标 prompt {PT:,} · max_tokens {MT:,} #####")
    t0 = time.perf_counter()
    text, reasoning, usage = [], [], None
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
                for ch in c.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content"):
                        text.append(d["content"])
                    if d.get("reasoning_content"):
                        reasoning.append(d["reasoning_content"])
    except Exception as e:  # noqa: BLE001
        print("  failed:", type(e).__name__, e)
        return 1
    wall = time.perf_counter() - t0
    full = "".join(text)
    think = "".join(reasoning)
    time.sleep(2)

    # round-derived tokens for this request: the contiguous [iter] block after n0
    lines = subprocess.run(["grep", r"\[iter\]", LOG], capture_output=True, text=True).stdout.splitlines()
    rounds = []
    for ln in lines[n0:]:
        m = re.search(r"jobs=(\d+) ser=([\d,\-]*) win=(-?\d+).*?acc=\+(-?\d+)", ln)
        if m:
            rounds.append((int(m.group(2).split(",")[0]) if m.group(2) else -1,
                           int(m.group(3)), int(m.group(4))))
    single = [r for r in rounds if r[1] > 0]
    round_tok = sum(r[2] + 1 for r in single)

    # count the response with the model's own tokenizer
    tok = None
    try:
        sys.path.insert(0, "/home/xujie/qwen38-exl3/exllamav3-spec")
        from exllamav3 import Config, Tokenizer
        cfg = Config.from_directory("/home/xujie/models/Qwen3.8-Flash-Next-Uncensored-EXL3-3bpw")
        tk = Tokenizer.from_config(cfg)
        tok = len(tk.encode(full, add_bos=False)[0])
        tok_think = len(tk.encode(think, add_bos=False)[0])
    except Exception as e:  # noqa: BLE001
        tok = None
        tok_think = 0
        print("  (tokenizer 不可用:", type(e).__name__, ")")

    print()
    print("  服务端上报 completion_tokens : {:,}".format(usage["completion_tokens"] if usage else -1))
    print("  实际收到的正文 token       : {}".format("{:,}".format(tok) if tok else "?"))
    print("  实际收到的思考 token       : {:,}".format(tok_think))
    print("  轮日志产出 (acc+1 合计)     : {:,}  ({} 个单 job 轮)".format(round_tok, len(single)))
    print("  墙钟                        : {:.1f} s".format(wall))
    if usage:
        print("  上报 decode T/s             : {:.1f}".format(usage.get("completion_tokens_per_sec", 0)))
    print()
    if tok:
        total_recv = tok + tok_think
        print("  ── 判定 ──")
        print("    收到正文/轮产出 = {:.2f}x".format(total_recv / round_tok if round_tok else 0))
        if total_recv > 0.9 * round_tok:
            print("    ★ 收到的 token ≈ 轮产出 → 全部到达客户端，(b) 只是上报数偏小")
        elif total_recv < 0.7 * round_tok:
            print("    ★ 收到的 token 远小于轮产出 → (a) 工作被丢弃，GPU 时间浪费")
        else:
            print("    ? 介于两者之间，需再测")
    return 0


if __name__ == "__main__":
    sys.exit(main())
