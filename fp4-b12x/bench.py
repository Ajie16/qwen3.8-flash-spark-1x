#!/usr/bin/env python3
"""Benchmark the FP4-Local + B12X service. Three modes, one script.

Replaces five near-duplicate scripts (fp4bench6 / fp4c8 / fp4hi / fp4prefill / accept) that had
drifted apart in argument handling and start-up warm-up.

    python3 bench.py decode  --n 1,2,4,8            # aggregate + per-stream decode tok/s
    python3 bench.py prefill --sizes 4000,16000,32000,64000,128000
    python3 bench.py accept                          # MTP acceptance per content type

Notes that matter when reading the numbers:

  * Every mode discards warm-up requests. The first request after a restart runs cold and reads
    ~20% low, which is enough to invert a config comparison.
  * decode and accept report the post-first-token window; prefill reports TTFT, which on an
    uncached prompt is the prefill cost. Mixing the two windows is how a "312 tok/s" claim gets made
    about a service that returns 108.
  * prefill sends a fresh random prefix every time. Prefix caching is on, so a repeated prompt would
    report 30 ms for a 64k-token prefill and mean nothing.
  * accept reads the cumulative counters from /metrics and diffs them around each case, because
    vLLM reports them cumulatively, not per request.
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import threading
import time
import urllib.request

DEFAULT_API = "http://10.100.65.1:8899"
DEFAULT_MODEL = "Qwen3.8-Flash-Local"

DECODE_PROMPT = "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
FILLER = ("The maintenance log for reactor bay seven records a pressure excursion at oh four hundred "
          "hours, followed by a manual override and a return to nominal. ")

ACCEPT_CASES = {
    "count(most predictable)": "Count from 1 to 200. Output only the numbers.",
    "code": "Write a Python implementation of a thread-safe LRU cache with get/put, tests included.",
    "prose-en": "Explain how a hash map works, covering collisions, resizing and complexity.",
    "prose-zh": "写一篇关于城市夜间经济的说明文，包括消费结构、就业带动和治理难点。",
    "technical-zh": "详细解释一致性哈希的原理、虚拟节点、扩容时的数据迁移过程。",
    "json": "Output a JSON invoice with seller, buyer, ten line items, taxes and totals.",
}


def _post(api: str, body: dict, timeout: float = 900):
    req = urllib.request.Request(api + "/v1/chat/completions", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _stream_once(api: str, model: str, prompt: str,
                 max_tokens: int) -> tuple[int, float, float, int]:
    """(completion_tokens, ttft_s, decode_s, prompt_tokens) for one streamed request."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    ttft = None
    usage: dict = {}
    with _post(api, body) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            ch = d.get("choices") or []
            if ch:
                delta = ch[0].get("delta") or {}
                if (delta.get("content") or delta.get("reasoning_content")) and ttft is None:
                    ttft = time.time() - t0
            if d.get("usage"):
                usage = d["usage"]
    end = time.time()
    return (int(usage.get("completion_tokens") or 0), ttft or 0.0,
            max(end - t0 - (ttft or 0.0), 0.001), int(usage.get("prompt_tokens") or 0))


def cmd_decode(args) -> None:
    for _ in range(args.warmup):
        _stream_once(args.api, args.model, DECODE_PROMPT, 200)
    print("  %-5s %10s %14s %9s" % ("N", "aggregate", "per-stream", "TTFT"))
    for n in [int(x) for x in args.n.split(",")]:
        results: list = [None] * n
        def worker(i: int) -> None:
            results[i] = _stream_once(args.api, args.model, DECODE_PROMPT, args.max_tokens)[:3]
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ok = [r for r in results if r]
        tokens = sum(r[0] for r in ok)
        aggregate = tokens / max(r[2] for r in ok)
        per = [round(r[0] / r[2], 1) for r in ok]
        print("  %-5d %10.1f %5.1f–%-6.1f %7.0fms"
              % (n, aggregate, min(per), max(per), 1000 * ok[0][1]))


def cmd_prefill(args) -> None:
    def unique_prompt(target: int) -> str:
        tag = "".join(random.choices(string.ascii_letters + string.digits, k=24))
        return tag + " " + FILLER * (target // 20 + 2)

    for _ in range(args.warmup):
        _stream_once(args.api, args.model, unique_prompt(2000), 1)
    print("  %10s %10s %14s" % ("prompt tok", "TTFT s", "prefill tok/s"))
    for target in [int(x) for x in args.sizes.split(",")]:
        try:
            # max_tokens=1 keeps decode out of the measurement; prompt_tokens comes from usage, so
            # the reported size is the tokenizer's count rather than a character estimate
            _, ttft, _, ptoks = _stream_once(args.api, args.model, unique_prompt(target), 1)
        except Exception as exc:                     # noqa: BLE001 - report and keep going
            print("  %10d  failed: %s" % (target, str(exc)[:60]))
            continue
        print("  %10d %10.2f %14.0f" % (ptoks, ttft, ptoks / ttft if ttft else 0))


def _spec_counters(api: str) -> tuple[float, float]:
    with urllib.request.urlopen(api + "/metrics", timeout=60) as r:
        text = r.read().decode()

    def grab(field: str) -> float:
        import re
        m = re.search(r"vllm:spec_decode_%s_total\{[^}]*\}\s+([0-9.]+)" % field, text)
        return float(m.group(1)) if m else 0.0

    return grab("num_draft_tokens"), grab("num_accepted_tokens")


def cmd_accept(args) -> None:
    for _ in range(args.warmup):
        _stream_once(args.api, args.model, "warmup", 64)
    print("  %-24s %8s %8s %9s" % ("case", "drafts", "accepted", "rate"))
    for name, prompt in ACCEPT_CASES.items():
        d0, a0 = _spec_counters(args.api)
        _stream_once(args.api, args.model, prompt, args.max_tokens)
        d1, a1 = _spec_counters(args.api)
        dd, da = d1 - d0, a1 - a0
        rate = 100.0 * da / dd if dd else 0.0
        print("  %-24s %8d %8d %8.1f%%" % (name, int(dd), int(da), rate))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("decode", "prefill", "accept"))
    ap.add_argument("--api", default=DEFAULT_API, help="base URL, default %s" % DEFAULT_API)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=2, help="discarded warm-up requests per mode")
    ap.add_argument("--n", default="1,2,4,8", help="decode: concurrency levels")
    ap.add_argument("--sizes", default="4000,16000,32000,64000,128000", help="prefill: prompt sizes")
    args = ap.parse_args()
    {"decode": cmd_decode, "prefill": cmd_prefill, "accept": cmd_accept}[args.mode](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
