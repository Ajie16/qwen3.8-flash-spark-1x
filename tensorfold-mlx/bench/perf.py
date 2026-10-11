#!/usr/bin/env python3
"""Performance pass on the served checkpoint.

Protocol, and why each part is here:

  * A warmup request is sent and discarded. The first request after a restart runs ~20% slow on
    this box (61 ms/round against 49 warm on the same deterministic case), and letting it land on
    whichever case happens to be first is how a whole-suite comparison produced a confident but
    completely wrong answer earlier in this session.
  * The four sparkDash types (structured / prose / code / json) use the prompt catalog from
    MiaAI-Lab/sparkDash src/shared/llmPrompts.js verbatim, so the numbers can be compared against
    the published tables. Note "prose" there is the HASH-MAP EXPLANATION prompt, not creative
    writing -- the label is misleading and the distinction is worth ~1.5x.
  * Creative prose is measured separately, because the published "prose" figure hides this gap.
  * Deterministic cases (greedy) are repeated, since the same byte-identical output still varies
    run to run; the median is reported rather than a single sample.
  * tensorfold.drafted / accepted / rounds come back in every response, which is what separates
    "acceptance collapsed" from "rounds got slower" without server-side instrumentation.
"""
from __future__ import annotations

import json
import random
import statistics
import time
import urllib.request

API = "http://10.100.65.1:8899/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Local"

# verbatim from sparkDash src/shared/llmPrompts.js
DASH_STRUCTURED = "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
DASH_PROSE = ("Write a detailed step-by-step explanation of how a hash map works, "
              "including collision handling, resizing, and time complexity. Be thorough.")
DASH_CODE_TAIL = ("Output only Python source. No comments, no docstrings, no markdown fences. "
                  "Then add tests and the helpers this needs. Keep writing code.")
DASH_CODE = ("def binary_search(nums, target) -> int: index of target in a sorted list, or -1.\n"
             + DASH_CODE_TAIL)
DASH_JSON = ("Write a JSON object for an invoice with seller, buyer, ten line items, taxes and totals.")


def chat(prompt, max_tokens, think=False, timeout=1800):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    wall = time.time() - t0
    usage = d.get("usage") or {}
    n = int(usage.get("completion_tokens") or 0)
    tf = dict(d.get("tensorfold") or {})
    tf["prompt_tokens"] = int(usage.get("prompt_tokens") or 0)   # lives in usage, not tensorfold
    return {"n": n, "wall": wall, "tf": tf,
            "rate_wall": (n - 1) / wall if wall > 0 and n > 1 else 0,
            "rate_srv": (n - 1) / tf["decode_s"] if tf.get("decode_s") and n > 1 else 0}


def line(name, r):
    tf = r["tf"]
    rounds = tf.get("rounds") or 0
    dr = tf.get("drafted") or 0
    ac = tf.get("accepted") or 0
    print("  %-22s %6.1f tok/s (%5.1f srv)  %4d tok  rounds=%-4d draft/r=%.2f acc=%4.1f%%"
          % (name, r["rate_wall"], r["rate_srv"], r["n"], rounds,
             dr / rounds if rounds else 0, 100 * ac / dr if dr else 0))


def main():
    print("  warmup (discarded) ...", flush=True)
    chat("预热", 64)
    print()

    print("  === sparkDash 官方四类（可与 Mia 公布数字对比）===")
    for name, p, mt in (("structured 数数", DASH_STRUCTURED, 400),
                        ("prose hash-map", DASH_PROSE, 600),
                        ("code binary_search", DASH_CODE, 600),
                        ("json invoice", DASH_JSON, 600)):
        try:
            line(name, chat(p, mt))
        except Exception as e:
            print("  %-22s failed: %s" % (name, e))

    print()
    print("  === 创作类（公开数字里看不到的那一档）===")
    for name, p in (("创作·中文散文", "写一篇关于城市夜间经济的散文。"),
                    ("创作·英文散文", "Write an essay about the nighttime economy of cities."),
                    ("中文技术讲解", "解释一下 MoE 混合专家模型里的专家路由是怎么工作的。")):
        try:
            line(name, chat(p, 600))
        except Exception as e:
            print("  %-22s failed: %s" % (name, e))

    print()
    print("  === 确定性用例重复（中位）===")
    reps = 5
    for name, p in (("code 5 轮", DASH_CODE), ("hash-map 5 轮", DASH_PROSE)):
        rates = []
        for i in range(reps):
            try:
                r = chat(p, 600)
                if i == 0:
                    print("    (首轮 %.1f tok/s, %.1f ms/round — 冷态，计入中位但看趋势)"
                          % (r["rate_wall"], 1000 * r["tf"]["decode_s"] / r["tf"]["rounds"]))
                rates.append(r["rate_wall"])
            except Exception as e:
                print("    run %d failed: %s" % (i + 1, e))
        if rates:
            print("  %-22s 中位 %.1f tok/s   极差 %.1f–%.1f   all=%s"
                  % (name, statistics.median(rates), min(rates), max(rates),
                     " ".join("%.0f" % x for x in rates)))

    print()
    print("  === prefill（nonce 在头部防前缀缓存）===")
    for target in (4000, 16000, 32000, 75000):
        nonce = "%d-%f" % (time.time_ns(), random.random())
        body = ("资料编号 %s。以下为背景材料，请阅读后一句话总结。\n" % nonce
                + "人工智能的发展经历了符号主义、连接主义和行为主义的多次起伏。" * (target // 30 + 1)
                + "\n请用一句话总结。")
        try:
            r = chat(body, 5, timeout=1800)
            tf = r["tf"]
            pt = tf.get("prompt_tokens") or 0
            ps = tf.get("prefill_s") or 0
            print("  prefill @~%-7d %8.0f tok/s   (%.0f tok / %.1fs)  decode_s=%.2f"
                  % (target, pt / ps if ps else 0, pt, ps, tf.get("decode_s") or 0))
        except Exception as e:
            print("  prefill @%-8d failed: %s" % (target, e))


if __name__ == "__main__":
    main()
