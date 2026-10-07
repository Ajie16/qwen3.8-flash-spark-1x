#!/usr/bin/env python3
"""Server-side decode benchmark: what a client actually sees through TabbyAPI.

The recipe's own harnesses drive exllamav3's `Generator` in-process. This one goes over HTTP the way
dsh/kimi do and reads the server's counters out of the usage block, which TabbyAPI only emits when
the request asks for it (`stream_options.include_usage` — the code path is
`return_usage = data.stream_options and data.stream_options.include_usage`). A first version sent a
plain non-streaming request, got `usage: null`, and crashed on it.

Prompts carry a fresh nonce so the prefix cache cannot serve them; without that a repeat request
reports almost no prefill and flatters the numbers.

Reports per load: prompt tokens, prefill tok/s, output tokens, decode tok/s, draft acceptance, wall
time. Acceptance is derived from `accepted_prediction_tokens` / `rejected_prediction_tokens`.

Usage:
    python bench_serve.py [max_tokens] [prompt,list] [thinking on|off] [url]

Example (the pair that produced docs/exllamav3-tabbyapi-benchmark.md):
    python bench_serve.py 400 code,prose,zh_code,zh_prose off
    python bench_serve.py 400 code,prose,zh_code,zh_prose on
"""
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid

PROMPTS = {
    "code": "Write a Python function that parses an nginx access log line into a dict with fields ip, "
            "timestamp, method, path, status, bytes. Include a docstring, type hints, and a short "
            "usage example. Then explain each regex group in one bullet each.",
    "prose": "Write a vivid 350-word short story about a lighthouse keeper on a remote island in "
             "Alaska who discovers something unexpected washed ashore after a storm. Use varied "
             "sentence structure and specific sensory details.",
    "zh_code": "用 Python 写一个把 nginx 访问日志行解析成字典的函数，字段为 ip、时间戳、方法、路径、"
               "状态码、字节数。包含文档字符串、类型标注和简短用例，然后逐条解释每个正则分组。",
    "zh_prose": "写一篇约三百五十字的中文短篇故事，讲一位守灯塔的老人在暴风雨后发现岸边漂来的东西。"
                "要求句式多变，包含具体的感官细节。",
}

MAXTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 400
NAMES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["code", "prose", "zh_code", "zh_prose"]
THINK = (sys.argv[3] if len(sys.argv) > 3 else "off").lower() == "on"
URL = (sys.argv[4] if len(sys.argv) > 4 else "http://10.100.64.1:8899") + "/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Local"


def one(name):
    body = {"model": MODEL, "max_tokens": MAXTOK, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"[run {uuid.uuid4().hex[:8]}] {PROMPTS[name]}"}],
            "chat_template_kwargs": {"enable_thinking": THINK}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code} {e.read().decode()[:120]}"
    except Exception as e:  # noqa: BLE001
        return None, type(e).__name__
    if not usage:
        return None, "no usage block (is stream_options.include_usage set?)"
    det = usage.get("completion_tokens_details") or {}
    return {"prompt": usage["prompt_tokens"], "out": usage["completion_tokens"],
            "pf_tps": usage.get("prompt_tokens_per_sec"),
            "dec_tps": usage.get("completion_tokens_per_sec"),
            "wall": time.perf_counter() - t0,
            "acc": det.get("accepted_prediction_tokens"),
            "rej": det.get("rejected_prediction_tokens")}, None


def main() -> int:
    print(f"##### TabbyAPI server-side · max_tokens={MAXTOK} · thinking={THINK} · {URL} #####")
    print(f"{'load':<10}{'prompt':>8}{'prefill_t/s':>13}{'out':>6}{'decode_t/s':>12}{'accept':>8}{'wall_s':>9}")
    print("-" * 68)
    rows = []
    for name in NAMES:
        if name not in PROMPTS:
            print(f"{name:<10}  no such prompt")
            continue
        r, err = one(name)
        if err:
            print(f"{name:<10}  {err}")
            continue
        tot = (r["acc"] or 0) + (r["rej"] or 0)
        acc = 100 * (r["acc"] or 0) / tot if tot else 0
        print(f"{name:<10}{r['prompt']:>8,}{r['pf_tps'] or 0:>13.0f}{r['out']:>6}{r['dec_tps'] or 0:>12.1f}"
              f"{acc:>7.0f}%{r['wall']:>9.1f}")
        rows.append((r["dec_tps"] or 0, r["out"]))
    if rows:
        print("-" * 68)
        print(f"{'mean':<10}{'':>8}{'':>13}{'':>6}{statistics.mean(v for v, _ in rows):>12.1f}")
        short = [v for v, o in rows if o < MAXTOK]
        if short:
            print(f"  note: {len(short)} run(s) stopped before max_tokens; those rates cover fewer "
                  f"tokens and are less comparable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
