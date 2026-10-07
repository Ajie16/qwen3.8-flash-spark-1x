# TabbyAPI server-side benchmark — Qwen3.8-Flash-Next EXL3 on one DGX Spark

Measured 2026-10-07 on **spark-2** (GB10, 121.7 GiB unified), against the running service at
`http://10.100.64.1:8899/v1`, through HTTP exactly as dsh/kimi use it.

This fills the gap the upstream recipe names in its own *Known limitations*:

> **Throughput through TabbyAPI is not yet re-measured on the current pin, in either profile.** The
> per-stream numbers above are `chat.py`, and the concurrency table is stock 1.5.0 at k=3.

Harness: [`../exllamav3-tabby/bench/bench_serve.py`](../exllamav3-tabby/bench/bench_serve.py). It
streams, asks for `stream_options.include_usage`, and reads the server's own counters. Every prompt
carries a fresh nonce so the prefix cache cannot serve it.

Configuration at capture (see [`serve-local.sh`](../exllamav3-tabby/serve-local.sh) and
[`env.sh`](../exllamav3-tabby/env.sh)): `feat/hybrid-draft` worktree, `EXL3_MTP_HEAD_N=163840`,
`EXL3_GR_TUNED=1`, `EXL3_INT8_GEMV=0`, `EXL3_MOE_COOP_WIDE=1`, `EXL3_GR_INT8=1`,
`EXL3_NGRAM_STREAM=0`, `EXL3_DRAFT_CONFIDENCE=0.6`, `PROFILE=concurrent` with `NGRAM_RAM=true`
(4 concurrent jobs, 1,048,576-token shared KV pool, n-gram table resident).

---

## 1. Single stream, 400 tokens, sequential

The service was verified idle before each run (log silent >100 s, zero established connections,
process at 0% CPU), and one request is in flight at a time.

| Load | thinking **off** | accept | thinking **on** (xhigh) | accept | thinking costs |
|---|---:|---:|---:|---:|---:|
| code | **73.4** | 85% | 55.3 | 61% | **−25%** |
| prose | 49.7 | 55% | 51.5 | 60% | +4% (noise) |
| zh_code | **68.9** | 74% | 61.2 | 66% | −11% |
| zh_prose | 48.8 | 56% | 45.2 | 49% | −7% |
| **mean** | **60.2** | | **53.3** | | **−11%** |

Peak **73.4 tok/s** on code. One caveat carried from the run: `zh_prose` with thinking off produced
297 tokens rather than 400 (it stopped early), so that cell covers fewer tokens and is slightly less
comparable.

**Thinking costs about 11%, and the mechanism is draft acceptance, not extra work per token.** On
code the acceptance drops 85% → 61%; reasoning text is simply harder for the MTP head to predict
than code output is. That is a quality-for-speed trade the client chooses, not an engine property.

**One result contradicts the upstream note** that in-process harnesses read *higher* than the server
("about 4 tok/s for kernel changes and about 10 for host-side ones, because the per-token console
write is itself a host sync"). Here the server measured **73.4** on code against **68.7** in-process
for the same load. The two do not use the same prompt path — the in-process harness hardcodes a
`<think></think>` prefix while the server renders the full template — so this is recorded as an open
discrepancy rather than a refutation.

---

## 2. What the real workload achieves

Mined from the service log over the same session (29 completed requests). TabbyAPI logs the server's
counters per request, so this carries the concurrency and prompt-size mix the clients actually
produce — a better picture of the deployment than any synthetic run.

|  |  |
|---|---:|
| decode, median | **54.4 tok/s** |
| decode, min / max | 4.3 / **78.8** tok/s |
| output ≥ 1000 tokens (6 reqs), mean | 42.1 tok/s |
| output < 1000 tokens (23 reqs), mean | 49.7 tok/s |

**First-token latency is driven by prefix-cache hit rate, by a factor of three:**

| cached | requests | mean first token |
|---|---:|---:|
| **0%** | 8 | **11.1 s** |
| 51–90% | 2 | 16.2 s |
| **91–100%** | 19 | **3.7 s** |

Two requests make the point in the raw log: `#3` had a 50,937-token prompt with **none** cached and
took **61.5 s** to first token at 828 tok/s; `#4` had a 52,678-token prompt with **96%** cached and
took **4.88 s**.

---

## 3. Why a client can see ~29 tok/s

The slow requests in the log are the ones that overlapped a long generation, not a slow engine.

```
18:28:12  #8  arrives, prompt 24,480, generates 6,662 over 265 s
18:29:15  #10 arrives           -> 16.8 tok/s
18:29:40  #11 arrives           ->  8.8 tok/s
18:30:27  #12 arrives           -> 16.2 tok/s
          #13                   ->  9.8 tok/s
after #8 finishes:
18:33+    #14 … #32, mostly sequential, cache 95-99%  -> 47 - 79 tok/s
```

Same pattern earlier: `#5` (169-token prompt, 64 tokens of output) took **21.3 s** for 4.3 tok/s
because its whole lifetime sat inside `#6`'s 11,889-token prefill.

This is what the engine is specified to do. The upstream concurrency table for exllamav3 gives
**104.4 tok/s aggregate at 4 streams, i.e. 26.1 tok/s per stream**, and the recipe says so outright:

> Per-stream rates at eight streams are 17 to 20 tok/s either way, so that is **throughput for a
> queue, not eight interactive users**.

So ~29 tok/s is the per-stream rate at roughly four concurrent jobs on a decode that is bound by
shared weight bandwidth, plus starvation while a long prompt prefills. It is not a regression.

**Ordering of levers that follow from this**, largest first:

1. **Client concurrency.** 4 streams → 26 tok/s each; 2 streams → 44. Dropping concurrency is worth
   more than any remaining kernel knob.
2. **Prefix-cache hit rate.** 11.1 s vs 3.7 s to first token. Anything that rewrites the prompt
   prefix — context compaction, a timestamp or random id in the system prompt — throws this away.
3. **Thinking on/off.** −11% for the whole session, chosen per client.
4. Kernel knobs. The remaining candidates are in
   [`exllamav3-optimization-backlog.md`](exllamav3-optimization-backlog.md).

---

## 4. Prefill

Measured earlier the same day, same box and engine, but with `NGRAM_RAM=false` (the `concurrent`
profile's default), so each cold request paid the n-gram table read from NVMe:

| prompt | tok/s |
|---|---:|
| 8k | 825 |
| 24k | 941 |
| 64k | 1,023 |
| 128k | 1,037 |
| 240k | 1,027 |

Fitting `intercept + slope × tokens` across those points gives a slope of 0.963 ms/token
(asymptotic **1,038 tok/s**) and an **intercept of ≈2.2 s** — the fixed per-request cost, which is
the n-gram read the recipe quotes at "about 1.8 s on a cold short prompt". Throughput therefore
*rises* with length before flattening, which is the fixed cost being amortised, not a property of
long contexts.

The service in section 1 runs `NGRAM_RAM=true`, which should remove that intercept, but **that has
not been measured** — the log's own prefill rates on cached long prompts run 100–530 tok/s, which is
the incremental-chunk cost, not a cold-prompt figure.

For scale: the MLX 4-bit affine path on this box reaches **2,425 tok/s** at 32k. EXL3 trellis
dequantisation is per-token compute and cannot be fused into the tensor-core matmul the way the MLX
dequant (multiply-add) can, so the gap is the format's, not this engine's — the same ceiling was
reached independently on the TensorFold EXL3 path (~820 tok/s) with a chunk-size experiment that
left throughput flat.
