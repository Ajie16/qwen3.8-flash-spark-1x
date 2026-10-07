# Decode-rate drop: root cause found and fixed

> **Confirmed in production use, 2026-10-07.** After restarting on the fixed engine the operator
> reports the drop no longer occurs on real agent traffic.

**Root cause: one line in `exllamav3/generator/job.py`. `rq_new_tokens` was assigned the current
segment's token count instead of the running total, so on the second and later requeues every
earlier segment was dropped from the reported output. The engine never slowed down — the published
decode rate was computed from a truncated token count over the full elapsed time.**

Fixed in [`0005-rq-token-accounting.patch`](0005-rq-token-accounting.patch), applied by
[`apply_rq_token_fix.py`](apply_rq_token_fix.py). Verified.

---

## The defect

`prepare_for_requeue()` builds the state a requeued job resumes from:

```python
"rq_new_tokens": self.new_tokens,   # every token accepted so far counts; ...
```

The comment says "every token accepted so far counts", but `self.new_tokens` is only the **current**
segment — `__init__` resets it to 0 for the requeued job, and `self.rq_new_tokens` is what holds the
earlier total. At report time the two are added:

```python
"new_tokens": self.rq_new_tokens + self.new_tokens,
```

so on the **second and later** requeues the running total is overwritten by the current segment's
count and everything before it is lost:

| | segment 1 | after requeue 1 | segment 2 | after requeue 2 | segment 3 | reported |
|---|---:|---:|---:|---:|---:|---:|
| correct | 6,005 | 6,005 | 4,096 | 10,101 | 4,096 | 14,197 + rounds |
| actual (bug) | 6,005 | 6,005 | 4,096 | **4,096** | 4,096 | 4,096 + rounds |

`accepted_draft_tokens` is carried whole (`self.accepted_draft_tokens` already includes every prior
segment, and the constructor reads it straight back), so it does **not** lose them. That asymmetry is
the observable signature: `accepted_draft_tokens > new_tokens`, which is impossible for a
speculative job because a round emits `accepted + 1` tokens.

## Why it looked like a slowdown

Every round is healthy — measured over 13,291 rounds of the reproducing workload:

| | median | range |
|---|---:|---:|
| `window` | 2 | 1 – 5 |
| `draft_ms` | 3.9 | 1.9 – 23.7 |
| `verify_ms` | **37.7** | 31.5 – 68.2 |
| `total_ms` | 41.8 | 33.5 – 1380.7 |

`verify_ms` is flat at ~38 ms for the whole run. Nothing slows down. What changes is the numerator:
the reported output is truncated at each requeue, and the reported rate is `truncated_output /
full_elapsed`.

The trigger is requeue count, which is why it appears "after a while" and only on some requests. A
request requeues whenever it generates past `max_rq_tokens`, which with `output_chunking` at its
default is `chunk_size` (4096) aligned up to the recurrent checkpoint interval:

```
[requeue] ser=0 new_tokens=6005 rq_new_tokens=0     accepted=2965
[requeue] ser=0 new_tokens=4096 rq_new_tokens=6005  accepted=5197
[requeue] ser=0 new_tokens=4096 rq_new_tokens=10101 accepted=7851
```

Three requeues. With the bug, the report would keep only the last segment.

## Verification

Same request shape, same prompt, same `max_tokens`, before and after the one-line change:

| | output | tok/s | identity `accepted <= output` | wall |
|---|---:|---:|---|---:|
| **before** | 4,897 | **17.2** | **violated by 3,587** | 299 s |
| **after** | **15,000** | **51.4** | holds | 308 s |

A 3.0x change in the published rate for identical work. The second arm of the same test moved from
16.8 to 51.1 tok/s the same way, and the elapsed times match, confirming only the accounting changed.

Reproduce with `exllamav3-tabby/bench/stop_rewind_probe.py 15000 __nomatch__` and check the identity
with `exllamav3-tabby/bench/identity_check.py`.

## How it was found

The path there is worth keeping, because the first three explanations were all wrong:

1. **Nine external hypotheses measured and ruled out** — concurrency (the slow requests ran alone),
   GPU throttling (clock flat at 2190 MHz under 220 s of load, every throttle flag clear), context
   depth (`66.6 / 66.6 / 65.1` tok/s at 4k / 128k / 240k), memory pressure (`sar` identical across a
   slow and a fast window), `EXL3_GR_TUNED` (45.8 vs 44.4), `output_chunking`/requeue, client read
   speed (−13 %), constrained generation (−4 %/−5 %), output length alone (45–54 tok/s).
2. **`EXL3_SPEC_SAMPLING` tested and rejected** — see
   [spec-sampling-negative-result.md](spec-sampling-negative-result.md); it is slower on every
   content type, in both window configurations.
3. **Per-request instrumentation added** (`docs/round-logging.md`), which showed every round healthy
   and located the anomaly in the accounting rather than the kernel.
4. **The identity check** — `accepted_draft_tokens <= new_tokens` must hold for every request and
   needs no instrumentation at all. Exactly 1 of 14 requests violated it, and that was the slow one.
5. **The counter that can only ever increase vs the one that can decrease** — a full-repo grep showed
   `new_tokens` decremented in exactly one place (`rewind_checkpoint`, banned-string only) and
   `rq_new_tokens` overwritten in one place, which is the bug.

## Operational notes learned the hard way

- **`mv` does not redirect a running process.** Rotation moved the log while the service kept writing
  to the moved inode, so a whole test round was read from the wrong file. Either restart to rotate,
  or use a fresh filename per run.
- **Kill by PID, and check for survivors.** `kill $(pgrep ... | head -1)` left a second instance
  alive holding port 8899; the replacement then logged `Port 8899 is currently in use. Switching to
  8900.` and died on `Insufficient VRAM`, while the tests silently kept hitting the old, unfixed
  engine. A round of "the fix didn't work" was entirely this. Always confirm the port is free, the
  process start time is after the edit, and `readlink /proc/PID/fd/1` points where expected.
