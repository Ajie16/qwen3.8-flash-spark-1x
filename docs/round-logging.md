# Per-round instrumentation for the exllamav3 draft/verify loop

`EXL3_LOG_ROUNDS=1` makes the engine print one line per draft+verify round to stderr. Off by
default and read once at import, so it costs nothing when unset.

## Why it exists

The service log's per-request averages could not explain a slowdown that a client reported as
"53 tok/s, then 30 after a few rounds". Everything reachable from outside was measured and ruled
out:

| Hypothesis | How it was tested | Result |
|---|---|---|
| Concurrency | rebuilt the overlap of all 29 completed requests from arrival/completion timestamps | the slow ones (#7, #18, #44, #54) ran **alone** |
| GPU throttling | 220 s of sustained load, sampling every 20 s | SM clock flat at **2190 MHz**, temp 48→61 °C, every throttle flag `Not Active` |
| Context depth | upstream `depthab.sh`, 400 tokens at 4k / 128k / 240k | **66.6 / 66.6 / 65.1** — flat |
| Memory pressure | `sar` history across a slow window (#54) and a fast one (#58) | identical, 84 % used / 16.5 GiB avail |
| `EXL3_GR_TUNED` | same prompt, both settings, with restarts | 45.8 vs 44.4 tok/s |
| `output_chunking` / requeue | 4 combinations of cold/long prompt × on/off | 45.9 / 47.8 / 51.6 / 53.7 tok/s |
| Client read speed | one request drained fast, one with 0.25 s sleep per chunk | server dropped only **13 %** |
| Constrained generation | plain vs `json_schema` vs grammar | **−4 % / −5 %** |
| Output length alone | synthetic 6 000-token generations | consistently **45–54 tok/s** |

Nine synthetic workloads all land in 45–54 tok/s. The lone long requests in the log span
**14.8–60.3 tok/s**. Something inside a round accounts for that and no external measurement could
see it.

## What it prints

```
[round] n=1234 jobs=1 window=5 draft_ms=9.1 verify_ms=48.8 total_ms=57.9 acc=+1 rej=+4
```

| Field | Meaning |
|---|---|
| `n` | round counter, monotonic per process |
| `jobs` | active jobs in the batch |
| `window` | draft tokens attempted this round; dynamic drafting (`-dds`) truncates it |
| `draft_ms` | the draft model's forwards for the window |
| `verify_ms` | the target forward, sampling and acceptance |
| `total_ms` | the whole round (draft + verify) |
| `acc` / `rej` | deltas of the job-level draft counters this round |

## The companion line: `[requeue]`

Emitting one line per requeue was what pinned the accounting bug, so it is worth knowing about even
though it is a separate diagnostic:

```
[requeue] ser=0 new_tokens=6005 rq_new_tokens=0     accepted=2965 kv=6139
[requeue] ser=0 new_tokens=4096 rq_new_tokens=6005  accepted=5197 kv=10235
[requeue] ser=0 new_tokens=4096 rq_new_tokens=10101 accepted=7851 kv=14331
```

| Field | Meaning |
|---|---|
| `new_tokens` | tokens produced in the segment that is ending |
| `rq_new_tokens` | the running total carried into this segment |
| `accepted` | accepted draft tokens so far, cumulative |
| `kv` | sequence position at the requeue |

Reading it: `rq_new_tokens` must grow by the `new_tokens` of the previous line. If it does not, the
running total is being overwritten rather than accumulated — which is exactly the bug described in
[decode-drop-root-cause.md](decode-drop-root-cause.md), where the second line read
`rq_new_tokens=4096` after a `new_tokens=6005` segment.

## How to read it

- **Rounds multiplying with healthy `verify_ms`** → acceptance collapsed. Check `acc` per round.
- **`verify_ms` growing with a steady `window`** → the forward itself slowed.
- **`window` collapsing** → dynamic drafting is truncating the chain hard.
- **`total_ms` far above `draft_ms + verify_ms`** → the engine is waiting outside the loop.

## Baseline on this box

40-token generation, short fresh prompt, `window=5`, 17 rounds:

| | median | range |
|---|---:|---:|
| `draft_ms` | 9.1 | 9.1 – 27.0 |
| **`verify_ms`** | **48.1** | 44.8 – 87.9 |
| `total_ms` | 57.2 | 53.9 – 114.9 |
| accepted per round | **1.0 of 5** | — |

Verify is **84 %** of the round. Acceptance of 1-in-5 means ~2 tokens per 57 ms round, which is the
35–38 tok/s that request reported. This matches the upstream `split_time.py` figure of "52 ms at
q=5" for the verify forward, so the instrumentation is calibrated against a known number rather
than being a new unanchored measurement.

## Enabling it

```bash
EXL3_LOG_ROUNDS=1 bash exllamav3-tabby/serve-local.sh     # rounds go to stderr
```

The launcher's output should be redirected with `>>`, not `>`. Using `>` wiped 80+ requests of
history during this investigation, which is why the earlier evidence had to be reconstructed from
notes.

## Reverting

It is a pure additive patch to `exllamav3/generator/generator.py` in the `feat/hybrid-draft`
worktree; nothing is reordered, so behaviour is identical with the flag unset. Save the applied
diff first, then:

```bash
git -C ~/qwen38-exl3/exllamav3-spec diff -- exllamav3/generator/generator.py > round-log.patch
git -C ~/qwen38-exl3/exllamav3-spec checkout -- exllamav3/generator/generator.py
```

## Known cosmetic artifact

The final round of a request can log `jobs=0` with negative `acc`/`rej` deltas, because the job is
reaped inside `iterate_gen` and the counters therefore drop between the before and after reads. It
is harmless; ignore the last line of a request.
