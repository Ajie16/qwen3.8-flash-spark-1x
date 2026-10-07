# Why decode rate collapses — measured answer

**Conclusion: decode tok/s is `(accepted per round + 1) / round_time`. The round time is set by the
target verify forward and is nearly independent of how many draft tokens were attempted, so
throughput is driven almost entirely by draft acceptance, which is content-dependent.**

Everything below is measured on spark-2 against the running service, with the round instrumentation
in `docs/round-logging.md`.

---

## 1. The formula

From the `[iter]` phase log, over requests of 500-700 tokens across ten content types:

```
decode tok/s = (acceptance_rate × draft_window + 1) / round_time
```

where `round_time` is dominated by the verify forward. Fitting the measured rounds:

```
round_time ≈ 30 ms + 6 ms × draft_window
```

so the window itself is nearly free relative to the forward. That is the whole mechanism: a round
costs about the same whether it verifies 1 draft token or 5, so the only thing that moves throughput
is how many of those tokens the target accepts.

## 2. Verification: acceptance predicts speed, r = +0.99

Six content types, one request each, service idle, 700 tokens:

| Content | acceptance | tok/s | median window | median round |
|---|---:|---:|---:|---:|
| zh_tech | 52.9 % | 46.2 | 1 | 35.8 ms |
| prose | 54.1 % | 47.9 | 1 | 35.4 ms |
| zh_prose | 54.6 % | 48.2 | 1 | 35.4 ms |
| code | 64.6 % | 56.1 | 2 | 40.7 ms |
| math | 68.5 % | 62.3 | 2 | 42.7 ms |
| json | 72.1 % | 64.9 | 3 | 46.2 ms |

**Pearson r = +0.99.** The formula predicts every one of the six within 35 %, most within 10 %:

| Content | measured | predicted |
|---|---:|---:|
| code | 56.1 | 56.3 |
| prose | 47.9 | 43.5 |
| math | 62.3 | 55.5 |
| json | 64.9 | 68.5 |
| zh_tech | 46.2 | 42.7 |
| zh_prose | 48.2 | 43.7 |

## 3. The same formula reaches the log's slow end

The service log's lone long requests spanned **14.8-60.3 tok/s**. The formula's floor is one token
per round, so:

| round time | tok/s at zero acceptance |
|---:|---:|
| 35 ms | 28.6 |
| 50 ms | 20.0 |
| **68 ms** | **14.7** ← the log's slowest, #44 at 14.8 |

`verify_ms` was observed up to **87.9 ms** in the same instrumentation, so a 68 ms round with
acceptance near zero is inside the measured envelope. **Caveat: a 14.8 tok/s case was not
reproduced directly.** Ten content types, including deliberately unpredictable tasks (random UUIDs,
random hex, random word lists), all held 46-65 tok/s; the unpredictable ones were not actually
unpredictable in output (acceptance stayed 56-69 %), so the low end is inferred from the formula and
the observed round-time range rather than observed end to end.

## 4. What was ruled out, and how

| Hypothesis | Measurement | Result |
|---|---|---|
| Concurrency | rebuilt the interval overlap of all 29 completed requests | the slow ones ran **alone** |
| GPU throttling | 220 s sustained load sampled every 20 s | SM clock flat **2190 MHz**, 48-61 °C, all throttle flags `Not Active`; the 43-minute `SW Power Capping` counter is historical |
| Context depth | upstream `depthab.sh` at 4k / 128k / 240k | **66.6 / 66.6 / 65.1** tok/s |
| Memory pressure | `sar` across slow (#54) and fast (#58) windows | identical, 84 % used / 16.5 GiB available |
| `EXL3_GR_TUNED` | same prompt both settings, with restarts | 45.8 vs 44.4 |
| `output_chunking` / requeue | 4 combinations of cold/long prompt × on/off | 45.9 / 47.8 / 51.6 / 53.7 |
| Client read speed | one request drained fast, one sleeping 0.25 s per chunk | server dropped only **13 %** |
| Constrained generation | plain vs `json_schema` vs grammar | **−4 % / −5 %** |
| Output length alone | synthetic 6 000-token generations | 45-54 tok/s |
| **Draft acceptance** | **10 content types, round instrumentation** | **r = +0.99 — this is it** |

Also resolved along the way: the impossible draft counters (`accepted > output`) are
`prepare_for_requeue` accumulating `accepted_draft_tokens` across requeues. It is an accounting
artefact, **not** a performance cause — the prefix cache covers the re-prefill it triggers, and
disabling chunking measured no better.

## 5. Why the window collapses

`window` falls from 5 to 1-2 a few rounds into every request. That is `EXL3_DRAFT_CONFIDENCE=0.6`
with `dynamic_draft`, truncating the chain when confidence drops. It is the engine correctly
stopping wasted draft work, and because the round costs roughly the same either way it is
approximately throughput-neutral. It is **not** a fault and should not be disabled.

## 6. The lever that follows

Acceptance is the only quantity worth moving, and the branch already contains the feature designed
for exactly this: **`EXL3_SPEC_SAMPLING`** (M2, `feat/spec-sampling`). Match-verify accepts a draft
only when the target samples the same token, so acceptance is `E_p[p(d_greedy)]`, which collapses
whenever `p` is flat. Exact speculative sampling accepts on the total-variation overlap
`E[Σ min(p, q)]` instead — strictly ≥ the match rate — while leaving the output distribution exactly
`p`. Its own design note projects prose acceptance from 45 % to 65-80 %.

The other candidate is `EXL3_HYBRID_NGRAM`, which helped only repetitive content (+34 % on
exact-repeat boilerplate, −10 % on JSON record lists in its own A/B) and is default off. Given the
table in §2 — JSON already has the *highest* acceptance at 72 % — hybrid n-gram addresses a case
this workload does not have.

**Predictable next measurement:** enable `EXL3_SPEC_SAMPLING=1`, rerun §2's six prompts, and check
whether acceptance rises and tok/s follows it. That is a falsifiable test with a stated mechanism,
not another sweep.
