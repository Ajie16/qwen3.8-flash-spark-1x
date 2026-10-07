# exllamav3 optimization backlog

What is left to try on the native engine, and what is already closed. Everything here is either
(a) a knob the upstream recipe never swept, (b) measured and rejected, or (c) a ceiling that no
knob can move. Written 2026-10-07, after the `EXL3_MTP_HEAD_N` and `EXL3_GR_TUNED` work.

Measure with `~/exl3bench.py` (the authoritative harness — an earlier ad-hoc bench reported
1.08M tok/s by counting a role-only placeholder chunk as the first token). Quote `chat.py` numbers
for user-visible throughput; in-process harnesses read ~4 tok/s high for kernel changes and ~10 for
host-side ones, because the per-token console write is itself a host sync.

---

## 1. Not yet swept — candidates

The upstream `tuning/` matrix covers prompt class × patch on/off × `-ndt`/`-dds`, the int8 mixer,
KV precision, context length, `EXL3_GR_RB` and mixer storage precision. **None of the following
appears in it.**

| # | Knob | Default | What it does | Why it is worth a run |
|---|---|---|---|---|
| 1 | `EXL3_MOE_TILE_N` | `0` = auto | `128` forces the N=128 MoE tile for dims that are multiples of 256, which otherwise take the N=256 instances (`exl3_moe.cu` `moe_tile_n_override`) | Both instances are **compiled**, so this is a switch between two kernels rather than a rebuild. The down projection is N=2560, a multiple of 256, so it is exactly the projection affected. The mirror-image experiment (128 → 256) was tried on the other engine and failed with `unsupported tile`, never on this one |
| 2 | `EXL3_MOE_FUSED_ROWS` | `128` | Row capacity of the fused MoE kernel's per-group temp buffers; experts with more rows assigned fall to the reconstruct path | This is the **routing point** between fused and reconstruct. The "dequant once" analysis is all about this split, and the split point itself was never swept |
| 3 | `EXL3_MOE_FUSED_ROWS_WIDE` | `256` | Same, for the wide-tile instance used when dims are multiples of 256 | Pairs with #1: the wide instance is what #1 selects away from |
| 4 | `EXL3_QC_PF_TWO_PASS_MIN_Q` | `256` | Query-length threshold for the prefill attention staging pass: below it the direct path reads less gmem (short trailing chunks over long contexts, low bitrates) | **Prefill is the known bottleneck** (~941 tok/s vs 2425 on the MLX engine). This directly selects a prefill attention path |
| 5 | `EXL3_MOE_RECON_ROWS` / `_BATCH` / `_MB` | `16384` / `16` / `256` | Padded rows per group, batch size, and dequantised weight scratch budget for the reconstruct tier | Only matters if #2/#3 push more experts into reconstruct, so sweep together |

### How to run them

One variable at a time, same session, paired against the current config. The whole point of the
paired run is that cross-session drift on this box reaches 16%:

```bash
# example: MoE tile N
EXL3_MOE_TILE_N=128 bash exllamav3-tabby/serve-local.sh     # then bench
bash exllamav3-tabby/serve-local.sh                          # then bench again, same session
```

The GPU-side process must be restarted for each arm — these are read once at import or first call.

---

## 2. Available on the worktree, not enabled

These exist in `feat/hybrid-draft` and are default-off.

| Knob | Measured | Why it is off |
|---|---|---|
| `EXL3_HYBRID_NGRAM` | exact-repeat boilerplate **+34%**; code+think, prose, decode@74k, needle **all par**; JSON record list **−10%** (the adaptive backoff bounds it — without it the same case was −57%) | A measured regression case exists. Enable only if the traffic is repeat-heavy rather than JSON-record-heavy |
| `EXL3_SPEC_SAMPLING` | **no measurement** — design stage, projected only | Mutually exclusive with hybrid; no data to justify it |
| `EXL3_SPEC_SHADOW` | observe-only, changes nothing | Pure overhead unless measuring |
| `EXL3_NGRAM_ADAPTIVE` | default on within hybrid | — |

---

## 3. Measured and closed — do not redo

From the upstream README, "Levers that are closed", all measured on the same stack at the
launcher's configuration. Reproduced here so nobody spends the days twice.

| Lever | Result |
|---|---|
| `-ndt` 6 / 7 / 8 | **Not a lever.** `-dds` already truncates the draft window by confidence. ndt 5 → 85.7 tok/s @ 4.12 tok/round @ 75.2%; 6 → 82.5 @ 3.81 @ 70.7%; 7 → 86.0 @ 4.35 @ 67.4%; 8 → 77.9 @ 4.55 @ 66.4%. Tokens per round barely move while acceptance rots |
| Row-batched fp16 GatedResidual (`EXL3_GR_RB=1`) | **−3 tok/s.** Reduction order changes, greedy trajectory shifts, acceptance 4.17 → 3.92 tok/round. Parity with the fp32 reference was identical |
| Mixer restructuring to stop re-reading streams | **Both rewrites slower at R=6.** Contracted-tiled 46.8 µs and row-blocked 47.1 µs against the shipped 44.8 µs. The traffic model is right — the slope falls — but the intercept tracks block count, and on a ~45 µs op a 48-SM part punishes starved parallelism harder than it rewards saved bandwidth. They cross over near R≈9 and R is `ndt+1 = 6` |
| CUDA-graph capture of the decode round | **Already done inside the engine.** Not a lever |
| Dequant-once MoE kernel | **Premise withdrawn.** The fused expert kernel looks linear in verify rows (0.127 ms at m=1 to 0.704 ms at m=8 for one layer) and therefore like per-row re-dequantisation, but unique experts touched scale nearly 1:1 with rows — 10, 20, 39, 56, 70 at m=1,2,4,6,8 — so the linearity is largely genuine distinct-weight traffic. Caveat recorded upstream: that measurement used random hidden states, which route near-uniformly, so the honest claim is that the *premise* was wrong, not that remaining overlap is zero |
| N-gram assist alongside MTP, in shipped code | **Mutually exclusive** — `Generator` asserts `not ngram_match_min` when a draft model is set. `EXL3_HYBRID_NGRAM` on the worktree is what lifts this |
| `EXL3_QC_PREFILL_NS` | **Not a lever — the engine self-tunes it.** `_pick_qc_prefill_num_stages` compiles and warms up both 2-stage and 1-stage on the first prefill per kernel family, then caches the winner. Upstream notes the span is −75%..+85% with no usable static rule and that 4090/5090 disagree per point |
| `EXL3_QC_STAGING=2` | A/B/debug mode (dequantise-then-attend with full-size fp16 temporaries). Default 1 is the fast path |

---

## 4. Ceilings — no knob moves these

| Ceiling | Value | Basis |
|---|---|---|
| **Speculation speedup** | **~2× cap** | Fused expert kernel cost grows with verify rows, and unique experts scale with rows on a 512-expert / top-10 MoE. So verify gets more expensive exactly as acceptance improves. This is why acceptance tuning alone cannot push past ~2× |
| Context | **262,144** | Needle exact at 240k, fails at 300k and 480k, memory wall near 480k |
| Prefill | **~941 tok/s** | EXL3 trellis dequantisation is per-token compute, not scheduling: a staging experiment that moved the chunk from 2.6k to 42k left prefill flat at ~820 on the other engine. The MLX 4-bit affine path reaches 2425 because its dequant is multiply-add and fuses into the tensor-core matmul |
| Single-stream decode, code | **85 tok/s** | Upstream ceiling, no console write |

---

## 5. Open discrepancy worth resolving

The two engines disagree on the hot-expert redundancy question that decides whether a dequant-once
MoE kernel is worth writing:

| Engine | Measurement | Claim |
|---|---|---|
| TensorFold EXL3 | in-situ census over 48 windows × 49,152 tokens of real hidden states | experts at ≥256 rows carry 34.9% of the work and are re-dequantised **31.7×** |
| exllamav3 | random hidden states, unique-expert counts vs rows | unique experts scale **~1:1** with rows (10/20/39/56/70 at m=1/2/4/6/8) |

The exllamav3 side is explicitly flagged as taken on random inputs, which route near-uniformly, so
its own author says the honest reading is "the premise for the rewrite was wrong", not "the overlap
is zero". **Neither has published an in-situ unique-expert census on real hidden states.** That
census is what would settle it, and it is the gate on a real kernel project.

---

## 6. Beyond engine knobs

The upstream README lists what the per-stream numbers do not cover. Two of those items are now
measured — see **[exllamav3-tabbyapi-benchmark.md](exllamav3-tabbyapi-benchmark.md)**:

- **TabbyAPI throughput on the current pin is measured.** Server-side, single stream, 400 tokens:
  **60.2 tok/s mean with thinking off** (73.4 peak on code) and **53.3 with thinking on** (xhigh).
  Thinking costs ~11% and the mechanism is draft acceptance, not per-token work — code acceptance
  falls 85% → 61% because reasoning text is harder for the MTP head to predict. On the real workload
  the log gives a median of **54.4 tok/s** with a 78.8 peak.
- **The concurrency picture is measured too.** A client that reported ~29 tok/s was reading the
  per-stream rate at ~4 concurrent jobs (upstream's table gives 104.4 aggregate / **26.1 per
  stream**), and the log shows the slowest requests all overlapped one 265-second generation. That
  is queue throughput, not a regression.

- **Prefix-cache hit rate is a user-visible lever larger than most kernel work.** Cache reuse is
  strict-prefix; a client that rewrites its context (compaction, or a timestamp/random id in the
  system prompt) goes fully cold, and the first request after a compaction pays a full prefill on
  a long session. That is a client behaviour, not an engine knob.

- **`thinking_budget` works here and is request-level.** `reasoning_budget_tokens` (aliases
  `reasoning_budget`, `thinking_budget`, `thinking_token_budget`) is resolved per request with a
  model-config fallback, and is implemented as tokens *injected into the output stream* — the
  reasoning-end token plus an optional message — not as a truncation. So there is no
  answer-starves-for-room failure mode. It is ignored, with a warning, when the request also uses
  `json_schema`, `regex_pattern` or `grammar_string`, because the injection would disable the
  job's filters.
